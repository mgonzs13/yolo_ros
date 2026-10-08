// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/plugins/detection_plugin.hpp"

#include <algorithm>
#include <string>
#include <thread>
#include <utility>

#include "huggingface_hub.h"
#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/qos.hpp"
#include "yolo_ros/yolo/model_factory.hpp"

PLUGINLIB_EXPORT_CLASS(yolo_ros::DetectionPlugin, yolo_ros::Plugin)

namespace yolo_ros {

void DetectionPlugin::declare_params(rclcpp_lifecycle::LifecycleNode &node,
                                     const std::string &prefix) {
  node.declare_parameter<std::string>(prefix + "model_type", "auto");
  node.declare_parameter<std::string>(prefix + "model", "yolo11m_segment.onnx");
  node.declare_parameter<std::string>(prefix + "model_repo", "");
  node.declare_parameter<std::string>(prefix + "model_filename", "");
  node.declare_parameter<std::string>(prefix + "cache_dir",
                                      "~/.cache/huggingface/hub");
  node.declare_parameter<bool>(prefix + "force_download", false);
  node.declare_parameter<std::string>(prefix + "device", "cuda:0");
  node.declare_parameter<std::string>(prefix + "provider", "auto");
  node.declare_parameter<bool>(prefix + "trt_fp16_enable", true);
  node.declare_parameter<bool>(prefix + "trt_engine_cache_enable", true);
  node.declare_parameter<std::string>(prefix + "trt_engine_cache_path", "");
  node.declare_parameter<float>(prefix + "threshold", 0.7f);
  node.declare_parameter<float>(prefix + "iou", 0.45f);
  node.declare_parameter<int>(prefix + "max_det", 300);
  node.declare_parameter<int>(prefix + "n_threads", -1);
  node.declare_parameter<int>(prefix + "max_fps", 0);
  node.declare_parameter<int>(prefix + "top_k", 5);
  node.declare_parameter<int>(prefix + "max_batch_size", 8);
}

void DetectionPlugin::get_params(const rclcpp_lifecycle::LifecycleNode &node,
                                 const std::string &prefix) {
  node.get_parameter(prefix + "model_type", this->yolo_params_.model_type);
  node.get_parameter(prefix + "model", this->yolo_params_.model_path);
  node.get_parameter(prefix + "model_repo", this->yolo_params_.model_repo);
  node.get_parameter(prefix + "model_filename",
                     this->yolo_params_.model_filename);
  node.get_parameter(prefix + "cache_dir", this->yolo_params_.cache_dir);
  node.get_parameter(prefix + "force_download",
                     this->yolo_params_.force_download);
  node.get_parameter(prefix + "device", this->yolo_params_.device);
  node.get_parameter(prefix + "provider", this->yolo_params_.provider);
  node.get_parameter(prefix + "trt_fp16_enable",
                     this->yolo_params_.trt_fp16_enable);
  node.get_parameter(prefix + "trt_engine_cache_enable",
                     this->yolo_params_.trt_engine_cache_enable);
  node.get_parameter(prefix + "trt_engine_cache_path",
                     this->yolo_params_.trt_engine_cache_path);
  node.get_parameter(prefix + "threshold", this->yolo_params_.threshold);
  node.get_parameter(prefix + "iou", this->yolo_params_.iou);

  int max_det = 300;
  node.get_parameter(prefix + "max_det", max_det);
  // Clamp before the detection resize: a negative value would cast to a huge
  // std::size_t and make resize() throw std::length_error.
  this->yolo_params_.max_det = std::max(0, max_det);

  node.get_parameter(prefix + "n_threads", this->yolo_params_.n_threads);
  node.get_parameter(prefix + "max_fps", this->yolo_params_.max_fps);
  node.get_parameter(prefix + "top_k", this->yolo_params_.top_k);

  int max_batch_size = 8;
  node.get_parameter(prefix + "max_batch_size", max_batch_size);
  this->max_batch_size_ = static_cast<std::size_t>(std::max(1, max_batch_size));
}

bool DetectionPlugin::setup(PluginContext &ctx) {
  this->context_ = std::make_unique<PluginContext>(PluginContext{ctx});
  this->cameras_.clear();

  for (const auto &camera : ctx.cameras) {
    CameraStream stream;
    stream.name = camera.name;
    stream.output_channel = this->output_channel(camera.name);
    stream.reader = ctx.blackboard.subscribe<CameraFrame>(camera.name, 1);
    ctx.blackboard.declare_channel<yolo_msgs::msg::DetectionArray>(
        stream.output_channel);
    ctx.topics.expose<yolo_msgs::msg::DetectionArray>(
        stream.output_channel, stream.output_channel, rclcpp::QoS(10),
        ctx.name);
    this->cameras_.push_back(std::move(stream));
  }

  return !this->cameras_.empty();
}

bool DetectionPlugin::create_model(std::string &error) {
  if (!this->yolo_params_.model_repo.empty() &&
      !this->yolo_params_.model_filename.empty()) {
    auto result = huggingface_hub::hf_hub_download_with_shards(
        this->yolo_params_.model_repo, this->yolo_params_.model_filename,
        this->yolo_params_.cache_dir, this->yolo_params_.force_download);

    if (result.success) {
      this->yolo_params_.model_path = result.path;
      RCLCPP_INFO(this->context_->logger, "[huggingface] model %s/%s -> %s",
                  this->yolo_params_.model_repo.c_str(),
                  this->yolo_params_.model_filename.c_str(),
                  result.path.c_str());
    } else {
      RCLCPP_ERROR(this->context_->logger,
                   "[huggingface] failed to download %s/%s; falling back to "
                   "[local] %s",
                   this->yolo_params_.model_repo.c_str(),
                   this->yolo_params_.model_filename.c_str(),
                   this->yolo_params_.model_path.c_str());
    }
  }

  this->model_ = yolo_ros::yolo::create_model(this->yolo_params_);

  if (!this->model_) {
    error = "failed to create the YOLO model";
    return false;
  }

  return true;
}

bool DetectionPlugin::activate() {
  std::string error;

  if (!this->create_model(error)) {
    RCLCPP_ERROR(this->context_->logger, "%s", error.c_str());
    return false;
  }

  if (this->cameras_.size() > 1) {
    this->scheduler_ = std::make_unique<engine::BatchScheduler<PendingFrame>>(
        this->cameras_.size(), this->max_batch_size_);
    this->scheduler_->start(
        [this](const engine::BatchScheduler<PendingFrame>::Batch &batch) {
          this->process_batch(batch);
        });
  }

  return true;
}

void DetectionPlugin::deactivate() {
  if (this->scheduler_) {
    this->scheduler_->stop();
    this->scheduler_.reset();
  }

  this->model_.reset();
}

void DetectionPlugin::run(const std::atomic<bool> &stop) {
  if (this->cameras_.size() == 1) {
    auto &camera = this->cameras_.front();

    while (!stop.load()) {
      std::shared_ptr<const CameraFrame> msg;

      if (!camera.reader.wait(msg, std::chrono::milliseconds(100))) {
        continue;
      }

      if (this->yolo_params_.max_fps > 0) {
        const auto now = std::chrono::steady_clock::now();
        const double period = 1.0 / this->yolo_params_.max_fps;

        if (std::chrono::duration<double>(now - this->last_frame_).count() <
            period) {
          continue;
        }

        this->last_frame_ = now;
      }

      this->publish(camera.output_channel, this->detect_and_filter(*msg));
    }

    return;
  }

  while (!stop.load()) {
    for (std::size_t i = 0; i < this->cameras_.size(); ++i) {
      auto &camera = this->cameras_[i];
      std::shared_ptr<const CameraFrame> msg;

      while (camera.reader.try_pop(msg)) {
        // latest-frame-wins: keep only the last queued frame
      }

      if (!msg) {
        continue;
      }

      if (this->yolo_params_.max_fps > 0) {
        const auto now = std::chrono::steady_clock::now();
        const double period = 1.0 / this->yolo_params_.max_fps;

        if (std::chrono::duration<double>(now - this->last_frame_).count() <
            period) {
          continue;
        }

        this->last_frame_ = now;
      }

      this->scheduler_->push(i, std::move(msg));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
}

yolo_msgs::msg::DetectionArray
DetectionPlugin::detect_and_filter(const CameraFrame &frame) {
  yolo_msgs::msg::DetectionArray output;
  output.header = frame.header;

  if (!this->model_) {
    return output;
  }

  const cv::Mat image = frame.bgr8();

  if (image.empty()) {
    RCLCPP_ERROR(this->context_->logger, "cv_bridge exception: empty image");
    return output;
  }

  std::vector<yolo_msgs::msg::Detection> detections;

  try {
    detections = this->model_->detect(image);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(this->context_->logger, "inference failed: %s", e.what());
    return output;
  }

  if (static_cast<int>(detections.size()) > this->yolo_params_.max_det) {
    detections.resize(static_cast<std::size_t>(this->yolo_params_.max_det));
  }

  output.detections = std::move(detections);
  return output;
}

void DetectionPlugin::process_batch(
    const engine::BatchScheduler<PendingFrame>::Batch &batch) {
  if (batch.empty() || !this->model_) {
    return;
  }

  std::vector<std::pair<std::size_t, PendingFrame>> decoded;
  std::vector<cv::Mat> images;
  decoded.reserve(batch.size());
  images.reserve(batch.size());

  for (const auto &item : batch) {
    const cv::Mat image = item.second->bgr8();

    if (image.empty()) {
      RCLCPP_ERROR(this->context_->logger, "cv_bridge exception: empty image");
      continue;
    }

    images.push_back(image);
    decoded.push_back(item);
  }

  if (images.empty()) {
    return;
  }

  std::vector<std::vector<yolo_msgs::msg::Detection>> results;

  try {
    results = this->model_->detect_batch(images);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(this->context_->logger, "batch inference failed: %s",
                 e.what());
    return;
  }

  for (std::size_t i = 0; i < decoded.size() && i < results.size(); ++i) {
    auto detections = std::move(results[i]);

    if (static_cast<int>(detections.size()) > this->yolo_params_.max_det) {
      detections.resize(static_cast<std::size_t>(this->yolo_params_.max_det));
    }

    yolo_msgs::msg::DetectionArray array;
    array.header = decoded[i].second->header;
    array.detections = std::move(detections);
    this->publish(this->cameras_[decoded[i].first].output_channel, array);
  }
}

void DetectionPlugin::publish(const std::string &channel,
                              const yolo_msgs::msg::DetectionArray &message) {
  this->context_->blackboard.publish<yolo_msgs::msg::DetectionArray>(
      channel, std::make_shared<yolo_msgs::msg::DetectionArray>(message));
}

} // namespace yolo_ros
