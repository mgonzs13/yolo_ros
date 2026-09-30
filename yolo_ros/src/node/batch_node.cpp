// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include "yolo_ros/node/batch_node.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <utility>

#include <opencv2/opencv.hpp>

#if defined(CV_BRIDGE_H)
#include <cv_bridge/cv_bridge.h>
#else
#include <cv_bridge/cv_bridge.hpp>
#endif
#include "huggingface_hub.h"
#include "yolo_ros/yolo/model_factory.hpp"

namespace yolo_ros::node {
namespace {

rclcpp::ReliabilityPolicy reliability_from(int value) {
  if (value == 0) {
    return rclcpp::ReliabilityPolicy::SystemDefault;
  }
  if (value == 1) {
    return rclcpp::ReliabilityPolicy::Reliable;
  }
  return rclcpp::ReliabilityPolicy::BestEffort;
}

} // namespace

BatchNode::BatchNode() : rclcpp_lifecycle::LifecycleNode("yolo_batch_node") {}

BatchNode::~BatchNode() { this->teardown(); }

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
BatchNode::on_configure(const rclcpp_lifecycle::State &) {
  if (!this->params_declared_) {
    this->declare_params();
    this->params_declared_ = true;
  }
  if (!this->load_params()) {
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
        CallbackReturn::FAILURE;
  }
  RCLCPP_INFO(get_logger(), "[%s] Configured", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
BatchNode::on_activate(const rclcpp_lifecycle::State &) {
  const std::size_t count = this->camera_names_.size();
  this->yolo_model_ = yolo_ros::yolo::create_model(this->yolo_params_);
  this->enable_inference_.store(this->yolo_params_.enable);

  const auto image_qos = rclcpp::QoS(1).reliability(
      reliability_from(this->yolo_params_.image_reliability));

  this->image_subscriptions_.clear();
  this->detection_publishers_.clear();
  for (std::size_t i = 0; i < count; ++i) {
    this->detection_publishers_.push_back(
        this->create_publisher<yolo_msgs::msg::DetectionArray>(
            this->camera_names_[i] + "/detections", 10));
    this->image_subscriptions_.push_back(
        this->create_subscription<sensor_msgs::msg::Image>(
            this->image_topics_[i], image_qos,
            [this, i](sensor_msgs::msg::Image::SharedPtr msg) {
              this->image_callback(i, std::move(msg));
            }));
  }

  this->enable_service_ = this->create_service<std_srvs::srv::SetBool>(
      "enable", std::bind(&BatchNode::enable_service_callback, this,
                          std::placeholders::_1, std::placeholders::_2));
  this->set_classes_service_ = this->create_service<yolo_msgs::srv::SetClasses>(
      "set_classes", std::bind(&BatchNode::set_classes_callback, this,
                               std::placeholders::_1, std::placeholders::_2));

  this->scheduler_ =
      std::make_unique<yolo_ros::engine::BatchScheduler<PendingFrame>>(
          count, this->max_batch_size_);
  this->scheduler_->start(
      [this](std::vector<std::pair<std::size_t, PendingFrame>> batch) {
        this->process_batch(std::move(batch));
      });

  // Reset the inference counters and start the fixed 5 s stats reporter.
  {
    std::lock_guard<std::mutex> lock(this->stats_mutex_);
    this->stats_ = Stats{};
    this->stats_.per_camera.assign(count, 0);
    this->stats_.interval_start = std::chrono::steady_clock::now();
  }
  this->stats_timer_ = this->create_wall_timer(
      std::chrono::seconds(5), std::bind(&BatchNode::report_stats, this));

  RCLCPP_INFO(get_logger(), "[%s] Activated with %zu camera(s), max batch %zu",
              this->get_name(), count, this->max_batch_size_);
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
BatchNode::on_deactivate(const rclcpp_lifecycle::State &) {
  this->teardown();
  RCLCPP_INFO(get_logger(), "[%s] Deactivated", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
BatchNode::on_cleanup(const rclcpp_lifecycle::State &) {
  this->teardown();
  RCLCPP_INFO(get_logger(), "[%s] Cleaned up", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
BatchNode::on_shutdown(const rclcpp_lifecycle::State &) {
  this->teardown();
  RCLCPP_INFO(get_logger(), "[%s] Shutting down", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

void BatchNode::teardown() {
  this->stats_timer_.reset();
  if (this->scheduler_) {
    this->scheduler_->stop();
    this->scheduler_.reset();
  }
  this->yolo_model_.reset();
  this->enable_service_.reset();
  this->set_classes_service_.reset();
  this->image_subscriptions_.clear();
  this->detection_publishers_.clear();
}

void BatchNode::declare_params() {
  this->declare_parameter<std::string>("model_type", "auto");
  this->declare_parameter<std::string>("model", "yolo11m_segment.onnx");
  this->declare_parameter<std::string>("model_repo", "");
  this->declare_parameter<std::string>("model_filename", "");
  this->declare_parameter<std::string>("cache_dir", "~/.cache/huggingface/hub");
  this->declare_parameter<bool>("force_download", false);
  this->declare_parameter<std::string>("device", "cuda:0");
  this->declare_parameter<std::string>("provider", "auto");
  this->declare_parameter<bool>("trt_fp16_enable", true);
  this->declare_parameter<bool>("trt_engine_cache_enable", true);
  this->declare_parameter<std::string>("trt_engine_cache_path", "");
  this->declare_parameter<float>("threshold", 0.7);
  this->declare_parameter<float>("iou", 0.45);
  this->declare_parameter<int>("max_det", 300);
  this->declare_parameter<bool>("enable", true);
  this->declare_parameter<int>("image_reliability", 2);
  this->declare_parameter<int>("n_threads", -1);
  this->declare_parameter<int>("max_fps", 0);
  this->declare_parameter<int>("top_k", 5);
  this->declare_parameter<std::vector<std::string>>("camera_names",
                                                    std::vector<std::string>{});
  this->declare_parameter<std::vector<std::string>>("image_topics",
                                                    std::vector<std::string>{});
  this->declare_parameter<int>("max_batch_size", 8);
}

bool BatchNode::load_params() {
  this->get_parameter("model_type", this->yolo_params_.model_type);
  this->get_parameter("model", this->yolo_params_.model_path);
  this->get_parameter("model_repo", this->yolo_params_.model_repo);
  this->get_parameter("model_filename", this->yolo_params_.model_filename);
  this->get_parameter("cache_dir", this->yolo_params_.cache_dir);
  this->get_parameter("force_download", this->yolo_params_.force_download);
  this->get_parameter("device", this->yolo_params_.device);
  this->get_parameter("provider", this->yolo_params_.provider);
  this->get_parameter("trt_fp16_enable", this->yolo_params_.trt_fp16_enable);
  this->get_parameter("trt_engine_cache_enable",
                      this->yolo_params_.trt_engine_cache_enable);
  this->get_parameter("trt_engine_cache_path",
                      this->yolo_params_.trt_engine_cache_path);
  this->get_parameter("threshold", this->yolo_params_.threshold);
  this->get_parameter("iou", this->yolo_params_.iou);
  this->get_parameter("max_det", this->yolo_params_.max_det);
  this->get_parameter("enable", this->yolo_params_.enable);
  this->get_parameter("image_reliability",
                      this->yolo_params_.image_reliability);
  this->get_parameter("n_threads", this->yolo_params_.n_threads);
  this->get_parameter("max_fps", this->yolo_params_.max_fps);
  this->get_parameter("top_k", this->yolo_params_.top_k);
  this->get_parameter("camera_names", this->camera_names_);
  this->get_parameter("image_topics", this->image_topics_);
  int max_batch = 8;
  this->get_parameter("max_batch_size", max_batch);
  this->max_batch_size_ = static_cast<std::size_t>(std::max(1, max_batch));

  if (this->camera_names_.size() != this->image_topics_.size()) {
    RCLCPP_ERROR(get_logger(),
                 "camera_names and image_topics must have the same length");
    return false;
  }
  // Hugging Face Hub: keep the same precedence as yolo_node.
  if (!this->yolo_params_.model_repo.empty() &&
      !this->yolo_params_.model_filename.empty()) {
    auto result = huggingface_hub::hf_hub_download_with_shards(
        this->yolo_params_.model_repo, this->yolo_params_.model_filename,
        this->yolo_params_.cache_dir, this->yolo_params_.force_download);
    if (result.success) {
      this->yolo_params_.model_path = result.path;
    } else {
      RCLCPP_ERROR(get_logger(),
                   "[huggingface] failed to download %s/%s; using %s",
                   this->yolo_params_.model_repo.c_str(),
                   this->yolo_params_.model_filename.c_str(),
                   this->yolo_params_.model_path.c_str());
    }
  }
  return true;
}

void BatchNode::image_callback(std::size_t camera,
                               const sensor_msgs::msg::Image::SharedPtr msg) {
  if (!this->enable_inference_.load()) {
    return;
  }
  if (this->yolo_params_.max_fps > 0) {
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - this->last_inference_time_)
            .count() < 1.0 / this->yolo_params_.max_fps) {
      return;
    }
    this->last_inference_time_ = now;
  }
  PendingFrame frame;
  frame.image = msg;
  this->scheduler_->push(camera, std::move(frame));
}

void BatchNode::process_batch(
    std::vector<std::pair<std::size_t, PendingFrame>> batch) {
  if (batch.empty() || !this->yolo_model_ || !this->enable_inference_.load()) {
    return;
  }
  std::vector<std::pair<std::size_t, PendingFrame>> decoded;
  std::vector<cv::Mat> images;
  decoded.reserve(batch.size());
  images.reserve(batch.size());
  for (auto &item : batch) {
    try {
      images.push_back(cv_bridge::toCvCopy(item.second.image,
                                           sensor_msgs::image_encodings::BGR8)
                           ->image);
    } catch (const cv_bridge::Exception &e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
      continue;
    }
    decoded.push_back(std::move(item));
  }
  if (images.empty()) {
    return;
  }
  std::vector<std::vector<yolo_msgs::msg::Detection>> results;
  const auto inference_start = std::chrono::steady_clock::now();
  try {
    results = this->yolo_model_->detect_batch(images);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(get_logger(), "batch inference failed: %s", e.what());
    return;
  }
  const double inference_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - inference_start)
          .count();
  for (std::size_t i = 0; i < decoded.size() && i < results.size(); ++i) {
    auto detections = std::move(results[i]);
    {
      std::lock_guard<std::mutex> lock(this->classes_mutex_);
      if (!this->allowed_classes_.empty()) {
        detections.erase(
            std::remove_if(detections.begin(), detections.end(),
                           [this](const yolo_msgs::msg::Detection &detection) {
                             return this->allowed_classes_.count(
                                        detection.class_name) == 0;
                           }),
            detections.end());
      }
    }
    if (static_cast<int>(detections.size()) > this->yolo_params_.max_det) {
      detections.resize(static_cast<std::size_t>(this->yolo_params_.max_det));
    }
    yolo_msgs::msg::DetectionArray array;
    array.header = decoded[i].second.image->header;
    array.detections = std::move(detections);
    this->detection_publishers_[decoded[i].first]->publish(array);
  }

  const std::size_t processed = std::min(decoded.size(), results.size());
  {
    std::lock_guard<std::mutex> lock(this->stats_mutex_);
    this->stats_.processed_total += processed;
    this->stats_.interval_images += processed;
    this->stats_.interval_batches += 1;
    this->stats_.interval_inference_ms += inference_ms;
    for (std::size_t i = 0; i < processed; ++i) {
      const std::size_t camera = decoded[i].first;
      if (camera < this->stats_.per_camera.size()) {
        this->stats_.per_camera[camera] += 1;
      }
    }
  }
}

void BatchNode::report_stats() {
  Stats snapshot;
  double elapsed = 0.0;
  {
    std::lock_guard<std::mutex> lock(this->stats_mutex_);
    snapshot = this->stats_;
    const auto now = std::chrono::steady_clock::now();
    elapsed = std::chrono::duration<double>(now - this->stats_.interval_start)
                  .count();
    this->stats_.interval_images = 0;
    this->stats_.interval_batches = 0;
    this->stats_.interval_inference_ms = 0.0;
    this->stats_.interval_start = now;
    std::fill(this->stats_.per_camera.begin(), this->stats_.per_camera.end(),
              0);
  }

  const double hz = elapsed > 0.0 ? snapshot.interval_images / elapsed : 0.0;
  const double avg_batch =
      snapshot.interval_batches > 0
          ? static_cast<double>(snapshot.interval_images) /
                static_cast<double>(snapshot.interval_batches)
          : 0.0;
  const double avg_inference_ms =
      snapshot.interval_batches > 0
          ? snapshot.interval_inference_ms /
                static_cast<double>(snapshot.interval_batches)
          : 0.0;

  std::string cameras;
  for (std::size_t i = 0; i < snapshot.per_camera.size(); ++i) {
    if (i > 0) {
      cameras += " ";
    }
    cameras += this->camera_names_[i];
    cameras += "=+";
    cameras += std::to_string(snapshot.per_camera[i]);
  }

  RCLCPP_INFO(get_logger(),
              "[%s] stats: processed=%llu (+%llu, %.1f Hz) batch=%.2f (max "
              "%zu) infer=%.1f ms | %s",
              this->get_name(),
              static_cast<unsigned long long>(snapshot.processed_total),
              static_cast<unsigned long long>(snapshot.interval_images), hz,
              avg_batch, this->max_batch_size_, avg_inference_ms,
              cameras.c_str());
}

void BatchNode::enable_service_callback(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
  this->enable_inference_.store(request->data);
  response->success = true;
  RCLCPP_INFO(get_logger(), "[%s] inference %s", this->get_name(),
              request->data ? "enabled" : "disabled");
}

void BatchNode::set_classes_callback(
    const std::shared_ptr<yolo_msgs::srv::SetClasses::Request> request,
    std::shared_ptr<yolo_msgs::srv::SetClasses::Response> response) {
  std::set<std::string> classes;
  for (const auto &name : request->classes) {
    if (!name.empty()) {
      classes.insert(name);
    }
  }
  {
    std::lock_guard<std::mutex> lock(this->classes_mutex_);
    this->allowed_classes_ = classes;
  }
  response->success = true;
  response->message =
      classes.empty()
          ? "publishing all classes"
          : "publishing " + std::to_string(classes.size()) + " class(es)";
  RCLCPP_INFO(get_logger(), "[%s] set_classes: %s", this->get_name(),
              response->message.c_str());
}

} // namespace yolo_ros::node
