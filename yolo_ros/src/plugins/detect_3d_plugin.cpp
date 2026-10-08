// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/plugins/detect_3d_plugin.hpp"

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/qos.hpp"

#if __has_include("tf2/exceptions.hpp")
#include "tf2/exceptions.hpp"
#else
#include "tf2/exceptions.h"
#endif

#include "tf2/time.h"

namespace yolo_ros {

void Detect3DPlugin::declare_params(rclcpp_lifecycle::LifecycleNode &node,
                                    const std::string &prefix) {
  node.declare_parameter<std::string>(prefix + "target_frame", "base_link");
  node.declare_parameter<int>(prefix + "depth_image_units_divisor", 1000);
  node.declare_parameter<bool>(prefix + "enable_orientation", false);
  node.declare_parameter<int>(prefix + "min_seg_points_for_orientation", 20);
}

void Detect3DPlugin::get_params(const rclcpp_lifecycle::LifecycleNode &node,
                                const std::string &prefix) {
  node.get_parameter(prefix + "target_frame", this->target_frame_);
  node.get_parameter(prefix + "depth_image_units_divisor",
                     this->depth_image_units_divisor_);
  node.get_parameter(prefix + "enable_orientation", this->enable_orientation_);
  node.get_parameter(prefix + "min_seg_points_for_orientation",
                     this->min_seg_points_for_orientation_);
}

bool Detect3DPlugin::setup(PluginContext &ctx) {
  this->context_ = std::make_unique<PluginContext>(PluginContext{ctx});
  this->cameras_.clear();

  for (const auto &camera : ctx.cameras) {
    if (!camera.has_depth) {
      RCLCPP_ERROR(ctx.logger, "plugin '%s': camera '%s' has no depth",
                   ctx.name.c_str(), camera.name.c_str());
      return false;
    }

    CameraStream stream;
    stream.name = camera.name;
    stream.output_channel = this->output_channel(camera.name);
    stream.sync = std::make_unique<
        ChannelSync<CameraFrame, yolo_msgs::msg::DetectionArray>>(
        ctx.blackboard,
        std::vector<std::string>{camera.frame_channel, camera.input_channel},
        10);
    ctx.blackboard.declare_channel<yolo_msgs::msg::DetectionArray>(
        stream.output_channel);
    ctx.topics.expose<yolo_msgs::msg::DetectionArray>(
        stream.output_channel, stream.output_channel, rclcpp::QoS(10),
        ctx.name);
    this->cameras_.push_back(std::move(stream));
  }

  return !this->cameras_.empty();
}

bool Detect3DPlugin::activate() { return true; }

void Detect3DPlugin::deactivate() {}

void Detect3DPlugin::run(const std::atomic<bool> &stop) {
  while (!stop.load()) {
    bool processed_any = false;

    for (auto &camera : this->cameras_) {
      ChannelSync<CameraFrame, yolo_msgs::msg::DetectionArray>::Result matched;

      // Non-blocking single drain per camera: blocking here would add the
      // per-camera timeout to every sweep (N cameras x ~10 ms).
      if (!camera.sync->next(matched, std::chrono::milliseconds(0))) {
        continue;
      }

      processed_any = true;
      const auto &frame = std::get<0>(matched);
      const auto &detections_msg = std::get<1>(matched);

      yolo_msgs::msg::DetectionArray new_detections_msg;
      new_detections_msg.header = detections_msg->header;
      new_detections_msg.detections = this->process_detections(
          frame, detections_msg, this->orientation_states_[camera.name]);
      this->context_->blackboard.publish<yolo_msgs::msg::DetectionArray>(
          camera.output_channel,
          std::make_shared<yolo_msgs::msg::DetectionArray>(new_detections_msg));
    }

    // One idle wait per full sweep keeps the latency independent of the
    // camera count; a sweep that matched skips it to drain the backlog.
    if (!processed_any) {
      this->context_->blackboard.wait_for_activity(
          std::chrono::milliseconds(10));
    }
  }
}

std::vector<yolo_msgs::msg::Detection> Detect3DPlugin::process_detections(
    const std::shared_ptr<const CameraFrame> &frame,
    const yolo_msgs::msg::DetectionArray::ConstSharedPtr &detections_msg,
    yolo_ros::depth::OrientationState &orientation_state) {
  std::vector<yolo_msgs::msg::Detection> new_detections;

  if (detections_msg->detections.empty()) {
    return new_detections;
  }

  // Passthrough depth view preserving the source encoding: 16UC1 raw units
  // (depth_utils divides them by depth_image_units_divisor below) or 32FC1
  // metres (already metric, so depth_utils ignores the divisor), like the
  // Python node's "passthrough" encoding.
  const cv::Mat depth_image = frame->depth_image();

  if (depth_image.empty() || !frame->depth_info) {
    return new_detections;
  }

  auto transform = this->get_transform(frame->header.frame_id);

  if (!transform) {
    return new_detections;
  }

  for (const auto &detection : detections_msg->detections) {
    auto bbox3d = yolo_ros::depth::convert_bb_to_3d(
        depth_image, *frame->depth_info, detection,
        this->depth_image_units_divisor_,
        {this->enable_orientation_, this->min_seg_points_for_orientation_},
        &orientation_state);

    if (!bbox3d) {
      continue;
    }

    yolo_msgs::msg::Detection new_detection = detection;
    new_detection.bbox3d = yolo_ros::depth::transform_3d_box(
        *bbox3d, transform->first, transform->second);
    new_detection.bbox3d.frame_id = this->target_frame_;
    new_detections.push_back(new_detection);

    if (!detection.keypoints.data.empty()) {
      auto keypoints3d = yolo_ros::depth::convert_keypoints_to_3d(
          depth_image, *frame->depth_info, detection,
          this->depth_image_units_divisor_);
      keypoints3d = yolo_ros::depth::transform_3d_keypoints(
          keypoints3d, transform->first, transform->second);
      keypoints3d.frame_id = this->target_frame_;
      new_detections.back().keypoints3d = keypoints3d;
    }
  }

  return new_detections;
}

std::optional<std::pair<std::array<double, 3>, std::array<double, 4>>>
Detect3DPlugin::get_transform(const std::string &frame_id) {
  if (this->context_ == nullptr || this->context_->tf_buffer == nullptr) {
    return std::nullopt;
  }

  try {
    // Zero time = latest available transform (same as the Python node).
    const auto transform = this->context_->tf_buffer->lookupTransform(
        this->target_frame_, frame_id, tf2::TimePointZero);

    std::array<double, 3> translation{transform.transform.translation.x,
                                      transform.transform.translation.y,
                                      transform.transform.translation.z};
    std::array<double, 4> rotation{
        transform.transform.rotation.w, transform.transform.rotation.x,
        transform.transform.rotation.y, transform.transform.rotation.z};

    return std::make_pair(translation, rotation);
  } catch (const tf2::TransformException &ex) {
    RCLCPP_ERROR(this->context_->logger, "Could not transform: %s", ex.what());
    return std::nullopt;
  }
}

} // namespace yolo_ros

PLUGINLIB_EXPORT_CLASS(yolo_ros::Detect3DPlugin, yolo_ros::Plugin)
