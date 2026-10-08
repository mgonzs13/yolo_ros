// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/plugins/tracking_plugin.hpp"

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <opencv2/core.hpp>

#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/qos.hpp"
#include "yolo_ros/tracking/bot_sort.hpp"
#include "yolo_ros/tracking/byte_tracker.hpp"
#include "yolo_ros/utils/string_utils.hpp"

namespace yolo_ros {

void TrackingPlugin::declare_params(rclcpp_lifecycle::LifecycleNode &node,
                                    const std::string &prefix) {
  // Statically typed parameters survive cleanup(), so on a reconfigure the
  // declaration must be skipped for the previously declared tracker_type;
  // dispatch below then reads the current value and declares the selected
  // tracker's knobs (unselected/previous ones are left untouched).
  if (!node.has_parameter(prefix + "tracker_type")) {
    node.declare_parameter<std::string>(prefix + "tracker_type", "bytetrack");
  }

  // Tracker-specific knobs are declared only for the selected tracker. The
  // templated bridges take the same node and prefix, so each tracker keeps its
  // own namespaced parameter set.
  const auto tracker_type = yolo_ros::utils::to_lower(
      node.get_parameter(prefix + "tracker_type").as_string());

  if (tracker_type == "bytetrack") {
    tracking::declare_byte_track_params(node, prefix);
  } else if (tracker_type == "botsort") {
    tracking::declare_bot_sort_params(node, prefix);
  }
}

void TrackingPlugin::get_params(const rclcpp_lifecycle::LifecycleNode &node,
                                const std::string &prefix) {
  this->tracker_params_.reset();
  this->tracker_type_ = yolo_ros::utils::to_lower(
      node.get_parameter(prefix + "tracker_type").as_string());

  if (this->tracker_type_ == "bytetrack") {
    this->tracker_params_ = std::make_shared<tracking::ByteTrackParams>(
        tracking::load_byte_track_params(node, prefix));
  } else if (this->tracker_type_ == "botsort") {
    this->tracker_params_ = std::make_shared<tracking::BotSortParams>(
        tracking::load_bot_sort_params(node, prefix));
  }
}

bool TrackingPlugin::setup(PluginContext &ctx) {
  this->context_ = std::make_unique<PluginContext>(PluginContext{ctx});
  this->cameras_.clear();

  for (const auto &camera : ctx.cameras) {
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

bool TrackingPlugin::activate() {
  if (!this->tracker_params_) {
    RCLCPP_WARN(this->context_->logger,
                "Unknown tracker_type '%s'; detections pass through",
                this->tracker_type_.c_str());
    return true;
  }

  for (auto &camera : this->cameras_) {
    camera.tracker = tracking::create_tracker(*this->tracker_params_);

    if (!camera.tracker) {
      RCLCPP_ERROR(this->context_->logger, "Failed to create tracker '%s'",
                   this->tracker_type_.c_str());
      return false;
    }
  }

  return true;
}

void TrackingPlugin::deactivate() {
  for (auto &camera : this->cameras_) {
    camera.tracker.reset();
  }
}

void TrackingPlugin::run(const std::atomic<bool> &stop) {
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
      this->process(camera, std::get<0>(matched), std::get<1>(matched));
    }

    // One idle wait per full sweep keeps the latency independent of the
    // camera count; a sweep that matched skips it to drain the backlog.
    if (!processed_any) {
      this->context_->blackboard.wait_for_activity(
          std::chrono::milliseconds(10));
    }
  }
}

void TrackingPlugin::process(
    CameraStream &camera, const std::shared_ptr<const CameraFrame> &frame,
    const yolo_msgs::msg::DetectionArray::ConstSharedPtr &detections) {
  yolo_msgs::msg::DetectionArray tracked_msg;
  tracked_msg.header = frame->header;

  // Safety fallback: no tracker (unknown `tracker_type`). Keep the pipeline
  // alive by forwarding the input detections untouched — downstream consumers
  // only lose the track ids, nothing crashes.
  if (camera.tracker == nullptr) {
    RCLCPP_WARN_ONCE(this->context_->logger,
                     "No active tracker; forwarding detections without track "
                     "ids");
    this->context_->blackboard.publish<yolo_msgs::msg::DetectionArray>(
        camera.output_channel, detections);
    return;
  }

  // Convert the DetectionArray into the tracker's plain input format. The
  // detection index is preserved so the original Detection (class name, mask,
  // keypoints, ...) can be fetched back after tracking.
  std::vector<tracking::TrackDetection> dets;
  dets.reserve(detections->detections.size());

  for (std::size_t i = 0; i < detections->detections.size(); ++i) {
    const auto &det = detections->detections[i];

    if (det.bbox.size.x <= 0 || det.bbox.size.y <= 0) {
      continue;
    }

    tracking::TrackDetection td;
    td.cx = det.bbox.center.position.x;
    td.cy = det.bbox.center.position.y;
    td.w = det.bbox.size.x;
    td.h = det.bbox.size.y;
    td.score = det.score;
    td.class_id = det.class_id;
    td.index = static_cast<int>(i);
    dets.push_back(td);
  }

  // Camera-motion compensation needs the current frame; decode it only when
  // the active tracker actually consumes it (BoT-SORT with gmc_method !=
  // none).
  cv::Mat image;

  if (camera.tracker->needs_frame()) {
    image = frame->bgr8();

    if (image.empty()) {
      RCLCPP_WARN_ONCE(this->context_->logger,
                       "Frame conversion failed; running the tracker without "
                       "camera-motion compensation or appearance features");
    }
  }

  const auto tracks = camera.tracker->update(dets, image);

  for (const auto &track : tracks) {
    if (track.index < 0 ||
        track.index >= static_cast<int>(detections->detections.size())) {
      continue;
    }

    // Copy the original detection (keeps class name, mask, etc.) and overlay
    // the Kalman-refined box and the stable track id.
    auto tracked_detection = detections->detections[track.index];
    tracked_detection.bbox.center.position.x = (track.x1 + track.x2) / 2;
    tracked_detection.bbox.center.position.y = (track.y1 + track.y2) / 2;
    tracked_detection.bbox.size.x = track.x2 - track.x1;
    tracked_detection.bbox.size.y = track.y2 - track.y1;
    tracked_detection.id = std::to_string(track.id);

    tracked_msg.detections.push_back(tracked_detection);
  }

  this->context_->blackboard.publish<yolo_msgs::msg::DetectionArray>(
      camera.output_channel,
      std::make_shared<yolo_msgs::msg::DetectionArray>(tracked_msg));
}

} // namespace yolo_ros

PLUGINLIB_EXPORT_CLASS(yolo_ros::TrackingPlugin, yolo_ros::Plugin)
