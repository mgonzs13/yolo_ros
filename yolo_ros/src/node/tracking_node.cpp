// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2023-2025 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/node/tracking_node.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#if defined(CV_BRIDGE_H)
#include <cv_bridge/cv_bridge.h>
#else
#include <cv_bridge/cv_bridge.hpp>
#endif

#include "rclcpp/exceptions.hpp"
#include "yolo_ros/tracking/bot_sort.hpp"
#include "yolo_ros/tracking/byte_tracker.hpp"
#include "yolo_ros/utils/string_utils.hpp"

namespace yolo_ros::node {

TrackingNode::TrackingNode()
    : rclcpp_lifecycle::LifecycleNode("tracking_node"), image_qos_profile_(1) {
  this->declare_params();
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
TrackingNode::on_configure(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "[%s] Configuring...", this->get_name());

  this->load_params();

  this->image_topic_ = this->get_parameter("image_topic").as_string();

  int image_reliability = this->get_parameter("image_reliability").as_int();
  rclcpp::ReliabilityPolicy qos_reliability_policy;
  if (image_reliability == 0) {
    qos_reliability_policy = rclcpp::ReliabilityPolicy::SystemDefault;
  } else if (image_reliability == 1) {
    qos_reliability_policy = rclcpp::ReliabilityPolicy::Reliable;
  } else {
    qos_reliability_policy = rclcpp::ReliabilityPolicy::BestEffort;
  }
  this->image_qos_profile_ = rclcpp::QoS(1)
                                 .reliability(qos_reliability_policy)
                                 .durability_volatile()
                                 .keep_last(1);

  this->tracking_publisher_ =
      this->create_publisher<yolo_msgs::msg::DetectionArray>("tracking", 10);

  RCLCPP_INFO(get_logger(), "[%s] Configured", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
TrackingNode::on_activate(const rclcpp_lifecycle::State &) {
  this->image_subscription_.subscribe(this->shared_from_this(),
                                      this->image_topic_, image_qos_profile_);
  this->detection_subscription_.subscribe(this->shared_from_this(),
                                          "detections", image_qos_profile_);

  this->synchronizer_ =
      std::make_shared<message_filters::Synchronizer<TrackingSyncPolicy>>(10);
  this->synchronizer_->connectInput(this->image_subscription_,
                                    this->detection_subscription_);
  this->synchronizer_->registerCallback(
      std::bind(&TrackingNode::recieve_callback, this, std::placeholders::_1,
                std::placeholders::_2));

  RCLCPP_INFO(get_logger(), "[%s] Activated", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
TrackingNode::on_deactivate(const rclcpp_lifecycle::State &) {
  this->detection_subscription_.unsubscribe();
  this->image_subscription_.unsubscribe();
  this->synchronizer_.reset();
  RCLCPP_INFO(get_logger(), "[%s] Deactivated", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
TrackingNode::on_cleanup(const rclcpp_lifecycle::State &) {
  this->tracker_.reset();
  this->tracking_publisher_.reset();
  RCLCPP_INFO(get_logger(), "[%s] Cleaned up", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
TrackingNode::on_shutdown(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "[%s] Shutting down", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

void TrackingNode::declare_params() {
  // Generic tracking node parameters (algorithm-independent).
  this->declare_parameter<int>("image_reliability", 2);
  this->declare_parameter<std::string>("image_topic", "image");

  // Tracker selection: the key of the tracker implementation to run
  // ("bytetrack" is the built-in default; unknown values disable tracking —
  // the node forwards detections unchanged — instead of crashing). Each
  // tracker declares and loads its own parameters through its specific
  // functions, so the node only dispatches by this key.
  this->declare_parameter<std::string>("tracker_type", "bytetrack");
  const std::string tracker_type = yolo_ros::utils::to_lower(
      this->get_parameter("tracker_type").as_string());

  if (tracker_type == "bytetrack") {
    yolo_ros::tracking::declare_byte_track_params(*this);
  } else if (tracker_type == "botsort") {
    yolo_ros::tracking::declare_bot_sort_params(*this);
  }
  // --- add new trackers here: declare their parameters when selected ---
}

void TrackingNode::load_params() {
  this->tracker_.reset();

  const std::string tracker_type = yolo_ros::utils::to_lower(
      this->get_parameter("tracker_type").as_string());

  if (tracker_type == "bytetrack") {
    try {
      const yolo_ros::tracking::ByteTrackParams params =
          yolo_ros::tracking::load_byte_track_params(*this);
      this->tracker_ = yolo_ros::tracking::create_tracker(params);
    } catch (const rclcpp::exceptions::ParameterNotDeclaredException &e) {
      // Only reachable when the tracker's parameters were never declared on
      // this node (e.g. `tracker_type` switched to bytetrack on a node that
      // configured with another tracker); fall back to the passthrough path.
      RCLCPP_ERROR(this->get_logger(),
                   "[%s] ByteTrack parameters not declared: %s",
                   this->get_name(), e.what());
    }
  } else if (tracker_type == "botsort") {
    try {
      const yolo_ros::tracking::BotSortParams params =
          yolo_ros::tracking::load_bot_sort_params(*this);
      this->tracker_ = yolo_ros::tracking::create_tracker(params);
    } catch (const rclcpp::exceptions::ParameterNotDeclaredException &e) {
      RCLCPP_ERROR(this->get_logger(),
                   "[%s] BoT-SORT parameters not declared: %s",
                   this->get_name(), e.what());
    }
  }
  // --- add new trackers here: load their parameters and create the tracker ---

  if (this->tracker_ == nullptr) {
    RCLCPP_ERROR(this->get_logger(),
                 "[%s] No tracker created for tracker_type '%s' "
                 "(supported: bytetrack, botsort); forwarding detections "
                 "unchanged",
                 this->get_name(), tracker_type.c_str());
  } else {
    RCLCPP_INFO(this->get_logger(), "[%s] Using tracker '%s'", this->get_name(),
                tracker_type.c_str());
  }
}

void TrackingNode::recieve_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr &msg_image,
    const yolo_msgs::msg::DetectionArray::ConstSharedPtr &msg_detections) {
  yolo_msgs::msg::DetectionArray tracked_msg;
  tracked_msg.header = msg_image->header;

  // Safety fallback: no tracker (unknown `tracker_type`). Keep the pipeline
  // alive by forwarding the input detections untouched — downstream nodes
  // only lose the track ids, nothing crashes.
  if (this->tracker_ == nullptr) {
    RCLCPP_WARN_ONCE(
        this->get_logger(),
        "[%s] No active tracker; forwarding detections without track ids",
        this->get_name());
    this->tracking_publisher_->publish(*msg_detections);
    return;
  }

  // Convert the DetectionArray into the tracker's plain input format. The
  // detection index is preserved so the original Detection (class name, mask,
  // keypoints, ...) can be fetched back after tracking.
  std::vector<yolo_ros::tracking::TrackDetection> dets;
  dets.reserve(msg_detections->detections.size());
  for (std::size_t i = 0; i < msg_detections->detections.size(); ++i) {
    const auto &det = msg_detections->detections[i];
    if (det.bbox.size.x <= 0 || det.bbox.size.y <= 0) {
      continue;
    }
    yolo_ros::tracking::TrackDetection td;
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
  // the active tracker actually consumes it (BoT-SORT with gmc_method != none).
  cv::Mat frame;
  if (this->tracker_->needs_frame()) {
    try {
      frame = cv_bridge::toCvShare(msg_image, "bgr8")->image;
    } catch (const cv_bridge::Exception &e) {
      RCLCPP_WARN_ONCE(this->get_logger(),
                       "[%s] Image conversion failed (%s); running the tracker "
                       "without camera-motion compensation or appearance "
                       "features",
                       this->get_name(), e.what());
    }
  }

  const auto tracks = this->tracker_->update(dets, frame);

  for (const auto &track : tracks) {
    if (track.index < 0 ||
        track.index >= static_cast<int>(msg_detections->detections.size())) {
      continue;
    }
    // Copy the original detection (keeps class name, mask, etc.) and overlay
    // the Kalman-refined box and the stable track id.
    auto tracked_detection = msg_detections->detections[track.index];
    tracked_detection.bbox.center.position.x = (track.x1 + track.x2) / 2;
    tracked_detection.bbox.center.position.y = (track.y1 + track.y2) / 2;
    tracked_detection.bbox.size.x = track.x2 - track.x1;
    tracked_detection.bbox.size.y = track.y2 - track.y1;
    tracked_detection.id = std::to_string(track.id);

    tracked_msg.detections.push_back(tracked_detection);
  }

  this->tracking_publisher_->publish(tracked_msg);
}

} // namespace yolo_ros::node
