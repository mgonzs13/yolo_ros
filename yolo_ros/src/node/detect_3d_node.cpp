// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2023-2025 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/node/detect_3d_node.hpp"

#include "yolo_ros/3d/depth_utils.hpp"

namespace yolo_ros::node {

Detect3DNode::Detect3DNode()
    : rclcpp_lifecycle::LifecycleNode("detect_3d_node"),
      tf_buffer_(this->get_clock()) {
  this->declare_params();
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
Detect3DNode::on_configure(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "[%s] Configuring...", this->get_name());

  this->load_params();

  this->detections_3d_publisher_ =
      this->create_publisher<yolo_msgs::msg::DetectionArray>("detections_3d",
                                                             10);

  // The tf listener works off the executor's callbacks; it must be alive
  // from configuration onwards so the buffer stays populated.
  this->tf_listener_ = std::make_shared<tf2_ros::TransformListener>(
      this->tf_buffer_, this->shared_from_this());

  RCLCPP_INFO(get_logger(), "[%s] Configured", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
Detect3DNode::on_activate(const rclcpp_lifecycle::State &) {
  auto reliability_to_policy = [](int r) {
    if (r == 0) {
      return rclcpp::ReliabilityPolicy::SystemDefault;
    }
    if (r == 1) {
      return rclcpp::ReliabilityPolicy::Reliable;
    }
    return rclcpp::ReliabilityPolicy::BestEffort;
  };

  rclcpp::QoS depth_image_qos = rclcpp::QoS(1).reliability(
      reliability_to_policy(this->depth_image_reliability_));
  rclcpp::QoS depth_info_qos = rclcpp::QoS(1).reliability(
      reliability_to_policy(this->depth_info_reliability_));

  this->depth_image_subscription_.subscribe(
      this->shared_from_this(), this->depth_image_topic_, depth_image_qos);
  this->depth_info_subscription_.subscribe(
      this->shared_from_this(), this->depth_info_topic_, depth_info_qos);
  this->detection_subscription_.subscribe(
      this->shared_from_this(), this->detections_topic_, rclcpp::QoS(10));

  this->synchronizer_ =
      std::make_shared<message_filters::Synchronizer<SyncPolicy3D>>(10);
  this->synchronizer_->connectInput(this->depth_image_subscription_,
                                    this->depth_info_subscription_,
                                    this->detection_subscription_);
  this->synchronizer_->registerCallback(
      std::bind(&Detect3DNode::recieve_callback, this, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3));

  RCLCPP_INFO(get_logger(), "[%s] Activated", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
Detect3DNode::on_deactivate(const rclcpp_lifecycle::State &) {
  this->detection_subscription_.unsubscribe();
  this->depth_info_subscription_.unsubscribe();
  this->depth_image_subscription_.unsubscribe();
  this->synchronizer_.reset();

  RCLCPP_INFO(get_logger(), "[%s] Deactivated", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
Detect3DNode::on_cleanup(const rclcpp_lifecycle::State &) {
  this->tf_listener_.reset();
  this->detections_3d_publisher_.reset();
  this->orientation_state_.last_axes.clear();

  RCLCPP_INFO(get_logger(), "[%s] Cleaned up", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
Detect3DNode::on_shutdown(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "[%s] Shutting down", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

void Detect3DNode::declare_params() {
  this->declare_parameter<std::string>("target_frame", "base_link");
  this->declare_parameter<int>("depth_image_units_divisor", 1000);
  this->declare_parameter<int>("depth_image_reliability", 2);
  this->declare_parameter<int>("depth_info_reliability", 2);
  this->declare_parameter<std::string>("depth_image_topic", "depth_image");
  this->declare_parameter<std::string>("depth_info_topic", "depth_info");
  this->declare_parameter<std::string>("detections_topic", "detections");
  this->declare_parameter<bool>("enable_orientation", false);
  this->declare_parameter<int>("min_seg_points_for_orientation", 20);
}

void Detect3DNode::load_params() {
  this->get_parameter("target_frame", this->target_frame_);
  this->get_parameter("depth_image_units_divisor",
                      this->depth_image_units_divisor_);
  this->get_parameter("depth_image_reliability",
                      this->depth_image_reliability_);
  this->get_parameter("depth_info_reliability", this->depth_info_reliability_);
  this->get_parameter("depth_image_topic", this->depth_image_topic_);
  this->get_parameter("depth_info_topic", this->depth_info_topic_);
  this->get_parameter("detections_topic", this->detections_topic_);
  this->get_parameter("enable_orientation", this->enable_orientation_);
  this->get_parameter("min_seg_points_for_orientation",
                      this->min_seg_points_for_orientation_);
}

void Detect3DNode::recieve_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr &depth_info_msg,
    const yolo_msgs::msg::DetectionArray::ConstSharedPtr &detections_msg) {
  yolo_msgs::msg::DetectionArray new_detections_msg;
  new_detections_msg.header = detections_msg->header;
  new_detections_msg.detections =
      this->process_detections(depth_msg, depth_info_msg, detections_msg);
  this->detections_3d_publisher_->publish(new_detections_msg);
}

std::vector<yolo_msgs::msg::Detection> Detect3DNode::process_detections(
    const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr &depth_info_msg,
    const yolo_msgs::msg::DetectionArray::ConstSharedPtr &detections_msg) {
  std::vector<yolo_msgs::msg::Detection> new_detections;

  if (detections_msg->detections.empty()) {
    return new_detections;
  }

  auto transform = this->get_transform(depth_info_msg->header.frame_id);
  if (!transform) {
    return new_detections;
  }

  // Build an OpenCV wrapper over the depth image without converting it, like
  // the Python node's "passthrough" encoding: keep the raw 16UC1 millimeters
  // (or 32FC1 meters) and divide by depth_image_units_divisor below.
  cv::Mat depth_image;
  try {
    auto cv_ptr = cv_bridge::toCvShare(
        depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    depth_image = cv_ptr->image;
  } catch (cv_bridge::Exception &e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    return new_detections;
  }

  for (const auto &detection : detections_msg->detections) {
    auto bbox3d = yolo_ros::depth::convert_bb_to_3d(
        depth_image, *depth_info_msg, detection,
        this->depth_image_units_divisor_,
        {this->enable_orientation_, this->min_seg_points_for_orientation_},
        &this->orientation_state_);
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
          depth_image, *depth_info_msg, detection,
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
Detect3DNode::get_transform(const std::string &frame_id) {
  try {
    // Zero time = latest available transform (same as the Python node).
    const auto transform = this->tf_buffer_.lookupTransform(
        this->target_frame_, frame_id, tf2::TimePointZero);

    std::array<double, 3> translation{transform.transform.translation.x,
                                      transform.transform.translation.y,
                                      transform.transform.translation.z};
    std::array<double, 4> rotation{
        transform.transform.rotation.w, transform.transform.rotation.x,
        transform.transform.rotation.y, transform.transform.rotation.z};

    return std::make_pair(translation, rotation);
  } catch (const tf2::TransformException &ex) {
    RCLCPP_ERROR(get_logger(), "Could not transform: %s", ex.what());
    return std::nullopt;
  }
}

} // namespace yolo_ros::node