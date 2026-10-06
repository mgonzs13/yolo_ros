// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2023-2025 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief Lifecycle node running a multi-object tracker over synchronized
/// image + detection messages.

#ifndef YOLO_ROS__NODE__TRACKING_NODE_HPP_
#define YOLO_ROS__NODE__TRACKING_NODE_HPP_

#include <memory>

#include "rclcpp/qos.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "yolo_msgs/msg/detection_array.hpp"
#include "yolo_ros/tracking/tracker.hpp"
#include "yolo_ros/utils/message_filters_compat.hpp"

/// @addtogroup yolo_nodes
/// @{
namespace yolo_ros::node {

/// @brief Approximate-time sync policy for (image, DetectionArray) pairs.
using TrackingSyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, yolo_msgs::msg::DetectionArray>;

/// @brief C++-ONNX equivalent of yolo_ros/yolo_ros/tracking_node.py.
///
/// Consumes the synchronized image + DetectionArray pair, runs a BYTETracker
/// over the detections and republishes the same detections on `tracking` with
/// the Kalman-refined bounding boxes and stable track ids filled in.
class TrackingNode : public rclcpp_lifecycle::LifecycleNode {
public:
  /// @brief Construct the node and declare the parameters.
  TrackingNode();

  /// @brief Create the tracker and set up the synchronized subscriptions and
  /// the tracking publisher.
  /// @param[in] state Previous lifecycle state.
  /// @return SUCCESS on success, FAILURE otherwise.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State &state);
  /// @brief Activate the tracking publisher.
  /// @param[in] state Previous lifecycle state.
  /// @return SUCCESS on success, FAILURE otherwise.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State &state);
  /// @brief Deactivate the tracking publisher.
  /// @param[in] state Previous lifecycle state.
  /// @return SUCCESS on success, FAILURE otherwise.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State &state);
  /// @brief Destroy the tracker and release the subscriptions.
  /// @param[in] state Previous lifecycle state.
  /// @return SUCCESS on success, FAILURE otherwise.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_cleanup(const rclcpp_lifecycle::State &state);
  /// @brief Tear everything down on shutdown.
  /// @param[in] state Previous lifecycle state.
  /// @return SUCCESS on success, FAILURE otherwise.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_shutdown(const rclcpp_lifecycle::State &state);

private:
  /// @brief Synchronized subscription to the input image topic.
  yolo_ros::utils::MessageFilterSubscriber<sensor_msgs::msg::Image>
      image_subscription_;
  /// @brief Synchronized subscription to the 2D detection topic.
  yolo_ros::utils::MessageFilterSubscriber<yolo_msgs::msg::DetectionArray>
      detection_subscription_;
  /// @brief Publisher of the tracked detections.
  rclcpp::Publisher<yolo_msgs::msg::DetectionArray>::SharedPtr
      tracking_publisher_;
  /// @brief Approximate-time synchronizer for the two subscriptions.
  std::shared_ptr<message_filters::Synchronizer<TrackingSyncPolicy>>
      synchronizer_;
  /// @brief QoS profile used for the image subscription.
  rclcpp::QoS image_qos_profile_;

  /// @brief Image topic to subscribe to (also used to sync with detections).
  std::string image_topic_;

  /// @brief The configured tracker (e.g. ByteTrack).
  std::unique_ptr<yolo_ros::tracking::Tracker> tracker_;

  /// @brief Declare the base parameters plus the selected tracker's own knobs.
  void declare_params();
  /// @brief Load the parameters and build the tracker via create_tracker().
  void load_params();

  /// @brief Synchronized callback: track the detections and republish them.
  /// @param[in] msg_image Synchronized image (used for timing/QoS).
  /// @param[in] msg_detections Detections to associate.
  void recieve_callback(
      const sensor_msgs::msg::Image::ConstSharedPtr &msg_image,
      const yolo_msgs::msg::DetectionArray::ConstSharedPtr &msg_detections);
};

} // namespace yolo_ros::node
/// @}

#endif // YOLO_ROS__NODE__TRACKING_NODE_HPP_
