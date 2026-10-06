// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2023-2025 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief Lifecycle node lifting 2D detections and pose keypoints into 3D.

#ifndef YOLO_ROS__NODE__DETECT_3D_NODE_HPP_
#define YOLO_ROS__NODE__DETECT_3D_NODE_HPP_

#include "rclcpp_lifecycle/lifecycle_node.hpp"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#if defined(CV_BRIDGE_H)
#include <cv_bridge/cv_bridge.h>
#else
#include <cv_bridge/cv_bridge.hpp>
#endif
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#if __has_include("tf2/exceptions.hpp")
#include "tf2/exceptions.hpp"
#else
#include "tf2/exceptions.h"
#endif
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "yolo_msgs/msg/bounding_box3_d.hpp"
#include "yolo_msgs/msg/detection_array.hpp"
#include "yolo_msgs/msg/key_point3_d.hpp"
#include "yolo_msgs/msg/key_point3_d_array.hpp"

#include "yolo_ros/3d/depth_utils.hpp"
#include "yolo_ros/utils/message_filters_compat.hpp"

/// @addtogroup yolo_nodes
/// @{
namespace yolo_ros::node {

/// @brief Approximate-time sync policy for (depth image, CameraInfo,
/// DetectionArray) triplets.
using SyncPolicy3D = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo,
    yolo_msgs::msg::DetectionArray>;

/// @brief C++-ONNX equivalent of yolo_ros/yolo_ros/detect_3d_node.py.
///
/// Synchronizes the depth image, its CameraInfo and the 2D DetectionArray,
/// lifts each 2D bbox (optionally using the segmentation mask to sample depth)
/// into a BoundingBox3D, projects the pose keypoints into 3D, and transforms
/// both into the target frame with tf2. Results are published on
/// `detections_3d`.
class Detect3DNode : public rclcpp_lifecycle::LifecycleNode {
public:
  /// @brief Construct the node and declare the parameters.
  Detect3DNode();

  /// @brief Set up the synchronized subscriptions, the transform listener and
  /// the detections_3d publisher.
  /// @param[in] state Previous lifecycle state.
  /// @return SUCCESS on success, FAILURE otherwise.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State &state);
  /// @brief Activate the publisher.
  /// @param[in] state Previous lifecycle state.
  /// @return SUCCESS on success, FAILURE otherwise.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State &state);
  /// @brief Deactivate the publisher.
  /// @param[in] state Previous lifecycle state.
  /// @return SUCCESS on success, FAILURE otherwise.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State &state);
  /// @brief Release the subscriptions and publisher.
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
  /// @brief Synchronized subscription to the depth image.
  yolo_ros::utils::MessageFilterSubscriber<sensor_msgs::msg::Image>
      depth_image_subscription_;
  /// @brief Synchronized subscription to the depth CameraInfo.
  yolo_ros::utils::MessageFilterSubscriber<sensor_msgs::msg::CameraInfo>
      depth_info_subscription_;
  /// @brief Synchronized subscription to the 2D detection topic.
  yolo_ros::utils::MessageFilterSubscriber<yolo_msgs::msg::DetectionArray>
      detection_subscription_;
  /// @brief Publisher of the 3D-enriched detections.
  rclcpp::Publisher<yolo_msgs::msg::DetectionArray>::SharedPtr
      detections_3d_publisher_;
  /// @brief Approximate-time synchronizer for the three subscriptions.
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy3D>> synchronizer_;

  /// @brief Frame into which results are transformed (empty = depth frame).
  std::string target_frame_;
  /// @brief Divisor applied to raw 16UC1 depth values to get metres.
  int depth_image_units_divisor_;
  /// @brief QoS reliability for the depth image subscription.
  int depth_image_reliability_;
  /// @brief QoS reliability for the CameraInfo subscription.
  int depth_info_reliability_;
  /// @brief Depth image topic.
  std::string depth_image_topic_;
  /// @brief Depth CameraInfo topic.
  std::string depth_info_topic_;
  /// @brief 2D detection topic to lift.
  std::string detections_topic_;
  /// @brief Enable oriented-bounding-box estimation (PCA plane frame).
  bool enable_orientation_;
  /// @brief Minimum valid depth points required for OBB estimation.
  int min_seg_points_for_orientation_;

  /// @brief Per-track sign-consistency cache for the OBB PCA axes.
  yolo_ros::depth::OrientationState orientation_state_;

  /// @brief tf2 buffer used to look up the target-frame transform.
  tf2_ros::Buffer tf_buffer_;
  /// @brief tf2 listener feeding tf_buffer_.
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  /// @brief Declare all ROS parameters with their defaults.
  void declare_params();
  /// @brief Read the declared parameters into the member fields.
  void load_params();

  /// @brief Synchronized callback: lift and publish the detections.
  /// @param[in] depth_msg Depth image.
  /// @param[in] depth_info_msg Camera intrinsics.
  /// @param[in] detections_msg 2D detections to lift.
  void recieve_callback(
      const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr &depth_info_msg,
      const yolo_msgs::msg::DetectionArray::ConstSharedPtr &detections_msg);

  /// @brief Convert every detection into 3D (boxes + keypoints) and transform
  /// them into the target frame.
  /// @param[in] depth_msg Depth image.
  /// @param[in] depth_info_msg Camera intrinsics.
  /// @param[in] detections_msg 2D detections to lift.
  /// @return The 3D-enriched detections.
  std::vector<yolo_msgs::msg::Detection> process_detections(
      const sensor_msgs::msg::Image::ConstSharedPtr &depth_msg,
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr &depth_info_msg,
      const yolo_msgs::msg::DetectionArray::ConstSharedPtr &detections_msg);

  /// @brief Look up the transform from @p frame_id to the target frame.
  /// @param[in] frame_id Source frame id.
  /// @return (translation, quaternion), or std::nullopt when the lookup fails.
  std::optional<std::pair<std::array<double, 3>, std::array<double, 4>>>
  get_transform(const std::string &frame_id);
};

} // namespace yolo_ros::node
/// @}

#endif // YOLO_ROS__NODE__DETECT_3D_NODE_HPP_
