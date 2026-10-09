// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief Node-level camera definitions and per-camera synchronized streams.

#ifndef YOLO_ROS__CAMERA__CAMERA_STREAMS_HPP_
#define YOLO_ROS__CAMERA__CAMERA_STREAMS_HPP_

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/logger.hpp"
#include "rclcpp/subscription.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "yolo_ros/blackboard/blackboard.hpp"
#include "yolo_ros/camera/camera_frame.hpp"
#include "yolo_ros/utils/message_filters_compat.hpp"

namespace yolo_ros {

/// @brief One camera definition read from the node parameters.
struct CameraConfig {
  /// @brief Camera name (also the frame channel and plugin prefix).
  std::string name;
  /// @brief RGB image topic (required).
  std::string rgb_topic;
  /// @brief Depth image topic (empty when the camera has no depth).
  std::string depth_topic;
  /// @brief Depth CameraInfo topic (empty when the camera has no depth).
  std::string depth_info_topic;
  /// @brief QoS reliability for the RGB subscription.
  int image_reliability = 2;
  /// @brief QoS reliability for the depth subscriptions.
  int depth_reliability = 2;

  /// @brief Whether this camera has a depth stream.
  bool has_depth() const { return !this->depth_topic.empty(); }
};

/// @brief Owns the camera subscriptions and publishes synchronized
/// CameraFrames on blackboard channel `<camera>`.
class CameraStreams {
public:
  /// @brief Approximate-time policy over (rgb, depth, camera_info).
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image, sensor_msgs::msg::Image,
      sensor_msgs::msg::CameraInfo>;

  /// @brief Construct with the blackboard that receives the frames.
  /// @param blackboard Blackboard shared with the plugins.
  /// @param logger Reserved for diagnostics; currently unused.
  CameraStreams(Blackboard &blackboard, rclcpp::Logger logger);

  /// @brief Declare/read and validate the camera parameters.
  /// @param node Node owning the parameters.
  /// @param names Camera name list from the `cameras` parameter.
  /// @param error Human-readable failure reason.
  /// @return True when every camera is valid.
  bool configure(rclcpp_lifecycle::LifecycleNode &node,
                 const std::vector<std::string> &names, std::string &error);

  /// @brief Create the subscriptions/synchronizers and start publishing.
  /// @param node Lifecycle node used to create the ROS entities.
  /// @param error Human-readable failure reason.
  /// @return True on success.
  bool activate(rclcpp_lifecycle::LifecycleNode &node, std::string &error);

  /// @brief Destroy the ROS entities (idempotent).
  void deactivate();

  /// @brief Validated camera definitions.
  const std::vector<CameraConfig> &cameras() const { return this->cameras_; }

  /// @brief Find a camera by name.
  /// @param name Camera name.
  /// @return Pointer to the config, or nullptr when unknown.
  const CameraConfig *find(const std::string &name) const;

private:
  /// @brief Per-camera runtime entities (only alive while activated).
  struct Stream {
    /// @brief RGB subscription used when the camera has no depth.
    typename rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr
        rgb_only_subscription;
    /// @brief Synchronized RGB subscription.
    yolo_ros::utils::MessageFilterSubscriber<sensor_msgs::msg::Image>
        rgb_subscription;
    /// @brief Synchronized depth subscription.
    yolo_ros::utils::MessageFilterSubscriber<sensor_msgs::msg::Image>
        depth_subscription;
    /// @brief Synchronized CameraInfo subscription.
    yolo_ros::utils::MessageFilterSubscriber<sensor_msgs::msg::CameraInfo>
        info_subscription;
    /// @brief Approximate-time synchronizer (depth cameras only).
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> synchronizer;
  };

  /// @brief Publish one frame on channel @p name.
  void publish_frame(const std::string &name,
                     const sensor_msgs::msg::Image::ConstSharedPtr &rgb,
                     const sensor_msgs::msg::Image::ConstSharedPtr &depth,
                     const sensor_msgs::msg::CameraInfo::ConstSharedPtr &info);

  /// @brief Blackboard receiving the frames.
  Blackboard &blackboard_;
  /// @brief Validated camera definitions.
  std::vector<CameraConfig> cameras_;
  /// @brief Per-camera entities, parallel to cameras_. Held by pointer because
  /// message_filters sources are non-copyable and non-movable.
  std::vector<std::unique_ptr<Stream>> streams_;
};

} // namespace yolo_ros

#endif // YOLO_ROS__CAMERA__CAMERA_STREAMS_HPP_
