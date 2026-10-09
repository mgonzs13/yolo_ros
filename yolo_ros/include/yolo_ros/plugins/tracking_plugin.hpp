// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief Multi-object tracking over synchronized camera frames + detections.

#ifndef YOLO_ROS__PLUGINS__TRACKING_PLUGIN_HPP_
#define YOLO_ROS__PLUGINS__TRACKING_PLUGIN_HPP_

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "yolo_msgs/msg/detection_array.hpp"
#include "yolo_ros/blackboard/channel_sync.hpp"
#include "yolo_ros/camera/camera_frame.hpp"
#include "yolo_ros/plugin/plugin.hpp"
#include "yolo_ros/tracking/tracker.hpp"

namespace yolo_ros {

/// @brief Runs the configured tracker on synchronized (frame, detections) and
/// publishes the refined detections with stable ids on `<camera>/tracking`.
class TrackingPlugin : public Plugin {
public:
  /// @brief Declare the tracker selection and tuning parameters under @p
  /// prefix.
  /// @param node Lifecycle node owning the parameters.
  /// @param prefix Instance prefix including the trailing dot, e.g. "track.".
  void declare_params(rclcpp_lifecycle::LifecycleNode &node,
                      const std::string &prefix) override;

  /// @brief Read the configured tracker and load its parameters under @p
  /// prefix.
  /// @param node Lifecycle node owning the parameters.
  /// @param prefix Instance prefix including the trailing dot, e.g. "track.".
  void get_params(const rclcpp_lifecycle::LifecycleNode &node,
                  const std::string &prefix) override;

  /// @brief Channel carrying the tracked detections of @p camera.
  /// @param camera Camera name.
  /// @return The output channel name (`<camera>/tracking`).
  std::string output_channel(const std::string &camera) const override {
    return camera + "/tracking";
  }

  /// @brief Subscribe to the inputs and declare the `tracking` channel.
  /// @return False when no camera was configured.
  bool setup(PluginContext &ctx) override;

  /// @brief Create the tracker selected by `tracker_type`.
  /// @return False when the tracker factory fails; true for the unknown-type
  /// pass-through case.
  bool activate() override;

  /// @brief Drop the tracker instance.
  void deactivate() override;

  /// @brief Synchronize the inputs and track until @p stop is set.
  /// @param stop Shared worker stop flag.
  void run(const std::atomic<bool> &stop) override;

private:
  /// @brief One tracked camera: frame+detection sync and its own tracker.
  struct CameraStream {
    /// @brief Camera name.
    std::string name;
    /// @brief Output channel/topic (`<cam>/tracking`).
    std::string output_channel;
    /// @brief Sync over the camera frame and the upstream detections.
    std::unique_ptr<ChannelSync<CameraFrame, yolo_msgs::msg::DetectionArray>>
        sync;
    /// @brief Tracker instance for this camera (null = passthrough).
    std::unique_ptr<tracking::Tracker> tracker;
  };

  /// @brief Track one synchronized (frame, detections) pair and publish the
  /// refined detections on the camera output channel.
  /// @param camera Camera stream owning the tracker and output channel.
  /// @param frame Synchronized camera frame.
  /// @param detections Synchronized 2D detections.
  void
  process(CameraStream &camera, const std::shared_ptr<const CameraFrame> &frame,
          const yolo_msgs::msg::DetectionArray::ConstSharedPtr &detections);

  /// @brief Plugin context copied from setup().
  std::unique_ptr<PluginContext> context_;
  /// @brief Selected cameras and their per-camera state.
  std::vector<CameraStream> cameras_;
  /// @brief Loaded tracker parameters; null for an unknown `tracker_type`.
  std::shared_ptr<tracking::TrackerParams> tracker_params_;

  /// @brief Selected tracker implementation ("bytetrack" or "botsort").
  std::string tracker_type_ = "bytetrack";
};

} // namespace yolo_ros

#endif // YOLO_ROS__PLUGINS__TRACKING_PLUGIN_HPP_
