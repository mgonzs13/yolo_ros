// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief Lifts 2D detections and keypoints into 3D using depth + tf2.

#ifndef YOLO_ROS__PLUGINS__DETECT_3D_PLUGIN_HPP_
#define YOLO_ROS__PLUGINS__DETECT_3D_PLUGIN_HPP_

#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "yolo_msgs/msg/detection_array.hpp"
#include "yolo_ros/3d/depth_utils.hpp"
#include "yolo_ros/blackboard/channel_sync.hpp"
#include "yolo_ros/camera/camera_frame.hpp"
#include "yolo_ros/plugin/plugin.hpp"

namespace yolo_ros {

/// @brief Synchronizes (CameraFrame, detections) per camera and publishes the
/// 3D-enriched detections on `<camera>/detections_3d`.
class Detect3DPlugin : public Plugin {
public:
  /// @brief Declare the tf2 and orientation parameters under @p prefix.
  /// @param node Lifecycle node owning the parameters.
  /// @param prefix Instance prefix including the trailing dot, e.g. "d3d.".
  void declare_params(rclcpp_lifecycle::LifecycleNode &node,
                      const std::string &prefix) override;

  /// @brief Read the tf2 and orientation parameters under @p prefix.
  /// @param node Lifecycle node owning the parameters.
  /// @param prefix Instance prefix including the trailing dot, e.g. "d3d.".
  void get_params(const rclcpp_lifecycle::LifecycleNode &node,
                  const std::string &prefix) override;

  /// @brief 3D detection channel of @p camera.
  /// @param camera Camera name.
  /// @return `<camera>/detections_3d`.
  std::string output_channel(const std::string &camera) const override {
    return camera + "/detections_3d";
  }

  /// @brief Build one (CameraFrame, detections) sync per depth camera and
  /// declare its output channel.
  /// @return False when a selected camera has no depth stream.
  bool setup(PluginContext &ctx) override;

  /// @brief Nothing to build; kept for lifecycle symmetry.
  /// @return Always true.
  bool activate() override;

  /// @brief Nothing to release.
  void deactivate() override;

  /// @brief Synchronize the inputs of every camera and lift each detection to
  /// 3D until @p stop is set.
  /// @param stop Shared worker stop flag.
  void run(const std::atomic<bool> &stop) override;

private:
  /// @brief One camera lifted to 3D.
  struct CameraStream {
    /// @brief Camera name.
    std::string name;
    /// @brief Output channel/topic (`<cam>/detections_3d`).
    std::string output_channel;
    /// @brief Sync over the camera frame and the upstream detections.
    std::unique_ptr<ChannelSync<CameraFrame, yolo_msgs::msg::DetectionArray>>
        sync;
  };

  /// @brief Lift every detection of one synchronized set into the target
  /// frame.
  /// @param frame Synchronized camera frame carrying depth and CameraInfo.
  /// @param detections_msg Synchronized 2D detections.
  /// @param orientation_state Per-camera OBB axis sign cache.
  /// @return The 3D-enriched detections (possibly empty).
  std::vector<yolo_msgs::msg::Detection> process_detections(
      const std::shared_ptr<const CameraFrame> &frame,
      const yolo_msgs::msg::DetectionArray::ConstSharedPtr &detections_msg,
      yolo_ros::depth::OrientationState &orientation_state);

  /// @brief Look up the transform from @p frame_id to `target_frame`.
  /// @param frame_id Source frame of the lookup.
  /// @return (translation, rotation), each as (x, y, z) and (w, x, y, z), or
  /// std::nullopt when tf2 is unavailable or the lookup fails.
  std::optional<std::pair<std::array<double, 3>, std::array<double, 4>>>
  get_transform(const std::string &frame_id);

  /// @brief Plugin context copied from setup().
  std::unique_ptr<PluginContext> context_;
  /// @brief Selected cameras and their per-camera state.
  std::vector<CameraStream> cameras_;
  /// @brief Per-camera OBB axis sign cache.
  std::map<std::string, yolo_ros::depth::OrientationState> orientation_states_;

  /// @brief Target frame for the published 3D boxes and keypoints.
  std::string target_frame_ = "base_link";
  /// @brief Raw depth units per metre.
  int depth_image_units_divisor_ = 1000;
  /// @brief Enable PCA-based orientation estimation.
  bool enable_orientation_ = false;
  /// @brief Minimum mask points required for orientation estimation.
  int min_seg_points_for_orientation_ = 20;
};

} // namespace yolo_ros

#endif // YOLO_ROS__PLUGINS__DETECT_3D_PLUGIN_HPP_
