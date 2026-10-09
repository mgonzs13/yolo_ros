// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief Draws detections onto images and publishes RViz markers.

#ifndef YOLO_ROS__PLUGINS__DEBUG_PLUGIN_HPP_
#define YOLO_ROS__PLUGINS__DEBUG_PLUGIN_HPP_

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "sensor_msgs/msg/image.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "yolo_msgs/msg/detection_array.hpp"
#include "yolo_msgs/msg/key_point3_d.hpp"
#include "yolo_ros/blackboard/channel_sync.hpp"
#include "yolo_ros/camera/camera_frame.hpp"
#include "yolo_ros/plugin/plugin.hpp"

namespace yolo_ros {

/// @brief Renders the debug image per camera from synchronized (frame,
/// detections) and rebuilds the 3D RViz markers from the chain input.
///
/// Publishes the per-camera output channels/topics `<cam>/debug_image`,
/// `<cam>/debug_bb_markers` and `<cam>/debug_kp_markers`.
///
/// `debug_image` is drawn from the camera frame synchronized with the freshest
/// non-3D upstream (the stream before the 3D plugin when it is in the chain),
/// so it is not gated by the slower 3D stream. Markers come from the chain
/// input (the previous plugin's output) and are emitted whenever those
/// detections carry `bbox3d`/`keypoints3d` data, on their own rate.
class DebugPlugin : public Plugin {
public:
  /// @brief Declare the marker lifetime parameter.
  /// @param[in,out] node Lifecycle node owning the parameters.
  /// @param[in] prefix Instance prefix including the trailing dot, e.g. "dbg.".
  void declare_params(rclcpp_lifecycle::LifecycleNode &node,
                      const std::string &prefix) override;

  /// @brief Read the declared debug parameters into the members.
  /// @param[in] node Lifecycle node owning the parameters.
  /// @param[in] prefix Instance prefix including the trailing dot, e.g. "dbg.".
  void get_params(const rclcpp_lifecycle::LifecycleNode &node,
                  const std::string &prefix) override;

  /// @brief Debug image channel produced for @p camera.
  /// @param[in] camera Camera name.
  /// @return `<camera>/debug_image`.
  std::string output_channel(const std::string &camera) const override {
    return camera + "/debug_image";
  }

  /// @brief Synchronize every selected camera frame with its upstream
  /// detections and expose the debug topics and channels.
  /// @param[in] ctx Plugin context carrying the blackboard and topic registry.
  /// @return False when no camera was selected.
  bool setup(PluginContext &ctx) override;

  /// @brief Draw and publish the per-camera debug image and 3D markers until
  /// @p stop.
  /// @param[in] stop Flag polled to leave the loop.
  void run(const std::atomic<bool> &stop) override;

private:
  /// @brief One debugged camera.
  struct CameraStream {
    /// @brief Camera name.
    std::string name;
    /// @brief Annotated image channel/topic (`<cam>/debug_image`).
    std::string image_channel;
    /// @brief Detection stream synchronized with the frame for drawing
    /// (freshest non-3D upstream).
    std::string image_input_channel;
    /// @brief Chain input channel carrying the 3D-enriched detections used for
    /// the RViz markers.
    std::string markers_channel;
    /// @brief 3D box markers channel/topic (`<cam>/debug_bb_markers`).
    std::string bb_markers_channel;
    /// @brief 3D keypoint markers channel/topic (`<cam>/debug_kp_markers`).
    std::string kp_markers_channel;
    /// @brief Sync over the camera frame and the 2D detections.
    std::unique_ptr<ChannelSync<CameraFrame, yolo_msgs::msg::DetectionArray>>
        sync;
    /// @brief Independent reader of the chain input for the 3D markers.
    ChannelReader<yolo_msgs::msg::DetectionArray> markers_reader;
  };

  /// @brief Publish the RViz markers of one camera for a 3D detection array.
  /// @param[in] camera Camera whose marker channels receive the arrays.
  /// @param[in] detections Detection array carrying bbox3d/keypoints3d data.
  void publish_markers(const CameraStream &camera,
                       const yolo_msgs::msg::DetectionArray &detections);

  /// @brief Stable, vivid BGR color for @p class_name (cached).
  /// @param[in] class_name Detection class name.
  /// @return The class color.
  cv::Scalar color_for_class(const std::string &class_name);

  /// @brief `class [id] score` text drawn next to a detection.
  /// @param[in] detection Detection to label.
  /// @return The formatted label text.
  std::string label_text(const yolo_msgs::msg::Detection &detection) const;

  /// @brief Draw the (possibly oriented) bounding box of @p detection.
  /// @param[in,out] image Image to draw on.
  /// @param[in] detection Detection whose box is drawn.
  /// @param[in] color Box and label color.
  /// @return The input image (mutated).
  cv::Mat draw_box(const cv::Mat &image,
                   const yolo_msgs::msg::Detection &detection,
                   const cv::Scalar &color);

  /// @brief Fill the mask on @p overlay and outline it on @p image.
  /// @param[in,out] overlay Shared mask layer, blended once by the caller.
  /// @param[in,out] image Final image receiving the crisp outline.
  /// @param[in] detection Detection carrying the mask.
  /// @param[in] color Mask color.
  void draw_mask(cv::Mat &overlay, cv::Mat &image,
                 const yolo_msgs::msg::Detection &detection,
                 const cv::Scalar &color);

  /// @brief Draw the keypoints and skeleton limbs of @p detection.
  /// @param[in,out] image Image to draw on.
  /// @param[in] detection Detection carrying the keypoints.
  /// @return The input image (mutated).
  cv::Mat draw_keypoints(const cv::Mat &image,
                         const yolo_msgs::msg::Detection &detection);

  /// @brief Draw a filled text label, slid back inside the frame if needed.
  /// @param[in,out] image Image to draw on.
  /// @param[in] text Label text.
  /// @param[in] anchor Desired top-left corner of the label.
  /// @param[in] background Label background color.
  /// @param[in] font_scale OpenCV font scale.
  /// @param[in] thickness OpenCV text thickness.
  /// @return Drawn label height, or 0 when fully off-frame.
  int draw_label(const cv::Mat &image, const std::string &text,
                 const cv::Point &anchor, const cv::Scalar &background,
                 double font_scale = 0.5, int thickness = 1);

  /// @brief Build the RViz CUBE marker for a 3D bounding box.
  /// @param[in] detection Detection carrying bbox3d.
  /// @param[in] color BGR color converted to the marker RGB.
  /// @return The configured marker.
  visualization_msgs::msg::Marker
  create_bb_marker(const yolo_msgs::msg::Detection &detection,
                   const cv::Scalar &color);

  /// @brief Build the RViz SPHERE marker for one 3D keypoint.
  /// @param[in] keypoint Keypoint to visualize.
  /// @return The configured marker.
  visualization_msgs::msg::Marker
  create_kp_marker(const yolo_msgs::msg::KeyPoint3D &keypoint);

  /// @brief Build the RViz LINE_LIST marker for one skeleton limb.
  /// @param[in] from Limb start keypoint.
  /// @param[in] to Limb end keypoint.
  /// @param[in] color BGR color converted to the marker RGB.
  /// @return The configured marker.
  visualization_msgs::msg::Marker
  create_limb_marker(const yolo_msgs::msg::KeyPoint3D &from,
                     const yolo_msgs::msg::KeyPoint3D &to,
                     const cv::Scalar &color);

  /// @brief Shared plugin context (blackboard, topics, logger).
  std::unique_ptr<PluginContext> context_;
  /// @brief Selected cameras and their per-camera state.
  std::vector<CameraStream> cameras_;
  /// @brief RViz marker lifetime in seconds (0 keeps markers persistent).
  double marker_lifetime_ = 0.5;
  /// @brief Cache of class name to BGR color.
  std::map<std::string, cv::Scalar> class_to_color_;
};

} // namespace yolo_ros

#endif // YOLO_ROS__PLUGINS__DEBUG_PLUGIN_HPP_
