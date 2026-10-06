// Copyright (c) 2025 Alejandro González Cantón
// Portions Copyright (c) 2023-2025 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief Lifecycle node rendering detections and RViz markers for debugging.

#ifndef YOLO_ROS__NODE__DEBUG_NODE_HPP_
#define YOLO_ROS__NODE__DEBUG_NODE_HPP_

#include "rclcpp_lifecycle/lifecycle_node.hpp"

#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "yolo_msgs/msg/detection_array.hpp"
#include "yolo_msgs/msg/key_point3_d.hpp"
#include "yolo_ros/utils/message_filters_compat.hpp"

#include <opencv2/opencv.hpp>
#include <rclcpp/context.hpp>

/// @addtogroup yolo_nodes
/// @{
namespace yolo_ros::node {
/// @brief Approximate-time sync policy for (image, DetectionArray) pairs.
using ApproximateSyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, yolo_msgs::msg::DetectionArray>;

/// @brief Lifecycle node that draws detections onto the input image and
/// publishes RViz MarkerArrays.
///
/// `debug_image` is driven by the synchronized image + 2D detections and shows
/// boxes, masks and keypoint skeletons. The 3D markers are driven by an
/// independent subscription to the (slower) 3D-enriched detection stream, so
/// `debug_image` publishes at the full 2D detection rate while the markers are
/// gated by the 3D stream.
class DebugNode : public rclcpp_lifecycle::LifecycleNode {
public:
  /// @brief Construct the node.
  DebugNode();

  /// @brief Set up the synchronized subscriptions and the three publishers.
  /// @param[in] state Previous lifecycle state.
  /// @return SUCCESS on success, FAILURE otherwise.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State &state);
  /// @brief Activate the publishers.
  /// @param[in] state Previous lifecycle state.
  /// @return SUCCESS on success, FAILURE otherwise.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State &state);
  /// @brief Deactivate the publishers.
  /// @param[in] state Previous lifecycle state.
  /// @return SUCCESS on success, FAILURE otherwise.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State &state);
  /// @brief Release the subscriptions and publishers.
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
      image_subscription;
  /// @brief Synchronized subscription to the 2D detection topic.
  yolo_ros::utils::MessageFilterSubscriber<yolo_msgs::msg::DetectionArray>
      detection_subscription;
  /// @brief Publisher of the annotated debug image.
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_publisher;
  /// @brief Publisher of the 3D bounding-box markers.
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      bb_markers_publisher;
  /// @brief Publisher of the 3D keypoint markers.
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      kp_markers_publisher;
  /// @brief Approximate-time synchronizer for image + 2D detections.
  std::shared_ptr<message_filters::Synchronizer<ApproximateSyncPolicy>>
      synchronizer;

  /// @brief QoS profile used for the image subscription.
  rclcpp::QoS image_qos_profile;

  /// @brief Image topic to subscribe to.
  std::string image_topic_;
  /// @brief 2D detection topic to subscribe to.
  std::string detections_topic_;
  /// @brief 3D detection topic driving the RViz markers.
  std::string markers_topic_;
  /// @brief Lifetime (seconds) of the RViz markers; a marker that stops being
  /// re-published (its detection vanished from the 3D stream) expires after
  /// this long. 0 keeps markers persistent.
  double marker_lifetime_ = 0.5;

  // Independently subscribes to the 3D-enriched stream (detections_3d) purely
  // to drive the RViz 3D markers. Keeping this separate from the
  // image<->2D-detections sync means debug_image publishes at the full 2D
  // detection rate while the (slower) 3D stream only gates the markers.
  /// @brief Independent subscription to the 3D detection stream.
  rclcpp::Subscription<yolo_msgs::msg::DetectionArray>::SharedPtr
      markers_subscription_;

  /// @brief Cached per-class BGR colors (assigned on first use).
  std::map<std::string, cv::Scalar> class_to_color;

  /// @brief Synchronized callback: draw the detections onto the image and
  /// publish it.
  /// @param[in] msg_image Incoming image message.
  /// @param[in] msg_detections Incoming 2D detections.
  void recieve_callback(
      const sensor_msgs::msg::Image::ConstSharedPtr &msg_image,
      const yolo_msgs::msg::DetectionArray::ConstSharedPtr &msg_detections);
  /// @brief Callback on the 3D detection stream: rebuild the RViz markers.
  /// @param[in] msg_detections Incoming 3D-enriched detections.
  void markers_callback(
      const yolo_msgs::msg::DetectionArray::ConstSharedPtr &msg_detections);

  /// @brief Return the cached color for @p class_name, assigning one on first
  /// use.
  /// @param[in] class_name Class name.
  /// @return The BGR color for the class.
  cv::Scalar color_for_class(const std::string &class_name);

  /// @brief Compose the label string ("class [id] score") for @p detection.
  /// @param[in] detection Detection to label.
  /// @return The label text.
  std::string label_text(const yolo_msgs::msg::Detection &detection) const;
  /// @brief Draw one bounding box (rotated for OBB detections).
  /// @param[in] image Image to draw on.
  /// @param[in] detection Detection to draw.
  /// @param[in] color BGR color.
  /// @return The image with the box drawn.
  cv::Mat draw_box(const cv::Mat &image,
                   const yolo_msgs::msg::Detection &detection,
                   const cv::Scalar &color);
  /// @brief Blend the instance mask of @p detection into @p image using
  /// @p overlay as scratch.
  /// @param[in,out] overlay Scratch overlay layer.
  /// @param[in,out] image Image to draw on.
  /// @param[in] detection Detection carrying the mask.
  /// @param[in] color BGR mask color.
  void draw_mask(cv::Mat &overlay, cv::Mat &image,
                 const yolo_msgs::msg::Detection &detection,
                 const cv::Scalar &color);
  /// @brief Draw the color-coded keypoint skeleton of @p detection.
  /// @param[in] image Image to draw on.
  /// @param[in] detection Detection carrying the keypoints.
  /// @return The image with the skeleton drawn.
  cv::Mat draw_keypoints(const cv::Mat &image,
                         const yolo_msgs::msg::Detection &detection);

  /// @brief Draw @p text inside a filled label box placed at @p anchor (its
  /// top-left corner), with the highest-contrast (black or white) glyphs and
  /// clamped so the label always stays inside the frame.
  /// @param[in] image Image to draw on.
  /// @param[in] text Text to render.
  /// @param[in] anchor Desired top-left corner of the label box.
  /// @param[in] background Fill color of the label background (BGR).
  /// @param[in] font_scale OpenCV Hershey font scale.
  /// @param[in] thickness Glyph stroke thickness.
  /// @return Height in pixels of the drawn label box, or 0 when it was skipped
  /// (degenerate or fully off-frame).
  int draw_label(const cv::Mat &image, const std::string &text,
                 const cv::Point &anchor, const cv::Scalar &background,
                 double font_scale = 0.5, int thickness = 1);

  /// @brief Build the RViz marker for a 3D bounding box.
  /// @param[in] detection Detection carrying a 3D box.
  /// @param[in] color Marker color.
  /// @return The marker message.
  visualization_msgs::msg::Marker
  create_bb_marker(const yolo_msgs::msg::Detection &detection,
                   const cv::Scalar &color);
  /// @brief Build the RViz marker for one 3D keypoint.
  /// @param[in] keypoint The 3D keypoint.
  /// @return The marker message.
  visualization_msgs::msg::Marker
  create_kp_marker(const yolo_msgs::msg::KeyPoint3D &keypoint);
  /// @brief Build the RViz line marker connecting two 3D keypoints (a skeleton
  /// limb).
  /// @param[in] from First limb endpoint.
  /// @param[in] to Second limb endpoint.
  /// @param[in] color BGR color of the limb.
  /// @return The marker message.
  visualization_msgs::msg::Marker
  create_limb_marker(const yolo_msgs::msg::KeyPoint3D &from,
                     const yolo_msgs::msg::KeyPoint3D &to,
                     const cv::Scalar &color);
};
} // namespace yolo_ros::node
/// @}

#endif // YOLO_ROS__NODE__DEBUG_NODE_HPP_
