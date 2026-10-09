// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief Zero-copy synchronized camera frame shared through the blackboard.

#ifndef YOLO_ROS__CAMERA__CAMERA_FRAME_HPP_
#define YOLO_ROS__CAMERA__CAMERA_FRAME_HPP_

#include <memory>
#include <mutex>

#include <opencv2/core.hpp>
#include <std_msgs/msg/header.hpp>

#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace yolo_ros {

/// @brief One synchronized camera sample: RGB plus optional depth and
/// CameraInfo.
///
/// The struct holds the incoming message shared_ptrs, so publishing a frame on
/// the blackboard copies no image data. Converted views are computed lazily by
/// the first consumer thread and cached for the remaining ones. The frame has
/// a std_msgs::msg::Header member, so message_filters detects its header and
/// timestamp automatically.
struct CameraFrame {
  /// @brief RGB image header; also the frame synchronization stamp.
  std_msgs::msg::Header header;
  /// @brief RGB image (always set).
  std::shared_ptr<const sensor_msgs::msg::Image> rgb;
  /// @brief Depth image (null when the camera has no depth stream).
  std::shared_ptr<const sensor_msgs::msg::Image> depth;
  /// @brief Depth CameraInfo (null when the camera has no depth stream).
  std::shared_ptr<const sensor_msgs::msg::CameraInfo> depth_info;

  /// @brief Default constructor; all optional members are empty.
  CameraFrame() = default;

  /// @brief Copy construction copies the message pointers and resets the
  /// conversion caches.
  ///
  /// message_filters::MessageEvent instantiates copy assignment for its
  /// payload type even for const events, so the mutex-guarded caches must not
  /// implicitly delete the copy operations.
  CameraFrame(const CameraFrame &other);

  /// @brief Copy assignment copies the message pointers and resets the
  /// conversion caches; the mutex is per-instance and never copied.
  CameraFrame &operator=(const CameraFrame &other);

  /// @brief Converted view of rgb as BGR8, computed once and cached.
  /// @return A cv::Mat valid while this frame is alive (empty when rgb is
  /// null).
  cv::Mat bgr8() const;

  /// @brief View of depth in its source encoding, computed once and cached.
  ///
  /// The view aliases the message buffer whenever cv_bridge can map the source
  /// encoding directly, so 16UC1 data stays zero-copy and 32FC1 metres keep
  /// their original float values instead of being truncated to 16 bits.
  /// @return A cv::Mat valid while this frame is alive (empty when depth is
  /// null or its encoding is unsupported).
  cv::Mat depth_image() const;

private:
  /// @brief Guards the conversion caches.
  mutable std::mutex cache_mutex_;
  /// @brief Cached BGR8 view of rgb.
  mutable cv::Mat bgr8_cache_;
  /// @brief Whether bgr8_cache_ has been computed.
  mutable bool bgr8_ready_ = false;
  /// @brief Cached view of depth in its source encoding.
  mutable cv::Mat depth_image_cache_;
  /// @brief Whether depth_image_cache_ has been computed.
  mutable bool depth_image_ready_ = false;
};

} // namespace yolo_ros

#endif // YOLO_ROS__CAMERA__CAMERA_FRAME_HPP_
