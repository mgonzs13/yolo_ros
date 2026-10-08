// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/camera/camera_frame.hpp"

#include <mutex>

#if defined(CV_BRIDGE_H)
#include <cv_bridge/cv_bridge.h>
#else
#include <cv_bridge/cv_bridge.hpp>
#endif

#include "sensor_msgs/image_encodings.hpp"

namespace yolo_ros {

CameraFrame::CameraFrame(const CameraFrame &other) { *this = other; }

CameraFrame &CameraFrame::operator=(const CameraFrame &other) {
  if (this != &other) {
    this->header = other.header;
    this->rgb = other.rgb;
    this->depth = other.depth;
    this->depth_info = other.depth_info;
    this->bgr8_cache_ = cv::Mat();
    this->bgr8_ready_ = false;
    this->depth_image_cache_ = cv::Mat();
    this->depth_image_ready_ = false;
  }

  return *this;
}

cv::Mat CameraFrame::bgr8() const {
  std::lock_guard<std::mutex> lock(this->cache_mutex_);

  if (!this->bgr8_ready_) {
    if (this->rgb) {
      try {
        this->bgr8_cache_ =
            cv_bridge::toCvShare(this->rgb, sensor_msgs::image_encodings::BGR8)
                ->image;
      } catch (const cv_bridge::Exception &) {
        this->bgr8_cache_ = cv::Mat();
      }
    }

    this->bgr8_ready_ = true;
  }

  return this->bgr8_cache_;
}

cv::Mat CameraFrame::depth_image() const {
  std::lock_guard<std::mutex> lock(this->cache_mutex_);

  if (!this->depth_image_ready_) {
    if (this->depth) {
      try {
        // Requesting no target encoding keeps the source encoding, so 16UC1
        // stays zero-copy and 32FC1 keeps its float values.
        this->depth_image_cache_ = cv_bridge::toCvShare(this->depth)->image;
      } catch (const cv_bridge::Exception &) {
        this->depth_image_cache_ = cv::Mat();
      }
    }

    this->depth_image_ready_ = true;
  }

  return this->depth_image_cache_;
}

} // namespace yolo_ros
