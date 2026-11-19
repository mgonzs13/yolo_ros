// Copyright (C) 2026 Miguel Ángel González Santamarta
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
    this->image_cache_ = cv::Mat();
    this->image_ready_ = false;
    this->bgr8_cache_ = cv::Mat();
    this->bgr8_ready_ = false;
    this->depth_image_cache_ = cv::Mat();
    this->depth_image_ready_ = false;
  }

  return *this;
}

cv::Mat CameraFrame::image() const {
  std::lock_guard<std::mutex> lock(this->cache_mutex_);

  if (!this->image_ready_) {
    if (this->rgb) {
      try {
        // No target encoding: cv_bridge aliases the message buffer whenever
        // the source encoding is directly mappable to a cv::Mat (rgb8, bgr8,
        // mono8, ...). The model preprocessing then handles the channel order.
        this->image_cache_ = cv_bridge::toCvShare(this->rgb)->image;
      } catch (const cv_bridge::Exception &) {
        this->image_cache_ = cv::Mat();
      }
    }

    this->image_ready_ = true;
  }

  return this->image_cache_;
}

bool CameraFrame::is_rgb8() const {
  return this->rgb && this->rgb->encoding == sensor_msgs::image_encodings::RGB8;
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
