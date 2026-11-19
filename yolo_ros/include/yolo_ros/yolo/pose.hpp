// Copyright (C) 2026 Alejandro González Cantón
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

/// @file
/// @brief Human pose task: detection boxes plus 2D keypoints.

#ifndef YOLO_ROS__YOLO__POSE_HPP_
#define YOLO_ROS__YOLO__POSE_HPP_

#include <vector>

#include "yolo_msgs/msg/detection.hpp"
#include "yolo_ros/engine/model.hpp"
#include "yolo_ros/yolo/utils.hpp"

/// @addtogroup yolo_tasks
/// @{
namespace yolo_ros::yolo {

/// @brief YOLO human pose model.
///
/// Handles both the raw pose export and the end-to-end export whose output row
/// is [1, K, 6 + nk] (box, score, class, then keypoints). The decoded
/// keypoints are filtered against the confidence threshold.
class YoloPose : public yolo_ros::engine::Model {
public:
  /// @brief Create the pose model from @p params.
  /// @param params Model and task configuration.
  YoloPose(yolo_ros::yolo::utils::YoloParams params);
  /// @brief Destroy the pose model.
  ~YoloPose();

protected:
  /// @brief Decode the pose tensors into Detection messages carrying 2D
  /// keypoints.
  /// @param[in] original_image_size Size of the original camera image.
  /// @param[in] resized_image_size Size of the letterboxed network input.
  /// @param[in] outputTensors Raw output tensors from inference().
  /// @return One Detection per person, each with a Pose2D.
  virtual std::vector<yolo_msgs::msg::Detection>
  postprocess(const cv::Size &original_image_size,
              const cv::Size &resized_image_size,
              const std::vector<Ort::Value> &outputTensors) override;
};
} // namespace yolo_ros::yolo
/// @}
#endif // YOLO_ROS__YOLO__POSE_HPP_
