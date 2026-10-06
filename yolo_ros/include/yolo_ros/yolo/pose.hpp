// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

/// @file
/// @brief Human pose task: detection boxes plus 2D keypoints.

#ifndef YOLO_ROS__YOLO__POSE_HPP_
#define YOLO_ROS__YOLO__POSE_HPP_

#include "yolo_msgs/msg/detection.hpp"
#include "yolo_ros/engine/model.hpp"
#include "yolo_ros/yolo/utils.hpp"
#include <vector>

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
