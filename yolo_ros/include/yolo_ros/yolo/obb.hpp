// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2023-2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief Oriented (rotated) bounding box task with its own per-class rotated
/// NMS.

#ifndef YOLO_ROS__YOLO__OBB_HPP_
#define YOLO_ROS__YOLO__OBB_HPP_

#include "yolo_msgs/msg/detection.hpp"
#include "yolo_ros/engine/model.hpp"
#include "yolo_ros/yolo/utils.hpp"
#include <vector>

/// @addtogroup yolo_tasks
/// @{
namespace yolo_ros::yolo {

/// @brief An oriented bounding box in the ultralytics xywhr parameterization
/// (center x, center y, width, height, rotation angle in radians).
///
/// The box is decoded into the letterboxed model-input frame; the angle is
/// unaffected by letterboxing (it is a rotation, not a scale).
struct ObbBox {
  float cx = 0.0f; ///< Center x in network-input coordinates.
  float cy = 0.0f; ///< Center y in network-input coordinates.
  float w = 0.0f;  ///< Box width along its own axis.
  float h = 0.0f;  ///< Box height along its own axis.
  /// @brief Rotation angle in radians.
  float angle = 0.0f; // radians
  /// @brief Detection confidence in [0, 1].
  float score = 0.0f;
  /// @brief Zero-based class id, or -1 when unset.
  int class_id = -1;
};

/// @brief Oriented (rotated) bounding box detection.
///
/// Ultralytics OBB heads predict one extra channel per anchor vs. Detect:
/// [cx, cy, w, h, class_scores..., angle]. The angle channel is already decoded
/// in radians by the exported graph (YOLO11: (sigmoid - 0.25) * pi in
/// [-pi/4, 3pi/4]; YOLO26: raw radians). OBB exports are always the raw path:
/// rotated NMS cannot be baked into the graph, so NMS is performed here (the
/// repository's pose/detect nodes additionally handle baked-head exports, which
/// OBB does not produce).
class YoloOBB : public yolo_ros::engine::Model {
public:
  /// @brief Create the OBB model from @p params.
  /// @param params Model and task configuration.
  YoloOBB(yolo_ros::yolo::utils::YoloParams params);
  /// @brief Destroy the OBB model.
  ~YoloOBB();

protected:
  /// @brief Decode the OBB tensors into Detection messages whose rotated box
  /// is carried as a BoundingBox2D (center.theta = angle in radians,
  /// size = rotated w/h), applying the per-class rotated NMS.
  /// @param[in] original_image_size Size of the original camera image.
  /// @param[in] resized_image_size Size of the letterboxed network input.
  /// @param[in] outputTensors Raw output tensors from inference().
  /// @return One Detection per kept oriented box.
  virtual std::vector<yolo_msgs::msg::Detection>
  postprocess(const cv::Size &original_image_size,
              const cv::Size &resized_image_size,
              const std::vector<Ort::Value> &outputTensors) override;
};
} // namespace yolo_ros::yolo
/// @}
#endif // YOLO_ROS__YOLO__OBB_HPP_
