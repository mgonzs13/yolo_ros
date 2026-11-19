// Copyright (C) 2025 Alejandro González Cantón
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
/// @brief Object detection task: decode raw YOLO tensors into boxes and NMS.

#ifndef YOLO_ROS__YOLO__DETECT_HPP_
#define YOLO_ROS__YOLO__DETECT_HPP_

#include <vector>

#include "yolo_ros/engine/model.hpp"
#include "yolo_ros/yolo/utils.hpp"

/// @addtogroup yolo_tasks
/// @{
namespace yolo_ros::yolo {

/// @brief Decode a raw (no baked NMS) detection tensor into boxes.
/// @param[in] preds Raw output tensors from the model.
/// @param[in] output_shape Shape of the raw detection tensor.
/// @param[in] original_image_size Size of the original camera image.
/// @param[in] resized_image_size Size of the letterboxed network input.
/// @return Candidate boxes in original-image coordinates (no NMS applied).
std::vector<yolo_ros::yolo::utils::Box> get_detection_without_nms(
    const std::vector<Ort::Value> &preds, std::vector<int64_t> output_shape,
    const cv::Size &original_image_size, const cv::Size &resized_image_size);

/// @brief Decode a raw detection tensor and apply the C++ per-class NMS.
/// @param[in] preds Raw output tensors from the model.
/// @param[in] original_image_size Size of the original camera image.
/// @param[in] resized_image_size Size of the letterboxed network input.
/// @param[in] num_classes Number of model classes.
/// @param[in] iou_threshold IoU threshold for NMS.
/// @param[in] conf_threshold Minimum score to keep a box.
/// @return NMS-filtered boxes in original-image coordinates.
std::vector<yolo_ros::yolo::utils::Box> get_detection_with_nms(
    const std::vector<Ort::Value> &preds, const cv::Size &original_image_size,
    const cv::Size &resized_image_size, const int num_classes,
    float iou_threshold, float conf_threshold);

/// @brief Standard YOLO object detector (no mask/keypoints).
///
/// Handles both raw exports (its own NMS) and end-to-end/baked-NMS exports
/// whose graph already emits filtered boxes.
class YoloDetect : public yolo_ros::engine::Model {
public:
  /// @brief Create the detector from @p params.
  /// @param params Model and task configuration.
  YoloDetect(yolo_ros::yolo::utils::YoloParams params);
  /// @brief Destroy the detector.
  ~YoloDetect();

protected:
  /// @brief Decode the detection tensor(s) into Detection messages, applying
  /// NMS when the export is the raw (non-baked) layout.
  /// @param[in] original_image_size Size of the original camera image.
  /// @param[in] resized_image_size Size of the letterboxed network input.
  /// @param[in] outputTensors Raw output tensors from inference().
  /// @return One Detection per kept box.
  virtual std::vector<yolo_msgs::msg::Detection>
  postprocess(const cv::Size &original_image_size,
              const cv::Size &resized_image_size,
              const std::vector<Ort::Value> &outputTensors) override;
};
} // namespace yolo_ros::yolo
/// @}
#endif // YOLO_ROS__YOLO__DETECT_HPP_
