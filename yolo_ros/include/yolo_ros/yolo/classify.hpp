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
/// @brief Image-level classification task (YOLO cls models).

#ifndef YOLO_ROS__YOLO__CLASSIFY_HPP_
#define YOLO_ROS__YOLO__CLASSIFY_HPP_

#include <vector>

#include "yolo_msgs/msg/detection.hpp"
#include "yolo_ros/engine/model.hpp"
#include "yolo_ros/yolo/utils.hpp"

/// @addtogroup yolo_tasks
/// @{
namespace yolo_ros::yolo {

/// @brief Image-level classifier (YOLO cls models).
///
/// Like the other tasks it only overrides the generic postprocess() hook;
/// preprocess()/inference() are the engine's. The exported graphs emit one
/// [1, N] row of per-class scores; ultralytics cls exports bake a softmax into
/// the graph, so the row is already a probability distribution. Each top-k
/// class is published as a Detection with an EMPTY bbox (there is no spatial
/// extent for an image-level label), carrying class_id / class_name / score —
/// matching how the rest of the package reuses Detection.
class YoloClassify : public yolo_ros::engine::Model {
public:
  /// @brief Create the classifier from @p params.
  /// @param params Model and task configuration.
  YoloClassify(yolo_ros::yolo::utils::YoloParams params);
  /// @brief Destroy the classifier.
  ~YoloClassify();

protected:
  /// @brief Read the [1, N] score row and return the top-k classes as
  /// Detections with an empty bbox.
  /// @param[in] original_image_size Size of the original camera image.
  /// @param[in] resized_image_size Size of the letterboxed network input.
  /// @param[in] preds Raw output tensors from inference().
  /// @return Up to top_k Detections, sorted by descending score.
  virtual std::vector<yolo_msgs::msg::Detection>
  postprocess(const cv::Size &original_image_size,
              const cv::Size &resized_image_size,
              const std::vector<Ort::Value> &preds) override;

private:
  /// @brief Number of classes published per image (probabilities, descending).
  int top_k_{5}; // classes published per image (softmax probs, desc order)
};

} // namespace yolo_ros::yolo
/// @}

#endif // YOLO_ROS__YOLO__CLASSIFY_HPP_
