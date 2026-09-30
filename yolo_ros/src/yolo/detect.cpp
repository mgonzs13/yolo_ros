// Copyright (c) 2025 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include "yolo_ros/yolo/detect.hpp"
#include "yolo_ros/yolo/utils.hpp"

namespace yolo_ros::yolo {

YoloDetect::YoloDetect(yolo_ros::yolo::utils::YoloParams params)
    : yolo_ros::engine::Model(params, "detect") {}

YoloDetect::~YoloDetect() {}

std::vector<yolo_msgs::msg::Detection>
YoloDetect::postprocess(const cv::Size &original_image_size,
                        const cv::Size &resized_image_size,
                        const std::vector<Ort::Value> &preds) {
  std::vector<yolo_msgs::msg::Detection> detection_array;
  std::vector<yolo_ros::yolo::utils::Box> detections;

  std::vector<int64_t> shape = preds[0].GetTensorTypeAndShapeInfo().GetShape();

  if (shape.back() == 6) {
    // Process predictions without applying NMS. The end-to-end head emits the
    // full max_det top-k (e.g. 300 rows), most of which are near-zero
    // confidence; drop anything below the threshold (rows are already sorted
    // by descending confidence, so this keeps the strongest detections —
    // mirroring the pose node's baked-head path).
    detections = get_detection_without_nms(preds, shape, original_image_size,
                                           resized_image_size);
    detections.erase(
        std::remove_if(detections.begin(), detections.end(),
                       [this](const yolo_ros::yolo::utils::Box &b) {
                         return b.score < this->conf_threshold;
                       }),
        detections.end());
  } else {
    // Process predictions applying NMS
    const size_t num_features = shape[1];
    const int num_classes = static_cast<int>(num_features) - 4;

    detections = get_detection_with_nms(
        preds, original_image_size, resized_image_size, num_classes,
        this->iou_threshold, this->conf_threshold);
  }

  for (size_t i = 0; i < detections.size(); ++i) {
    yolo_msgs::msg::Detection detection;
    detection.bbox =
        yolo_ros::yolo::utils::convert_to_bounding_box(detections[i]);
    detection.score = detections[i].score;
    detection.class_id = detections[i].class_id;
    detection.id = "0";
    if (detections[i].class_id < static_cast<int>(this->class_names.size())) {
      detection.class_name = this->class_names[detections[i].class_id];
    } else {
      detection.class_name = "unknown";
    }
    detection_array.push_back(detection);
  }

  return detection_array;
}

std::vector<yolo_ros::yolo::utils::Box> get_detection_without_nms(
    const std::vector<Ort::Value> &preds, std::vector<int64_t> output_shape,
    const cv::Size &original_image_size, const cv::Size &resized_image_size) {
  std::vector<yolo_ros::yolo::utils::Box> boxes;
  for (size_t i = 0; i < preds.size(); ++i) {
    auto pred = preds[i].GetTensorData<float>();

    // The end-to-end/baked-head layout is [1, K, 6] (batch, detections,
    // features); the classic layout is [K, 6]. The number of detections is
    // the last-but-one dimension, not the batch dim (output_shape[0] == 1).
    const size_t num_detections =
        output_shape.size() >= 3
            ? static_cast<size_t>(output_shape[output_shape.size() - 2])
            : static_cast<size_t>(output_shape[0]);

    for (size_t j = 0; j < num_detections; ++j) {
      yolo_ros::yolo::utils::Box box;
      box.x1 = pred[j * 6 + 0];
      box.y1 = pred[j * 6 + 1];
      box.x2 = pred[j * 6 + 2];
      box.y2 = pred[j * 6 + 3];
      box.score = pred[j * 6 + 4];
      box.class_id = static_cast<int>(pred[j * 6 + 5]);
      box.index = j;

      yolo_ros::yolo::utils::Box scaled_box = yolo_ros::yolo::utils::scale_box(
          box, original_image_size, resized_image_size);
      boxes.push_back(scaled_box);
    }
  }

  return boxes;
}

std::vector<yolo_ros::yolo::utils::Box> get_detection_with_nms(
    const std::vector<Ort::Value> &preds, const cv::Size &original_image_size,
    const cv::Size &resized_image_size, const int num_classes,
    float iou_threshold, float conf_threshold) {

  std::vector<yolo_ros::yolo::utils::Box> boxes =
      yolo_ros::yolo::utils::get_boxes(preds, original_image_size,
                                       resized_image_size, num_classes,
                                       conf_threshold);

  // Boxes are sorted in place by nms(); indices index the same vector.
  auto indices =
      yolo_ros::yolo::utils::nms(boxes, iou_threshold, conf_threshold);
  std::vector<yolo_ros::yolo::utils::Box> filtered_boxes;
  for (size_t i = 0; i < indices.size(); ++i) {
    filtered_boxes.push_back(boxes[indices[i]]);
  }
  return filtered_boxes;
}

} // namespace yolo_ros::yolo