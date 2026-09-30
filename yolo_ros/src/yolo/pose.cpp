// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include "yolo_ros/yolo/pose.hpp"
#include "yolo_msgs/msg/key_point2_d.hpp"
#include "yolo_ros/utils/logs.hpp"
#include <vector>

namespace yolo_ros::yolo {

namespace {

// COCO human pose keypoint configuration, matching kpt_shape=(17, 3) in the
// Ultralytics Pose head. The exported ONNX graph lays out one (x, y, visible)
// triple per keypoint, with the coordinates (like the boxes) already decoded
// into the letterboxed model-input frame and the visibility sigmoid'd.
constexpr int kNumKeypoints = 17;
constexpr int kKeypointDims = 3;                               // x, y, visible
constexpr int kKeypointValues = kNumKeypoints * kKeypointDims; // 51

} // namespace

YoloPose::YoloPose(yolo_ros::yolo::utils::YoloParams params)
    : yolo_ros::engine::Model(params, "pose") {}

YoloPose::~YoloPose() {}

std::vector<yolo_msgs::msg::Detection>
YoloPose::postprocess(const cv::Size &original_image_size,
                      const cv::Size &resized_image_size,
                      const std::vector<Ort::Value> &preds) {
  std::vector<yolo_msgs::msg::Detection> detection_array;
  std::vector<yolo_ros::yolo::utils::BoxWithKeypoints> boxes_with_kpts;

  const std::vector<int64_t> shape =
      preds[0].GetTensorTypeAndShapeInfo().GetShape();

  // Two exported layouts are supported, mirroring the detect node:
  //  - End-to-end/baked-head pose models (yolo26 family exported with the
  //    postprocess in-graph) output [*, K, 6 + nk]: each row is
  //    [x1, y1, x2, y2, score, class_idx, kpt...] and NMS is already applied.
  //  - Classic pose exports (yolov8/v11 pose) output [1, 4 + nc + nk, D]: one
  //    column per anchor holding [cx, cy, w, h, class_scores..., kpt...],
  //    decoded by the graph into the letterboxed frame; NMS is done here.
  const bool baked_head =
      shape.size() >= 2 && shape.back() == 4 + 1 + 1 + kKeypointValues;

  if (baked_head) {
    const float *raw = preds[0].GetTensorData<float>();
    const size_t num_rows = shape.size() == 3 ? static_cast<size_t>(shape[1])
                                              : static_cast<size_t>(shape[0]);
    const size_t stride = static_cast<size_t>(shape.back());

    for (size_t i = 0; i < num_rows; ++i) {
      const float *row = raw + i * stride;

      yolo_ros::yolo::utils::Box box;
      box.x1 = row[0];
      box.y1 = row[1];
      box.x2 = row[2];
      box.y2 = row[3];
      box.score = row[4];
      box.class_id = static_cast<int>(row[5]);
      box.index = static_cast<int>(i);
      box = yolo_ros::yolo::utils::scale_box(box, original_image_size,
                                             resized_image_size);

      std::vector<yolo_ros::yolo::utils::Keypoint> kpts(
          static_cast<size_t>(kNumKeypoints));
      for (int k = 0; k < kNumKeypoints; ++k) {
        kpts[static_cast<size_t>(k)].x = row[6 + k * kKeypointDims + 0];
        kpts[static_cast<size_t>(k)].y = row[6 + k * kKeypointDims + 1];
        kpts[static_cast<size_t>(k)].visible = row[6 + k * kKeypointDims + 2];
      }
      kpts = yolo_ros::yolo::utils::scale_keypoints(kpts, original_image_size,
                                                    resized_image_size);

      boxes_with_kpts.emplace_back(box, std::move(kpts));
    }

    // The end-to-end head emits the full max_det top-k (e.g. 300 rows), most
    // of which are near-zero confidence; drop anything below the threshold
    // (rows are already sorted by descending confidence, so this keeps the
    // strongest detections — mirroring the Python node's post-NMS output).
    boxes_with_kpts.erase(
        std::remove_if(
            boxes_with_kpts.begin(), boxes_with_kpts.end(),
            [this](const yolo_ros::yolo::utils::BoxWithKeypoints &b) {
              return b.score < this->conf_threshold;
            }),
        boxes_with_kpts.end());
  } else {
    // Classic raw path: features = 4 (box) + num_classes + nk (keypoints).
    const size_t num_features = static_cast<size_t>(shape[1]);
    const size_t num_detections = static_cast<size_t>(shape[2]);
    const int num_classes =
        static_cast<int>(num_features) - 4 - kKeypointValues;

    if (num_classes < 1) {
      YOLO_LOG_WARN("YoloPose: unexpected output feature count %zu; expected "
                    "4 + nc + %d",
                    num_features, kKeypointValues);
      return detection_array;
    }

    const float *raw = preds[0].GetTensorData<float>();
    for (size_t i = 0; i < num_detections; ++i) {
      const float center_x = raw[0 * num_detections + i];
      const float center_y = raw[1 * num_detections + i];
      const float width = raw[2 * num_detections + i];
      const float height = raw[3 * num_detections + i];

      int class_id = -1;
      float max_score = -1.0f;
      for (int j = 0; j < num_classes; ++j) {
        const float score = raw[(4 + j) * num_detections + i];
        if (score > max_score) {
          max_score = score;
          class_id = j;
        }
      }

      yolo_ros::yolo::utils::Box box;
      box.x1 = (center_x - width / 2);
      box.y1 = (center_y - height / 2);
      box.x2 = (center_x + width / 2);
      box.y2 = (center_y + height / 2);
      box.score = max_score;
      box.class_id = class_id;
      box.index = static_cast<int>(i);
      box = yolo_ros::yolo::utils::scale_box(box, original_image_size,
                                             resized_image_size);

      // Keypoint channels sit after the class scores; each keypoint is one
      // (x, y, visible) triple.
      std::vector<yolo_ros::yolo::utils::Keypoint> kpts(
          static_cast<size_t>(kNumKeypoints));
      for (int k = 0; k < kNumKeypoints; ++k) {
        const size_t channel = 4 + static_cast<size_t>(num_classes) +
                               static_cast<size_t>(k) * kKeypointDims;
        kpts[static_cast<size_t>(k)].x =
            raw[(channel + 0) * num_detections + i];
        kpts[static_cast<size_t>(k)].y =
            raw[(channel + 1) * num_detections + i];
        kpts[static_cast<size_t>(k)].visible =
            raw[(channel + 2) * num_detections + i];
      }
      kpts = yolo_ros::yolo::utils::scale_keypoints(kpts, original_image_size,
                                                    resized_image_size);

      boxes_with_kpts.emplace_back(box, std::move(kpts));
    }

    // Confidence filter, then per-class NMS (sorts in place; the returned
    // indices index the same, now-sorted vector).
    boxes_with_kpts.erase(
        std::remove_if(
            boxes_with_kpts.begin(), boxes_with_kpts.end(),
            [this](const yolo_ros::yolo::utils::BoxWithKeypoints &b) {
              return b.score < this->conf_threshold;
            }),
        boxes_with_kpts.end());

    const auto indices = yolo_ros::yolo::utils::nms(
        boxes_with_kpts, this->iou_threshold, this->conf_threshold);
    std::vector<yolo_ros::yolo::utils::BoxWithKeypoints> filtered;
    filtered.reserve(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      filtered.push_back(boxes_with_kpts[static_cast<size_t>(indices[i])]);
    }
    boxes_with_kpts.swap(filtered);
  }

  for (const auto &b : boxes_with_kpts) {
    yolo_msgs::msg::Detection detection;
    detection.bbox = yolo_ros::yolo::utils::convert_to_bounding_box(b);
    detection.score = b.score;
    detection.class_id = b.class_id;
    detection.id = "0";
    if (b.class_id < static_cast<int>(this->class_names.size())) {
      detection.class_name = this->class_names[b.class_id];
    } else {
      detection.class_name = "unknown";
    }

    // Publish only the keypoints whose visibility clears the confidence
    // threshold, with a 1-based id (same filtering and ids as the Python
    // node's parse_keypoints()). Coordinates are in original-image pixels.
    for (int k = 0; k < kNumKeypoints; ++k) {
      const auto &kp = b.keypoints[static_cast<size_t>(k)];
      if (kp.visible < this->conf_threshold) {
        continue;
      }
      yolo_msgs::msg::KeyPoint2D kp_msg;
      kp_msg.id = k + 1;
      kp_msg.point.x = kp.x;
      kp_msg.point.y = kp.y;
      kp_msg.score = kp.visible;
      detection.keypoints.data.push_back(kp_msg);
    }

    detection_array.push_back(detection);
  }

  return detection_array;
}

} // namespace yolo_ros::yolo
