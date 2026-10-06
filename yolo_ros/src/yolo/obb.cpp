// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2023-2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/yolo/obb.hpp"
#include "yolo_ros/utils/logs.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace yolo_ros::yolo {

namespace {

// Corners of an oriented box in ultralytics xywhr convention (see
// ultralytics.utils.ops.xywhr2xyxyxyxy): vec1 = (w/2)(cos, sin) along the
// rotated width axis, vec2 = (h/2)(-sin, cos) along the rotated height axis.
// Order is pt1, pt2, pt3, pt4 (a convex quad, counterclockwise for angle 0).
std::array<cv::Point2f, 4> obb_corners(float cx, float cy, float w, float h,
                                       float angle) {
  const float c = std::cos(angle);
  const float s = std::sin(angle);
  const cv::Point2f center(cx, cy);
  const cv::Point2f vec1(w / 2 * c, w / 2 * s);
  const cv::Point2f vec2(-h / 2 * s, h / 2 * c);
  return {center + vec1 + vec2, center + vec1 - vec2, center - vec1 - vec2,
          center - vec1 + vec2};
}

// Intersection-over-union of two oriented boxes, computed from the area of
// their convex intersection polygon (OpenCV) over the union of the two areas.
float rotated_iou(const ObbBox &a, const ObbBox &b) {
  const auto pa = obb_corners(a.cx, a.cy, a.w, a.h, a.angle);
  const auto pb = obb_corners(b.cx, b.cy, b.w, b.h, b.angle);

  const float area_a = static_cast<float>(cv::contourArea(pa));
  const float area_b = static_cast<float>(cv::contourArea(pb));
  if (area_a <= 0.0f || area_b <= 0.0f) {
    return 0.0f;
  }

  std::vector<cv::Point2f> intersection;
  const float inter_area =
      cv::intersectConvexConvex(pa, pb, intersection, true);
  const float union_area = area_a + area_b - inter_area;
  return union_area > 0.0f ? inter_area / union_area : 0.0f;
}

// Per-class rotated NMS, mirroring yolo_ros::yolo::utils::nms() but using the
// oriented IoU above. Sorts `boxes` in place by descending confidence and
// returns the kept indices into the same (now-sorted) vector.
std::vector<int> rotated_nms(std::vector<ObbBox> &boxes, float iou_threshold,
                             float conf_threshold) {
  std::vector<int> indices;
  std::sort(boxes.begin(), boxes.end(),
            [](const ObbBox &a, const ObbBox &b) { return a.score > b.score; });

  std::vector<bool> suppressed(boxes.size(), false);
  for (size_t i = 0; i < boxes.size(); ++i) {
    if (boxes[i].score < conf_threshold || suppressed[i]) {
      continue;
    }
    indices.push_back(static_cast<int>(i));
    for (size_t j = i + 1; j < boxes.size(); ++j) {
      if (suppressed[j] || boxes[j].class_id != boxes[i].class_id) {
        continue;
      }
      if (rotated_iou(boxes[i], boxes[j]) > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }
  return indices;
}

// Inverse-letterbox a box decoded into the model-input frame back into the
// original image (same geometry as yolo_ros::yolo::utils::scale_box, but for
// the center + width/height + angle parameterization).
ObbBox scale_obb(const ObbBox &box, const cv::Size &original_image_size,
                 const cv::Size &resized_image_size) {
  const float gain = std::min(static_cast<float>(resized_image_size.width) /
                                  original_image_size.width,
                              static_cast<float>(resized_image_size.height) /
                                  original_image_size.height);
  const float pad_x =
      (resized_image_size.width - original_image_size.width * gain) / 2;
  const float pad_y =
      (resized_image_size.height - original_image_size.height * gain) / 2;

  ObbBox scaled;
  scaled.cx = std::clamp((box.cx - pad_x) / gain, 0.0f,
                         static_cast<float>(original_image_size.width));
  scaled.cy = std::clamp((box.cy - pad_y) / gain, 0.0f,
                         static_cast<float>(original_image_size.height));
  scaled.w = std::max(0.0f, box.w / gain);
  scaled.h = std::max(0.0f, box.h / gain);
  scaled.angle = box.angle; // rotation is invariant under letterboxing
  scaled.score = box.score;
  scaled.class_id = box.class_id;
  return scaled;
}

} // namespace

YoloOBB::YoloOBB(yolo_ros::yolo::utils::YoloParams params)
    : yolo_ros::engine::Model(params, "obb") {}

YoloOBB::~YoloOBB() {}

std::vector<yolo_msgs::msg::Detection>
YoloOBB::postprocess(const cv::Size &original_image_size,
                     const cv::Size &resized_image_size,
                     const std::vector<Ort::Value> &preds) {
  std::vector<yolo_msgs::msg::Detection> detection_array;
  if (preds.empty()) {
    return detection_array;
  }

  const std::vector<int64_t> shape =
      preds[0].GetTensorTypeAndShapeInfo().GetShape();

  // Raw OBB export: [1, 4 + nc + 1, N]. Per anchor the channels are
  // [cx, cy, w, h, class_scores..., angle] with the class scores already
  // sigmoid'd and the angle already decoded (radians) by the ONNX graph.
  if (shape.size() < 3) {
    YOLO_LOG_WARN("YoloOBB: unexpected output rank %zu; expected [1, 4 + nc + "
                  "1, N]",
                  shape.size());
    return detection_array;
  }
  const int64_t num_features = shape[1];
  const int64_t num_detections = shape[2];
  const int num_classes = static_cast<int>(num_features) - 5;
  if (num_classes < 1) {
    YOLO_LOG_WARN("YoloOBB: unexpected output feature count %lld; expected "
                  "4 + nc + 1",
                  static_cast<long long>(num_features));
    return detection_array;
  }
  const size_t angle_channel = static_cast<size_t>(num_features) - 1;

  const float *raw = preds[0].GetTensorData<float>();
  const size_t n = static_cast<size_t>(num_detections);

  std::vector<ObbBox> boxes;
  boxes.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    int class_id = -1;
    float max_score = -1.0f;
    for (int j = 0; j < num_classes; ++j) {
      const float score = raw[(4 + j) * n + i];
      if (score > max_score) {
        max_score = score;
        class_id = j;
      }
    }
    // Skip anchors whose best class score is below the threshold (most of the
    // 8400 anchors are background), avoiding the box/scale/NMS work for them.
    if (max_score < this->conf_threshold) {
      continue;
    }

    ObbBox box;
    box.cx = raw[0 * n + i];
    box.cy = raw[1 * n + i];
    box.w = raw[2 * n + i];
    box.h = raw[3 * n + i];
    box.angle = raw[angle_channel * n + i];
    box.score = max_score;
    box.class_id = class_id;
    if (box.w <= 0.0f || box.h <= 0.0f) {
      continue;
    }
    boxes.push_back(scale_obb(box, original_image_size, resized_image_size));
  }

  const auto indices =
      rotated_nms(boxes, this->iou_threshold, this->conf_threshold);

  for (const int idx : indices) {
    const ObbBox &b = boxes[static_cast<size_t>(idx)];
    yolo_msgs::msg::Detection detection;
    // OBB reuses BoundingBox2D: the rotation angle rides in `center.theta`
    // (radians), matching the upstream Python node's parse_boxes().
    detection.bbox.center.position.x = b.cx;
    detection.bbox.center.position.y = b.cy;
    detection.bbox.center.theta = b.angle;
    detection.bbox.size.x = b.w;
    detection.bbox.size.y = b.h;
    detection.score = b.score;
    detection.class_id = b.class_id;
    detection.id = "0";
    if (b.class_id < static_cast<int>(this->class_names.size())) {
      detection.class_name = this->class_names[b.class_id];
    } else {
      detection.class_name = "unknown";
    }
    detection_array.push_back(detection);
  }

  return detection_array;
}

} // namespace yolo_ros::yolo
