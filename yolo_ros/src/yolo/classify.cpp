// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include "yolo_ros/yolo/classify.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace yolo_ros::yolo {

YoloClassify::YoloClassify(yolo_ros::yolo::utils::YoloParams params)
    : yolo_ros::engine::Model(params, "classify") {
  this->top_k_ = params.top_k;
}

YoloClassify::~YoloClassify() {}

std::vector<yolo_msgs::msg::Detection>
YoloClassify::postprocess(const cv::Size &, const cv::Size &,
                          const std::vector<Ort::Value> &preds) {
  std::vector<yolo_msgs::msg::Detection> detections;

  if (preds.empty()) {
    return detections;
  }

  const std::vector<int64_t> shape =
      preds[0].GetTensorTypeAndShapeInfo().GetShape();

  // The classification output is a single [1, N] score row (some exporters
  // add extra singleton dims, e.g. [1, 1, N]); the number of classes is the
  // product of every dimension after the batch one.
  size_t num_classes = 1;

  for (size_t i = 1; i < shape.size(); ++i) {
    num_classes *= static_cast<size_t>(shape[i]);
  }

  if (num_classes == 0) {
    return detections;
  }

  const float *raw = preds[0].GetTensorData<float>();
  std::vector<float> scores(raw, raw + num_classes);

  // Ultralytics cls exports bake the softmax into the ONNX graph, so the
  // output is already a probability distribution: re-applying a softmax here
  // (double softmax) would flatten it into near-uniform noise. Detect that
  // case (all values in [0, 1] and summing to ~1) and only softmax when the
  // graph emitted raw logits (e.g. hand-rolled cls exports).
  const float sum = std::accumulate(scores.begin(), scores.end(), 0.0f,
                                    [](float a, float b) { return a + b; });
  bool already_probs = std::isfinite(sum) && std::abs(sum - 1.0f) < 1e-2f;

  if (already_probs) {
    already_probs = std::all_of(scores.begin(), scores.end(),
                                [](float s) { return s >= 0.0f && s <= 1.0f; });
  }

  if (!already_probs) {
    const float max_score = *std::max_element(scores.begin(), scores.end());
    float exp_sum = 0.0f;

    for (float &s : scores) {
      s = std::exp(s - max_score);
      exp_sum += s;
    }

    for (float &s : scores) {
      s /= exp_sum;
    }
  }

  // Top-k classes, sorted by descending probability. A cls model always has
  // an argmax, so the array is never empty: it holds min(top_k, N) entries
  // and consumers can floor on `score` (conf_threshold is not applied here,
  // unlike the detection postprocessors).
  const size_t k = std::min(static_cast<size_t>(this->top_k_), num_classes);
  std::vector<size_t> indices(num_classes);
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(
      indices.begin(), indices.begin() + k, indices.end(),
      [&scores](size_t a, size_t b) { return scores[a] > scores[b]; });

  detections.reserve(k);

  for (size_t i = 0; i < k; ++i) {
    const size_t class_id = indices[i];
    yolo_msgs::msg::Detection detection;
    // No bounding box: an image-level label has no spatial extent. The rest
    // of the fields (bbox/mask/keypoints) stay at their message defaults.
    detection.score = scores[class_id];
    detection.class_id = static_cast<int32_t>(class_id);
    detection.id = "0";

    if (class_id < this->class_names.size()) {
      detection.class_name = this->class_names[class_id];
    } else {
      detection.class_name = "unknown";
    }

    detections.push_back(detection);
  }

  return detections;
}

} // namespace yolo_ros::yolo