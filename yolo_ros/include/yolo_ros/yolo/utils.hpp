// Copyright (c) 2025 Alejandro González Cantón
// SPDX-License-Identifier: MIT

/// @file
/// @brief Shared box, keypoint and parameter types plus the IoU/NMS,
/// letterbox and coordinate-scaling helpers used by every YOLO task.

#ifndef YOLO_ROS__YOLO__UTILS_HPP_
#define YOLO_ROS__YOLO__UTILS_HPP_

#include "onnxruntime_cxx_api.h"
#include "yolo_msgs/msg/bounding_box2_d.hpp"
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <vector>

/// @addtogroup yolo_tasks
/// @{
namespace yolo_ros::yolo::utils {
/// @brief Axis-aligned detection box in original-image pixel coordinates
/// (top-left x1/y1, bottom-right x2/y2), with its confidence, position in the
/// frame's detection list and class id.
struct Box {
  float x1;     ///< Top-left x.
  float y1;     ///< Top-left y.
  float x2;     ///< Bottom-right x.
  float y2;     ///< Bottom-right y.
  float score;  ///< Confidence in [0, 1].
  int index;    ///< Detection index.
  int class_id; ///< Zero-based class id.

  /// @brief Construct a box from its corners, index and class.
  /// @param x1 Top-left x.
  /// @param y1 Top-left y.
  /// @param x2 Bottom-right x.
  /// @param y2 Bottom-right y.
  /// @param score Confidence in [0, 1].
  /// @param index Index of this detection in the frame's detection list.
  /// @param class_id Zero-based class id.
  Box(float x1, float y1, float x2, float y2, float score, int index,
      int class_id)
      : x1(x1), y1(y1), x2(x2), y2(y2), score(score), index(index),
        class_id(class_id) {}
  /// @brief Default-construct a zeroed box.
  Box() = default;
  /// @brief Virtual destructor (allows keypoint/mask subclasses).
  virtual ~Box() = default;
};

/// @brief A Box carrying the mask coefficients of a segmentation prototype
/// blend (one coefficient per prototype mask).
struct BoxWithMask : public Box {
  /// @brief Mask coefficients, combined with the prototype masks to obtain the
  /// instance mask.
  std::vector<float> mask_coeffs;

  /// @brief Default-construct an empty masked box.
  BoxWithMask() = default;
  /// @brief Promote a plain Box to a masked box with no coefficients.
  /// @param box Source box.
  BoxWithMask(Box box)
      : Box(box.x1, box.y1, box.x2, box.y2, box.score, box.index, box.class_id),
        mask_coeffs(std::vector<float>()) {}
  /// @brief Promote a plain Box and attach mask coefficients.
  /// @param box Source box.
  /// @param mask_coeffs Mask coefficients.
  BoxWithMask(Box box, std::vector<float> mask_coeffs)
      : Box(box.x1, box.y1, box.x2, box.y2, box.score, box.index, box.class_id),
        mask_coeffs(mask_coeffs) {}
};

// A single 2D pose keypoint. `visible` is the sigmoid'd visibility already
// decoded by the exported ONNX graph; it doubles as the per-keypoint
// confidence and, compared against conf_threshold, as the publish filter
// (matches the Python node, which drops keypoints with conf < threshold).
/// @brief A single 2D pose keypoint in image coordinates.
struct Keypoint {
  float x = 0.0f; ///< Keypoint x (pixels).
  float y = 0.0f; ///< Keypoint y (pixels).
  /// @brief Keypoint visibility/confidence in [0, 1], already sigmoid-decoded
  /// by the exported graph.
  float visible = 0.0f;
};

/// @brief A Box carrying the pose keypoints of one person.
struct BoxWithKeypoints : public Box {
  /// @brief Keypoints in COCO order, (x, y, visible) per keypoint.
  std::vector<Keypoint> keypoints; // COCO order, (x, y, visible) per kp

  /// @brief Default-construct an empty pose box.
  BoxWithKeypoints() = default;
  /// @brief Promote a plain Box to a pose box with no keypoints.
  /// @param box Source box.
  BoxWithKeypoints(Box box)
      : Box(box.x1, box.y1, box.x2, box.y2, box.score, box.index, box.class_id),
        keypoints(std::vector<Keypoint>()) {}
  /// @brief Promote a plain Box and attach keypoints.
  /// @param box Source box.
  /// @param keypoints Keypoints in COCO order.
  BoxWithKeypoints(Box box, std::vector<Keypoint> keypoints)
      : Box(box.x1, box.y1, box.x2, box.y2, box.score, box.index, box.class_id),
        keypoints(std::move(keypoints)) {}
};

/// @brief Configuration shared by every YOLO model wrapper (model source,
/// device, thresholds and the node-level options).
struct YoloParams {
  /// @brief Task selector: "YOLO"/"Detect"/"Segment"/"Pose"/"OBB"/"Classify"
  /// or "auto" (filename heuristic). Case-insensitive.
  std::string model_type;
  /// @brief Path to the local .onnx model.
  std::string model_path;
  // Hugging Face Hub download: when both model_repo and model_filename are
  // set, the model is downloaded (or reused from the HF cache) and its path
  // overrides model_path at configure time.
  /// @brief Hugging Face Hub repo id, e.g. "agonzc34/yolo26m".
  std::string model_repo; // HF repo id, e.g. "agonzc34/yolo26m"
  /// @brief File inside the Hugging Face repo, e.g. "yolo26m.onnx".
  std::string model_filename; // file inside the repo, e.g. "yolo26m.onnx"
  /// @brief Hugging Face cache directory, default ~/.cache/huggingface/hub.
  std::string cache_dir; // HF cache dir, default ~/.cache/huggingface/hub
  /// @brief Re-download the model even when a cached copy exists.
  bool force_download = false; // re-download even if cached
  /// @brief Execution provider preference: "auto" (CUDA -> CPU fallback chain)
  /// or "tensorrt"/"trt" (TensorRT -> CUDA -> CPU), "cuda" (CUDA -> CPU),
  /// "cpu" (case-insensitive).
  std::string provider = "auto";
  /// @brief Execution device; its ordinal (e.g. "cuda:0", "trt:1", "1") is
  /// passed as device_id to the CUDA and TensorRT execution providers.
  std::string device;
  /// @brief TensorRT: enable FP16 precision.
  bool trt_fp16_enable = true;
  /// @brief TensorRT: persist the built engine across sessions.
  bool trt_engine_cache_enable = true;
  /// @brief TensorRT: engine cache base directory (empty ->
  /// ~/.cache/yolo_ros/trt_engines).
  std::string trt_engine_cache_path;
  /// @brief Detection confidence threshold in [0, 1].
  float threshold;
  /// @brief IoU threshold for the C++ NMS.
  float iou;
  /// @brief Gate inference, matching the Python node's `enable` parameter.
  bool enable; // gate inference (matches the Python node's `enable`)
  /// @brief Cap on the number of detections published per image.
  int max_det; // cap on the number of detections published per image
  /// @brief Image subscription reliability (QoS) policy.
  int image_reliability;
  /// @brief Image topic to subscribe to.
  std::string image_topic;
  /// @brief Number of intra-op threads used by ONNX Runtime.
  int n_threads;
  /// @brief Cap on the inference/publish rate in Hz; 0 = unlimited. Frames are
  /// dropped by the node while the subscription stays live.
  int max_fps; // cap on the inference/publish rate in Hz; 0 = unlimited
               // (process every received frame). Frames are dropped by the
               // node, the subscription stays live.
  /// @brief Classification only: number of top classes published per image
  /// (softmax probabilities, sorted descending).
  int top_k = 5; // classification: number of top classes to publish per
                 // image (softmax probabilities, sorted descending)
  /// @brief Expected input channel order: "rgb" (default; ultralytics exports
  /// expect RGB) or "bgr" (OpenCV default). ONNX graphs do not encode this; the
  /// node reads it from the ONNX metadata ("input_color") when present and
  /// otherwise uses this parameter.
  std::string input_color = "rgb";
};

/// @brief Intersection over union of two boxes.
/// @tparam BoxT Box-like type exposing x1, y1, x2, y2.
/// @param[in] box1 First box.
/// @param[in] box2 Second box.
/// @return IoU in [0, 1], or 0 when the union area is zero.
template <typename BoxT> float iou(const BoxT &box1, const BoxT &box2) {
  float x1 = std::max(box1.x1, box2.x1);
  float y1 = std::max(box1.y1, box2.y1);
  float x2 = std::min(box1.x2, box2.x2);
  float y2 = std::min(box1.y2, box2.y2);

  float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  float area1 =
      std::max(0.0f, box1.x2 - box1.x1) * std::max(0.0f, box1.y2 - box1.y1);
  float area2 =
      std::max(0.0f, box2.x2 - box2.x1) * std::max(0.0f, box2.y2 - box2.y1);
  float union_area = area1 + area2 - intersection;

  return union_area > 0 ? intersection / union_area : 0;
}

/// @brief Per-class non-maximum suppression.
///
/// Sorts @p boxes in place by confidence, then returns the kept indices into
/// the *same* (now-sorted) vector, so the caller reads `boxes[indices[i]]`.
/// Works for Box or BoxWithMask (any type exposing `.score`, `.class_id` and
/// `.x1..y2`). No per-candidate heap allocation.
/// @tparam BoxT Box-like type exposing score, class_id and x1..y2.
/// @param[in,out] boxes Candidate boxes; sorted in place by score.
/// @param[in] iou_threshold Suppression threshold in [0, 1].
/// @param[in] conf_threshold Minimum score to consider a box.
/// @return Indices of the kept boxes into the sorted @p boxes vector.
template <typename BoxT>
std::vector<int> nms(std::vector<BoxT> &boxes, float iou_threshold,
                     float conf_threshold) {
  std::vector<int> indices;

  std::sort(boxes.begin(), boxes.end(),
            [](const BoxT &a, const BoxT &b) { return a.score > b.score; });

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
      if (iou(boxes[i], boxes[j]) > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }

  return indices;
}

/// @brief Resize @p img into @p new_shape preserving aspect ratio, padding the
/// remainder with @p color.
/// @param[in] img Source BGR image.
/// @param[in] new_shape Target network input size.
/// @param[in] color Padding color.
/// @return The letterboxed image.
cv::Mat letterbox(const cv::Mat &img, const cv::Size &new_shape,
                  const cv::Scalar &color);
/// @brief Convert an OpenCV BGR image to RGB (channel order only).
/// @param[in] img Source BGR image.
/// @return A new RGB image.
cv::Mat bgr_to_rgb(const cv::Mat &img);
/// @brief Crop/pad a letterboxed image back to the original image dimensions.
/// @param[in] letterboxed Letterboxed image.
/// @param[in] original_image_size Size of the original camera image.
/// @param[in] resized_image_size Size of the letterboxed network input.
/// @return The image with the letterbox padding removed.
cv::Mat inverse_letterbox(const cv::Mat &letterboxed,
                          const cv::Size &original_image_size,
                          const cv::Size &resized_image_size);
/// @brief Scale a box from the letterboxed model-input frame back to the
/// original image frame.
/// @param[in] box Box in network-input coordinates.
/// @param[in] original_image_size Size of the original camera image.
/// @param[in] resized_image_size Size of the letterboxed network input.
/// @return The box in original-image coordinates.
Box scale_box(const Box &box, const cv::Size &original_image_size,
              const cv::Size &resized_image_size);
/// @brief Scale keypoints from the letterboxed model-input frame back to the
/// original image frame.
/// @param[in] keypoints Keypoints in network-input coordinates.
/// @param[in] original_image_size Size of the original camera image.
/// @param[in] resized_image_size Size of the letterboxed network input.
/// @return The keypoints in original-image coordinates.
std::vector<Keypoint> scale_keypoints(const std::vector<Keypoint> &keypoints,
                                      const cv::Size &original_image_size,
                                      const cv::Size &resized_image_size);
/// @brief Convert an internal Box to the public BoundingBox2D message.
/// @param[in] box Box to convert.
/// @return The equivalent BoundingBox2D.
yolo_msgs::msg::BoundingBox2D
convert_to_bounding_box(const yolo_ros::yolo::utils::Box &box);

/// @brief Decode raw YOLO detection tensors into boxes without NMS.
/// @param[in] preds Raw output tensors from the model.
/// @param[in] original_image_size Size of the original camera image.
/// @param[in] resized_image_size Size of the letterboxed network input.
/// @param[in] num_classes Number of model classes.
/// @param[in] conf_threshold Minimum score to keep a box.
/// @return Candidate boxes in original-image coordinates.
std::vector<yolo_ros::yolo::utils::Box>
get_boxes(const std::vector<Ort::Value> &preds,
          const cv::Size &original_image_size,
          const cv::Size &resized_image_size, const int num_classes,
          const float conf_threshold);

} // namespace yolo_ros::yolo::utils
/// @}

#endif // YOLO_ROS__YOLO__UTILS_HPP_
