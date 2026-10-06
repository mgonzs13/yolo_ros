// Copyright (c) 2025 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include "yolo_ros/yolo/segment.hpp"
#include "yolo_msgs/msg/point2_d.hpp"
#include "yolo_ros/utils/logs.hpp"
#include "yolo_ros/yolo/utils.hpp"

#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace yolo_ros::yolo {

namespace {

/// Prototype masks emitted by the segmentation head.
constexpr int kProtos = 32;
/// Threshold applied to the sigmoid'd prototype blend to binarise an instance
/// mask.
constexpr double kMaskThreshold = 0.5;

/// Blend the prototype masks for every masked box, extract the instance
/// boundary and fill the Detection messages.
/// @param[in] bbox_array NMS-filtered boxes carrying their mask coefficients.
/// @param[in] preds Raw output tensors (preds[1] holds the prototypes).
/// @param[in] original_image_size Size of the original camera image.
/// @param[in] resized_image_size Size of the letterboxed network input.
/// @param[in] class_names Class vocabulary indexed by class id.
/// @return One Detection per box, each carrying its mask contour.
std::vector<yolo_msgs::msg::Detection> masks_to_detections(
    const std::vector<yolo_ros::yolo::utils::BoxWithMask> &bbox_array,
    const std::vector<Ort::Value> &preds, const cv::Size &original_image_size,
    const cv::Size &resized_image_size,
    const std::vector<std::string> &class_names) {
  std::vector<yolo_msgs::msg::Detection> detection_array;

  const std::vector<int64_t> mask_shape =
      preds[1].GetTensorTypeAndShapeInfo().GetShape(); // [1, 32, maskH, maskW]
  const int mask_h = static_cast<int>(mask_shape[2]);
  const int mask_w = static_cast<int>(mask_shape[3]);

  std::vector<cv::Mat> mask_protos;
  const float *mask_ptr = preds[1].GetTensorData<float>();
  for (int64_t i = 0; i < mask_shape[1]; ++i) {
    cv::Mat mask(mask_h, mask_w, CV_32F,
                 const_cast<float *>(mask_ptr + i * mask_h * mask_w));
    mask_protos.push_back(mask);
  }

  std::vector<std::vector<cv::Point>> masks;
  for (size_t i = 0; i < bbox_array.size(); ++i) {
    const auto &mask_coeffs = bbox_array[i].mask_coeffs;
    cv::Mat seg_mask = cv::Mat::zeros(mask_h, mask_w, CV_32F);

    // Linear combination of prototype masks
    for (int m = 0; m < kProtos; ++m) {
      seg_mask += mask_coeffs[m] * mask_protos[m];
    }

    // Apply sigmoid activation
    cv::exp(-seg_mask, seg_mask);
    seg_mask = 1.0 / (1.0 + seg_mask);

    // Resize the *probability* mask to the original image, then binarise it.
    // Interpolating the float field before thresholding follows the true
    // iso-contour; upscaling an already-binary 160x160 mask instead leaves a
    // stair-stepped boundary (and inflates it, since findContours treats any
    // interpolated non-zero as foreground).
    cv::Mat resized_mask = yolo_ros::yolo::utils::inverse_letterbox(
        seg_mask, original_image_size, resized_image_size);
    cv::Mat filtered_seg_mask;
    cv::threshold(resized_mask, filtered_seg_mask, kMaskThreshold, 255.0,
                  cv::THRESH_BINARY);
    filtered_seg_mask.convertTo(filtered_seg_mask, CV_8U);

    // Crop to bounding box
    cv::Rect roi(bbox_array[i].x1, bbox_array[i].y1,
                 bbox_array[i].x2 - bbox_array[i].x1,
                 bbox_array[i].y2 - bbox_array[i].y1);
    roi &= cv::Rect(0, 0, filtered_seg_mask.cols, filtered_seg_mask.rows);
    cv::Mat cropped_mask = cv::Mat::zeros(filtered_seg_mask.size(), CV_8U);
    if (roi.area() > 0) {
      filtered_seg_mask(roi).copyTo(cropped_mask(roi));
    }

    // Find contours in the cropped mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(cropped_mask, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    // Find the largest contour
    double max_area = 0;
    int max_contour_index = -1;
    for (size_t j = 0; j < contours.size(); ++j) {
      double area = cv::contourArea(contours[j]);
      if (area > max_area) {
        max_area = area;
        max_contour_index = static_cast<int>(j);
      }
    }
    if (max_contour_index != -1) {
      masks.push_back(contours[max_contour_index]);
    } else {
      masks.push_back(std::vector<cv::Point>());
    }
  }

  for (size_t i = 0; i < bbox_array.size(); ++i) {
    yolo_msgs::msg::Detection detection;
    detection.bbox =
        yolo_ros::yolo::utils::convert_to_bounding_box(bbox_array[i]);

    // Segmentation mask
    detection.mask.height = original_image_size.height;
    detection.mask.width = original_image_size.width;
    detection.mask.data.clear();
    for (const auto &point : masks[i]) {
      yolo_msgs::msg::Point2D point2d;
      point2d.x = point.x;
      point2d.y = point.y;
      detection.mask.data.push_back(point2d);
    }

    // Some additional information
    detection.score = bbox_array[i].score;
    detection.class_id = bbox_array[i].class_id;
    detection.id = "0";
    if (bbox_array[i].class_id < static_cast<int>(class_names.size())) {
      detection.class_name = class_names[bbox_array[i].class_id];
    } else {
      detection.class_name = "unknown";
    }
    detection_array.push_back(detection);
  }

  return detection_array;
}

} // namespace

YoloSegment::YoloSegment(yolo_ros::yolo::utils::YoloParams params)
    : yolo_ros::engine::Model(params, "segment") {}

YoloSegment::~YoloSegment() {}

std::vector<yolo_msgs::msg::Detection>
YoloSegment::postprocess(const cv::Size &original_image_size,
                         const cv::Size &resized_image_size,
                         const std::vector<Ort::Value> &preds) {
  const std::vector<int64_t> shape =
      preds[0].GetTensorTypeAndShapeInfo().GetShape();

  // Two exported layouts are supported, mirroring the detect/pose nodes:
  //  - End-to-end/baked-head segment models (yolo26 family exported with the
  //    post-process in-graph) output [*, K, 6 + n_protos]: each row is
  //    [x1, y1, x2, y2, score, class_id, coeffs...] and NMS is already applied.
  //  - Classic exports (yolov8/v11, and yolo26 with end2end=False) output
  //    [1, 4 + nc + n_protos, D]: one column per anchor, NMS done here.
  const bool baked_head = shape.size() >= 2 && shape.back() == 4 + 2 + kProtos;

  std::vector<yolo_ros::yolo::utils::BoxWithMask> bbox_array;
  if (baked_head) {
    bbox_array = get_segmentation_baked_head(
        preds, original_image_size, resized_image_size, this->conf_threshold);
  } else {
    const size_t num_features = static_cast<size_t>(shape[1]);
    const int num_classes = static_cast<int>(num_features) - 4 - kProtos;
    if (num_classes < 1) {
      YOLO_LOG_WARN("YoloSegment: unexpected output feature count %zu; "
                    "expected 4 + nc + %d",
                    num_features, kProtos);
      return {};
    }

    bbox_array = get_segmentation_with_nms(
        preds, original_image_size, resized_image_size, num_classes,
        this->iou_threshold, this->conf_threshold);
  }

  return masks_to_detections(bbox_array, preds, original_image_size,
                             resized_image_size, this->class_names);
}

std::vector<yolo_ros::yolo::utils::BoxWithMask> get_segmentation_baked_head(
    const std::vector<Ort::Value> &preds, const cv::Size &original_image_size,
    const cv::Size &resized_image_size, const float conf_threshold) {
  std::vector<yolo_ros::yolo::utils::BoxWithMask> boxes;

  const std::vector<int64_t> shape =
      preds[0].GetTensorTypeAndShapeInfo().GetShape();
  const float *raw = preds[0].GetTensorData<float>();

  // [1, K, 6 + n_protos] (or [K, 6 + n_protos]): rows are detections and the
  // last dimension holds the features.
  const size_t num_rows = shape.size() >= 3
                              ? static_cast<size_t>(shape[shape.size() - 2])
                              : static_cast<size_t>(shape[0]);
  const size_t stride = static_cast<size_t>(shape.back());

  for (size_t i = 0; i < num_rows; ++i) {
    const float *row = raw + i * stride;
    if (row[4] < conf_threshold) {
      continue;
    }

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

    std::vector<float> mask_coeffs(static_cast<size_t>(kProtos));
    for (int m = 0; m < kProtos; ++m) {
      mask_coeffs[static_cast<size_t>(m)] = row[6 + m];
    }

    boxes.emplace_back(box, std::move(mask_coeffs));
  }

  return boxes;
}

std::vector<yolo_ros::yolo::utils::BoxWithMask> get_segmentation_with_nms(
    const std::vector<Ort::Value> &preds, const cv::Size &original_image_size,
    const cv::Size &resized_image_size, const int num_classes,
    float iou_threshold, float conf_threshold) {

  const float *raw_output =
      preds[0].GetTensorData<float>(); // Extract raw output data from the
  const size_t num_detections =
      preds[0].GetTensorTypeAndShapeInfo().GetShape()[2];

  std::vector<yolo_ros::yolo::utils::BoxWithMask> seg_boxes;

  // 1. Get the bounding boxes (anchors below conf_threshold are skipped)
  std::vector<yolo_ros::yolo::utils::Box> boxes =
      yolo_ros::yolo::utils::get_boxes(preds, original_image_size,
                                       resized_image_size, num_classes,
                                       conf_threshold);

  // 2. Add the mask coefficients to the boxes
  std::vector<yolo_ros::yolo::utils::BoxWithMask> boxes_with_mask;
  for (size_t i = 0; i < boxes.size(); ++i) {
    if (boxes[i].score < conf_threshold) {
      continue;
    }
    yolo_ros::yolo::utils::BoxWithMask box_with_mask(boxes[i]);
    std::vector<float> mask_coeffs(kProtos);
    // get_boxes() returns a *compacted* list (anchors below the confidence
    // threshold are dropped), so the coefficient column must be addressed by
    // the box's original anchor index, not by its position in this vector.
    const size_t anchor = static_cast<size_t>(boxes[i].index);
    for (int m = 0; m < kProtos; ++m) {
      mask_coeffs[static_cast<size_t>(m)] =
          raw_output[(num_classes + 4 + m) * num_detections + anchor];
    }
    box_with_mask.mask_coeffs = mask_coeffs;
    boxes_with_mask.push_back(box_with_mask);
  }

  // BoxesWithMask are sorted in place by nms(); indices index the same vector.
  auto indices = yolo_ros::yolo::utils::nms(boxes_with_mask, iou_threshold,
                                            conf_threshold);

  std::vector<yolo_ros::yolo::utils::BoxWithMask> filtered_boxes;
  for (size_t i = 0; i < indices.size(); ++i) {
    filtered_boxes.push_back(boxes_with_mask[indices[i]]);
  }

  return filtered_boxes;
}

} // namespace yolo_ros::yolo
