// Copyright (c) 2025 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include "yolo_ros/yolo/utils.hpp"
#include <algorithm>
#include <cstdio>

namespace yolo_ros::yolo::utils {
cv::Mat letterbox(const cv::Mat &img, const cv::Size &new_shape,
                  const cv::Scalar &color) {
  float ratio = std::min(static_cast<float>(new_shape.width) / img.cols,
                         static_cast<float>(new_shape.height) / img.rows);
  cv::Mat img_out;

  int new_width = static_cast<int>(img.cols * ratio);
  int new_height = static_cast<int>(img.rows * ratio);

  int pad_w = new_shape.width - new_width;
  int pad_h = new_shape.height - new_height;

  int pad_left = pad_w / 2;
  int pad_right = pad_w - pad_left;
  int pad_top = pad_h / 2;
  int pad_bottom = pad_h - pad_top;

  cv::resize(img, img_out, cv::Size(new_width, new_height), 0, 0,
             cv::INTER_LINEAR);
  cv::copyMakeBorder(img_out, img_out, pad_top, pad_bottom, pad_left, pad_right,
                     cv::BORDER_CONSTANT, color);
  return img_out;
}

cv::Mat bgr_to_rgb(const cv::Mat &img) {
  cv::Mat rgb;
  cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
  return rgb;
}

yolo_ros::yolo::utils::Box scale_box(const yolo_ros::yolo::utils::Box &box,
                                     const cv::Size &original_image_size,
                                     const cv::Size &resized_image_size) {
  yolo_ros::yolo::utils::Box scaled_box;
  float gain = std::min(static_cast<float>(resized_image_size.width) /
                            original_image_size.width,
                        static_cast<float>(resized_image_size.height) /
                            original_image_size.height);
  float pad_x =
      (resized_image_size.width - original_image_size.width * gain) / 2;
  float pad_y =
      (resized_image_size.height - original_image_size.height * gain) / 2;

  scaled_box.x1 = (box.x1 - pad_x) / gain;
  scaled_box.y1 = (box.y1 - pad_y) / gain;
  scaled_box.x2 = (box.x2 - pad_x) / gain;
  scaled_box.y2 = (box.y2 - pad_y) / gain;

  scaled_box.x1 = std::clamp(scaled_box.x1, 0.0f,
                             static_cast<float>(original_image_size.width));
  scaled_box.y1 = std::clamp(scaled_box.y1, 0.0f,
                             static_cast<float>(original_image_size.height));
  scaled_box.x2 = std::clamp(scaled_box.x2, 0.0f,
                             static_cast<float>(original_image_size.width));
  scaled_box.y2 = std::clamp(scaled_box.y2, 0.0f,
                             static_cast<float>(original_image_size.height));

  scaled_box.score = box.score;
  scaled_box.class_id = box.class_id;
  scaled_box.index = box.index;

  return scaled_box;
}

std::vector<Keypoint> scale_keypoints(const std::vector<Keypoint> &keypoints,
                                      const cv::Size &original_image_size,
                                      const cv::Size &resized_image_size) {
  // Same inverse-letterbox geometry as scale_box(): the exported graph decodes
  // keypoints into the letterboxed (model input) frame, so undo the padding
  // and resize to map back to the original image.
  const float gain = std::min(static_cast<float>(resized_image_size.width) /
                                  original_image_size.width,
                              static_cast<float>(resized_image_size.height) /
                                  original_image_size.height);
  const float pad_x =
      (resized_image_size.width - original_image_size.width * gain) / 2;
  const float pad_y =
      (resized_image_size.height - original_image_size.height * gain) / 2;

  std::vector<Keypoint> scaled = keypoints;
  for (auto &kp : scaled) {
    kp.x = std::clamp((kp.x - pad_x) / gain, 0.0f,
                      static_cast<float>(original_image_size.width));
    kp.y = std::clamp((kp.y - pad_y) / gain, 0.0f,
                      static_cast<float>(original_image_size.height));
  }
  return scaled;
}

std::vector<yolo_ros::yolo::utils::Box>
get_boxes(const std::vector<Ort::Value> &preds,
          const cv::Size &original_image_size,
          const cv::Size &resized_image_size, const int num_classes,
          const float conf_threshold) {
  std::vector<yolo_ros::yolo::utils::Box> boxes;

  const float *raw_output =
      preds[0].GetTensorData<float>(); // Extract raw output data from the
                                       // first output tensor
  const size_t num_detections =
      preds[0].GetTensorTypeAndShapeInfo().GetShape()[2];

  const float *ptr = raw_output;
  for (size_t i = 0; i < num_detections; ++i) {
    int class_id = -1;
    float max_score = -1.0f;

    for (int j = 0; j < num_classes; ++j) {
      const float score = ptr[(4 + j) * num_detections + i];
      if (score > max_score) {
        max_score = score;
        class_id = j;
      }
    }

    // Skip anchors whose best class score is below the confidence threshold.
    // Most of the 8400 anchors are background, so this avoids building and
    // scaling a box (and later NMS work) for them.
    if (max_score < conf_threshold) {
      continue;
    }

    yolo_ros::yolo::utils::Box box;
    const float center_x = ptr[0 * num_detections + i];
    const float center_y = ptr[1 * num_detections + i];
    const float width = ptr[2 * num_detections + i];
    const float height = ptr[3 * num_detections + i];

    box.x1 = (center_x - width / 2);
    box.y1 = (center_y - height / 2);
    box.x2 = (center_x + width / 2);
    box.y2 = (center_y + height / 2);
    box.score = max_score;
    box.class_id = class_id;
    box.index = static_cast<int>(i);

    yolo_ros::yolo::utils::Box scaled_box = yolo_ros::yolo::utils::scale_box(
        box, original_image_size, resized_image_size);
    boxes.push_back(scaled_box);
  }

  return boxes;
}

yolo_msgs::msg::BoundingBox2D
convert_to_bounding_box(const yolo_ros::yolo::utils::Box &box) {
  yolo_msgs::msg::BoundingBox2D bounding_box;
  bounding_box.center.position.x = (box.x1 + box.x2) / 2;
  bounding_box.center.position.y = (box.y1 + box.y2) / 2;
  bounding_box.size.x = box.x2 - box.x1;
  bounding_box.size.y = box.y2 - box.y1;
  return bounding_box;
}

cv::Mat inverse_letterbox(const cv::Mat &letterboxed,
                          const cv::Size &original_image_size,
                          const cv::Size &resized_image_size) {
  // Resize the (low-res) mask up to the letterboxed / model-input frame.
  cv::Mat resized_image;
  cv::resize(letterboxed, resized_image, resized_image_size, 0, 0,
             cv::INTER_LINEAR);

  // Size of the actual (unpadded) image content inside the letterboxed frame.
  float scale = std::min(static_cast<float>(resized_image_size.width) /
                             original_image_size.width,
                         static_cast<float>(resized_image_size.height) /
                             original_image_size.height);
  int new_w = static_cast<int>(original_image_size.width * scale);
  int new_h = static_cast<int>(original_image_size.height * scale);

  int pad_x = (resized_image_size.width - new_w) / 2;
  int pad_y = (resized_image_size.height - new_h) / 2;

  // Crop only the unpadded content, then rescale back to the original size.
  cv::Rect content_roi(pad_x, pad_y, new_w, new_h);
  cv::Mat cropped = resized_image(content_roi);

  cv::Mat restored;
  cv::resize(cropped, restored, original_image_size, 0, 0, cv::INTER_LINEAR);

  return restored;
}
} // namespace yolo_ros::yolo::utils