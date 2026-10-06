// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <vector>

#include "test_helpers.hpp"
#include "yolo_ros/yolo/utils.hpp"

namespace yolo_ros::yolo::utils {
namespace {

TEST(Iou, IdenticalBoxes) {
  Box a(0, 0, 10, 10, 1.0f, 0, 0);
  Box b(0, 0, 10, 10, 1.0f, 0, 0);
  EXPECT_FLOAT_EQ(iou(a, b), 1.0f);
}

TEST(Iou, DisjointBoxes) {
  Box a(0, 0, 10, 10, 1.0f, 0, 0);
  Box b(20, 20, 30, 30, 1.0f, 1, 0);
  EXPECT_FLOAT_EQ(iou(a, b), 0.0f);
}

TEST(Iou, PartialOverlap) {
  Box a(0, 0, 10, 10, 1.0f, 0, 0);
  Box b(0, 0, 5, 5, 1.0f, 1, 0);
  EXPECT_NEAR(iou(a, b), 0.25f, 1e-6f);
}

TEST(Iou, ZeroAreaUnionIsZero) {
  Box a(0, 0, 0, 0, 1.0f, 0, 0);
  Box b(0, 0, 0, 0, 1.0f, 1, 0);
  EXPECT_FLOAT_EQ(iou(a, b), 0.0f);
}

TEST(Nms, SuppressesOverlappingSameClass) {
  std::vector<Box> boxes = {
      Box(0, 0, 10, 10, 0.6f, 0, 0),
      Box(1, 1, 11, 11, 0.9f, 1, 0),
      Box(50, 50, 60, 60, 0.7f, 2, 0),
  };
  const auto keep = nms(boxes, 0.5f, 0.0f);
  ASSERT_EQ(keep.size(), 2u);
  EXPECT_FLOAT_EQ(boxes[keep[0]].score, 0.9f);
  EXPECT_FLOAT_EQ(boxes[keep[1]].score, 0.7f);
}

TEST(Nms, KeepsDifferentClasses) {
  std::vector<Box> boxes = {
      Box(0, 0, 10, 10, 0.9f, 0, 0),
      Box(0, 0, 10, 10, 0.9f, 1, 1),
  };
  EXPECT_EQ(nms(boxes, 0.5f, 0.0f).size(), 2u);
}

TEST(Nms, DropsBelowConfidence) {
  std::vector<Box> boxes = {
      Box(0, 0, 10, 10, 0.9f, 0, 0),
      Box(50, 50, 60, 60, 0.2f, 1, 0),
  };
  const auto keep = nms(boxes, 0.5f, 0.5f);
  ASSERT_EQ(keep.size(), 1u);
  EXPECT_FLOAT_EQ(boxes[keep[0]].score, 0.9f);
}

TEST(TensorHelper, OwnsShapeAndValues) {
  yolo_ros::test::Tensor tensor({1, 3}, {1.0f, 2.0f, 3.0f});
  const auto info = tensor.value().GetTensorTypeAndShapeInfo();
  EXPECT_EQ(info.GetShape(), (std::vector<int64_t>{1, 3}));
  const float *data = tensor.value().GetTensorData<float>();
  EXPECT_FLOAT_EQ(data[0], 1.0f);
  EXPECT_FLOAT_EQ(data[2], 3.0f);
}

TEST(Letterbox, PreservesAspectAndPads) {
  cv::Mat img(3, 4, CV_8UC3, cv::Scalar(255, 0, 0));
  const cv::Mat out = letterbox(img, cv::Size(8, 8), cv::Scalar(114, 114, 114));
  ASSERT_EQ(out.cols, 8);
  ASSERT_EQ(out.rows, 8);
  EXPECT_EQ(out.at<cv::Vec3b>(0, 0)[0], 114); // top pad row
  EXPECT_EQ(out.at<cv::Vec3b>(7, 7)[0], 114); // bottom pad row
  EXPECT_EQ(out.at<cv::Vec3b>(4, 4)[0], 255); // genuine content
}

TEST(BgrToRgb, SwapsRedAndBlue) {
  cv::Mat bgr(1, 1, CV_8UC3, cv::Scalar(10, 20, 30)); // B=10, G=20, R=30
  const cv::Mat rgb = bgr_to_rgb(bgr);
  const cv::Vec3b px = rgb.at<cv::Vec3b>(0, 0);
  EXPECT_EQ(px[0], 30);
  EXPECT_EQ(px[1], 20);
  EXPECT_EQ(px[2], 10);
}

TEST(InverseLetterbox, RestoresOriginalSize) {
  cv::Mat letterboxed(8, 8, CV_8U, cv::Scalar(200));
  letterboxed.rowRange(1, 7).setTo(cv::Scalar(50));
  const cv::Mat out =
      inverse_letterbox(letterboxed, cv::Size(4, 3), cv::Size(8, 8));
  ASSERT_EQ(out.cols, 4);
  ASSERT_EQ(out.rows, 3);
  double mn = 0.0;
  double mx = 0.0;
  cv::minMaxLoc(out, &mn, &mx);
  EXPECT_DOUBLE_EQ(mn, 50.0);
  EXPECT_DOUBLE_EQ(mx, 50.0);
}

TEST(ScaleBox, MapsLetterboxedFrameToOriginal) {
  // original 640x480 -> resized 640x640: gain 1, vertical pad 80.
  const Box box(0, 80, 640, 560, 0.8f, 3, 1);
  const Box out = scale_box(box, cv::Size(640, 480), cv::Size(640, 640));
  EXPECT_FLOAT_EQ(out.x1, 0.0f);
  EXPECT_FLOAT_EQ(out.y1, 0.0f);
  EXPECT_FLOAT_EQ(out.x2, 640.0f);
  EXPECT_FLOAT_EQ(out.y2, 480.0f);
  EXPECT_FLOAT_EQ(out.score, 0.8f);
  EXPECT_EQ(out.class_id, 1);
  EXPECT_EQ(out.index, 3);
}

TEST(ScaleBox, ClampsToImageBounds) {
  const Box box(-100, -100, 10000, 10000, 0.5f, 0, 0);
  const Box out = scale_box(box, cv::Size(640, 480), cv::Size(640, 640));
  EXPECT_FLOAT_EQ(out.x1, 0.0f);
  EXPECT_FLOAT_EQ(out.y1, 0.0f);
  EXPECT_FLOAT_EQ(out.x2, 640.0f);
  EXPECT_FLOAT_EQ(out.y2, 480.0f);
}

TEST(ScaleKeypoints, MapsAndClamps) {
  std::vector<Keypoint> kps(3);
  kps[0].x = 320.0f;
  kps[0].y = 320.0f;
  kps[0].visible = 0.9f;
  kps[1].x = -100.0f;
  kps[1].y = -100.0f;
  kps[1].visible = 0.5f;
  kps[2].x = 10000.0f;
  kps[2].y = 10000.0f;
  kps[2].visible = 0.1f;
  const auto out = scale_keypoints(kps, cv::Size(640, 480), cv::Size(640, 640));
  ASSERT_EQ(out.size(), 3u);
  EXPECT_FLOAT_EQ(out[0].x, 320.0f);
  EXPECT_FLOAT_EQ(out[0].y, 240.0f);
  EXPECT_FLOAT_EQ(out[0].visible, 0.9f);
  EXPECT_FLOAT_EQ(out[1].x, 0.0f);
  EXPECT_FLOAT_EQ(out[1].y, 0.0f);
  EXPECT_FLOAT_EQ(out[2].x, 640.0f);
  EXPECT_FLOAT_EQ(out[2].y, 480.0f);
}

TEST(ConvertToBoundingBox, CenterAndSize) {
  const Box box(0, 0, 10, 20, 0.5f, 0, 0);
  const auto msg = convert_to_bounding_box(box);
  EXPECT_FLOAT_EQ(msg.center.position.x, 5.0f);
  EXPECT_FLOAT_EQ(msg.center.position.y, 10.0f);
  EXPECT_FLOAT_EQ(msg.size.x, 10.0f);
  EXPECT_FLOAT_EQ(msg.size.y, 20.0f);
}

TEST(GetBoxes, DecodesAndFilters) {
  const int nc = 1;
  const size_t n = 2;
  const std::vector<int64_t> shape{1, 4 + nc, static_cast<int64_t>(n)};
  // Channel-major layout: [cx0, cx1, cy0, cy1, w0, w1, h0, h1, s0, s1].
  std::vector<float> data = {50.0f, 10.0f, 50.0f, 10.0f, 20.0f,
                             4.0f,  20.0f, 4.0f,  0.9f,  0.1f};
  yolo_ros::test::Tensor tensor(shape, data);
  std::vector<Ort::Value> preds;
  preds.push_back(std::move(tensor.value()));

  const auto boxes =
      get_boxes(preds, cv::Size(100, 100), cv::Size(100, 100), nc, 0.5f);
  ASSERT_EQ(boxes.size(), 1u);
  EXPECT_FLOAT_EQ(boxes[0].x1, 40.0f);
  EXPECT_FLOAT_EQ(boxes[0].y1, 40.0f);
  EXPECT_FLOAT_EQ(boxes[0].x2, 60.0f);
  EXPECT_FLOAT_EQ(boxes[0].y2, 60.0f);
  EXPECT_FLOAT_EQ(boxes[0].score, 0.9f);
  EXPECT_EQ(boxes[0].class_id, 0);
  EXPECT_EQ(boxes[0].index, 0);
}

TEST(GetBoxes, SelectsArgmaxClass) {
  const int nc = 2;
  const std::vector<int64_t> shape{1, 4 + nc, 1};
  // One anchor: cx, cy, w, h, class0 score, class1 score.
  std::vector<float> data = {50.0f, 50.0f, 20.0f, 20.0f, 0.2f, 0.8f};
  yolo_ros::test::Tensor tensor(shape, data);
  std::vector<Ort::Value> preds;
  preds.push_back(std::move(tensor.value()));

  const auto boxes =
      get_boxes(preds, cv::Size(100, 100), cv::Size(100, 100), nc, 0.5f);
  ASSERT_EQ(boxes.size(), 1u);
  EXPECT_EQ(boxes[0].class_id, 1);
  EXPECT_FLOAT_EQ(boxes[0].score, 0.8f);
  EXPECT_EQ(boxes[0].index, 0);
}

} // namespace
} // namespace yolo_ros::yolo::utils
