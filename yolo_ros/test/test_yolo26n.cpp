// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "test_helpers.hpp"
#include "test_model_helpers.hpp"
#include "yolo_ros/yolo/classify.hpp"
#include "yolo_ros/yolo/detect.hpp"
#include "yolo_ros/yolo/pose.hpp"
#include "yolo_ros/yolo/segment.hpp"

namespace yolo_ros::yolo {
namespace {

constexpr char kRepo[] = "zwh20081/yolo26-onnx";

class ExposedDetect : public YoloDetect {
public:
  using YoloDetect::YoloDetect;
  std::size_t num_class_names() const { return class_names.size(); }
};

class Detect26nTest : public ::testing::Test {
protected:
  void SetUp() override {
    const std::string model =
        yolo_ros::test::download_model(kRepo, "yolo26n.onnx");
    image_ = yolo_ros::test::download_sample_image();
    if (model.empty() || image_.empty()) {
      GTEST_SKIP() << "yolo26n.onnx or the sample image is unavailable";
    }
    model_path_ = model;
    detector_ =
        std::make_unique<ExposedDetect>(yolo_ros::test::make_params(model));
  }

  cv::Mat image_;
  std::string model_path_;
  std::unique_ptr<ExposedDetect> detector_;
};

TEST_F(Detect26nTest, LoadsCocoVocabulary) {
  EXPECT_EQ(detector_->num_class_names(), 80u);
}

TEST_F(Detect26nTest, ProducesWellFormedDetections) {
  const auto dets = detector_->detect(image_);
  ASSERT_FALSE(dets.empty());
  for (const auto &d : dets) {
    EXPECT_GT(d.score, 0.0f);
    EXPECT_LE(d.score, 1.0f);
    EXPECT_GE(d.class_id, 0);
    EXPECT_LT(d.class_id, 80);
    EXPECT_FALSE(d.class_name.empty());
    EXPECT_NE(d.class_name, "unknown");
    EXPECT_EQ(d.id, "0");
    EXPECT_GE(d.bbox.center.position.x - d.bbox.size.x / 2.0, -1.0);
    EXPECT_GE(d.bbox.center.position.y - d.bbox.size.y / 2.0, -1.0);
    EXPECT_LE(d.bbox.center.position.x + d.bbox.size.x / 2.0,
              static_cast<double>(image_.cols) + 1.0);
    EXPECT_LE(d.bbox.center.position.y + d.bbox.size.y / 2.0,
              static_cast<double>(image_.rows) + 1.0);
  }
}

TEST_F(Detect26nTest, HigherThresholdFiltersLowScores) {
  const auto low = detector_->detect(image_);
  ASSERT_FALSE(low.empty());

  auto strict_params = yolo_ros::test::make_params(model_path_);
  strict_params.threshold = 0.5f;
  ExposedDetect strict(strict_params);
  const auto high = strict.detect(image_);

  ASSERT_LE(high.size(), low.size());
  for (const auto &d : high) {
    EXPECT_GE(d.score, 0.5f);
  }
  const bool any_above = std::any_of(
      low.begin(), low.end(),
      [](const yolo_msgs::msg::Detection &d) { return d.score >= 0.5f; });
  if (any_above) {
    EXPECT_FALSE(high.empty());
  }
}

TEST_F(Detect26nTest, FallsBackFromInvalidCudaDevice) {
  auto params = yolo_ros::test::make_params(model_path_);
  params.provider = "cuda";
  // An out-of-range ordinal should make the CUDA session fail to initialize,
  // exercising the fallback to CPU.
  params.device = "cuda:999";
  auto detector = std::make_unique<ExposedDetect>(params);
  EXPECT_EQ(detector->active_provider(), "cpu");
}

// A raw segmentation tensor is [1, 4 + nc + 32, N] (channel-major). The mask
// coefficients are per-anchor and must stay attached to the box decoded from
// that same anchor, even after get_boxes() compacts away the anchors below the
// confidence threshold. Regression test: reading coefficients at the compacted
// position instead of the original anchor index blends the wrong prototypes and
// produces masks that do not match their boxes.
TEST(SegmentationMaskCoefficients, FollowTheirOriginalAnchor) {
  constexpr int nc = 1;
  constexpr int kProtos = 32;
  constexpr size_t n = 3;
  constexpr int rows = 4 + nc + kProtos;
  const std::vector<int64_t> shape{1, rows, static_cast<int64_t>(n)};

  // Channel-major [rows][n]: value(row, anchor) at row * n + anchor.
  std::vector<float> data(static_cast<size_t>(rows) * n, 0.0f);
  auto set = [&](int row, size_t anchor, float v) {
    data[static_cast<size_t>(row) * n + anchor] = v;
  };
  auto set_box = [&](size_t anchor, float cx, float cy, float w, float h,
                     float score) {
    set(0, anchor, cx);
    set(1, anchor, cy);
    set(2, anchor, w);
    set(3, anchor, h);
    set(4, anchor, score); // single class
  };

  // Anchor 0: background (below threshold). Its coefficients are a sentinel
  // that must never leak into a real detection.
  set_box(0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
  set(4 + nc + 0, 0, 100.0f);
  // Anchor 1: kept, prototype 1.
  set_box(1, 50.0f, 50.0f, 20.0f, 20.0f, 0.9f);
  set(4 + nc + 1, 1, 1.0f);
  // Anchor 2: kept, prototype 2.
  set_box(2, 200.0f, 200.0f, 20.0f, 20.0f, 0.8f);
  set(4 + nc + 2, 2, 1.0f);

  yolo_ros::test::Tensor tensor(shape, data);
  std::vector<Ort::Value> preds;
  preds.push_back(std::move(tensor.value()));

  // Same original/resized size => identity scaling, boxes stay as decoded.
  const auto boxes = get_segmentation_with_nms(
      preds, cv::Size(300, 300), cv::Size(300, 300), nc, 0.5f, 0.25f);
  ASSERT_EQ(boxes.size(), 2u);

  for (const auto &box : boxes) {
    ASSERT_EQ(box.mask_coeffs.size(), static_cast<size_t>(kProtos));
    if (box.score > 0.85f) {
      // Anchor 1 must carry prototype 1, not the background anchor's sentinel.
      EXPECT_FLOAT_EQ(box.mask_coeffs[1], 1.0f);
      EXPECT_FLOAT_EQ(box.mask_coeffs[0], 0.0f);
      EXPECT_FLOAT_EQ(box.mask_coeffs[2], 0.0f);
    } else {
      // Anchor 2 must carry prototype 2.
      EXPECT_FLOAT_EQ(box.mask_coeffs[2], 1.0f);
      EXPECT_FLOAT_EQ(box.mask_coeffs[1], 0.0f);
      EXPECT_FLOAT_EQ(box.mask_coeffs[0], 0.0f);
    }
  }
}

// The end-to-end/baked-head segment layout is [1, K, 6 + n_protos]: rows are
// [x1, y1, x2, y2, score, class_id, coeffs...] already NMS-sorted, with no
// per-class scores to argmax over. Decoding it as the raw layout produces
// garbage, so lock the row decoder down.
TEST(SegmentationBakedHead, DecodesRowsAndCoefficients) {
  constexpr int kProtos = 32;
  constexpr size_t kRows = 3;
  constexpr size_t kStride = 6 + kProtos;
  const std::vector<int64_t> shape{1, static_cast<int64_t>(kRows),
                                   static_cast<int64_t>(kStride)};

  std::vector<float> data(kRows * kStride, 0.0f);
  auto set_row = [&](size_t row, float x1, float y1, float x2, float y2,
                     float score, float cls, int proto, float coeff) {
    float *r = data.data() + row * kStride;
    r[0] = x1;
    r[1] = y1;
    r[2] = x2;
    r[3] = y2;
    r[4] = score;
    r[5] = cls;
    r[6 + proto] = coeff;
  };
  set_row(0, 10.0f, 20.0f, 30.0f, 40.0f, 0.9f, 2.0f, 0, 1.0f);
  set_row(1, 100.0f, 100.0f, 120.0f, 140.0f, 0.8f, 5.0f, 1, 2.0f);
  // Below threshold: must be dropped.
  set_row(2, 0.0f, 0.0f, 5.0f, 5.0f, 0.1f, 0.0f, 2, 99.0f);

  yolo_ros::test::Tensor tensor(shape, data);
  std::vector<Ort::Value> preds;
  preds.push_back(std::move(tensor.value()));

  // Identity scaling: original and resized sizes match.
  const auto boxes = get_segmentation_baked_head(preds, cv::Size(200, 200),
                                                 cv::Size(200, 200), 0.25f);
  ASSERT_EQ(boxes.size(), 2u);

  EXPECT_FLOAT_EQ(boxes[0].x1, 10.0f);
  EXPECT_FLOAT_EQ(boxes[0].y1, 20.0f);
  EXPECT_FLOAT_EQ(boxes[0].x2, 30.0f);
  EXPECT_FLOAT_EQ(boxes[0].y2, 40.0f);
  EXPECT_FLOAT_EQ(boxes[0].score, 0.9f);
  EXPECT_EQ(boxes[0].class_id, 2);
  ASSERT_EQ(boxes[0].mask_coeffs.size(), static_cast<size_t>(kProtos));
  EXPECT_FLOAT_EQ(boxes[0].mask_coeffs[0], 1.0f);

  EXPECT_FLOAT_EQ(boxes[1].score, 0.8f);
  EXPECT_EQ(boxes[1].class_id, 5);
  EXPECT_FLOAT_EQ(boxes[1].mask_coeffs[1], 2.0f);
  // The dropped row's sentinel coefficient must not leak into a kept box.
  for (const auto &box : boxes) {
    EXPECT_FLOAT_EQ(box.mask_coeffs[2], 0.0f);
  }
}

class Segment26nTest : public ::testing::Test {
protected:
  void SetUp() override {
    const std::string model =
        yolo_ros::test::download_model(kRepo, "yolo26n-seg.onnx");
    image_ = yolo_ros::test::download_sample_image();
    if (model.empty() || image_.empty()) {
      GTEST_SKIP() << "yolo26n-seg.onnx or the sample image is unavailable";
    }
    segment_ =
        std::make_unique<YoloSegment>(yolo_ros::test::make_params(model));
  }
  cv::Mat image_;
  std::unique_ptr<YoloSegment> segment_;
};

TEST_F(Segment26nTest, MasksMatchImageSize) {
  const auto dets = segment_->detect(image_);
  ASSERT_FALSE(dets.empty());
  bool any_nonempty = false;
  for (const auto &d : dets) {
    // A mis-decoded layout (e.g. an end-to-end export read as the raw one)
    // yields garbage class ids and "unknown" names; guard against that.
    EXPECT_GE(d.class_id, 0);
    EXPECT_LT(d.class_id, 80);
    EXPECT_FALSE(d.class_name.empty());
    EXPECT_NE(d.class_name, "unknown");
    EXPECT_GT(d.score, 0.0f);
    EXPECT_LE(d.score, 1.0f);
    EXPECT_EQ(d.mask.width, static_cast<int32_t>(image_.cols));
    EXPECT_EQ(d.mask.height, static_cast<int32_t>(image_.rows));
    if (!d.mask.data.empty()) {
      any_nonempty = true;
      EXPECT_GE(d.mask.data.size(), 3u);
      // The contour must not be the whole (rectangular) bounding box: that is
      // the signature of a saturated/blended-wrong prototype mask.
      std::vector<cv::Point> contour;
      contour.reserve(d.mask.data.size());
      for (const auto &p : d.mask.data) {
        contour.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
      }
      const double bbox_area = d.bbox.size.x * d.bbox.size.y;
      if (bbox_area > 0.0) {
        EXPECT_LT(cv::contourArea(contour), bbox_area);
      }
    }
    for (const auto &p : d.mask.data) {
      EXPECT_GE(p.x, 0);
      EXPECT_GE(p.y, 0);
      EXPECT_LT(p.x, image_.cols);
      EXPECT_LT(p.y, image_.rows);
    }
  }
  EXPECT_TRUE(any_nonempty);
}

class Pose26nTest : public ::testing::Test {
protected:
  void SetUp() override {
    const std::string model =
        yolo_ros::test::download_model(kRepo, "yolo26n-pose.onnx");
    image_ = yolo_ros::test::download_people_image();
    if (model.empty() || image_.empty()) {
      GTEST_SKIP() << "yolo26n-pose.onnx or the people image is unavailable";
    }
    pose_ = std::make_unique<YoloPose>(yolo_ros::test::make_params(model));
  }
  cv::Mat image_;
  std::unique_ptr<YoloPose> pose_;
};

TEST_F(Pose26nTest, KeypointsAreWellFormed) {
  const auto dets = pose_->detect(image_);
  ASSERT_FALSE(dets.empty());
  bool any_keypoints = false;
  for (const auto &d : dets) {
    for (const auto &kp : d.keypoints.data) {
      any_keypoints = true;
      EXPECT_GE(kp.id, 1);
      EXPECT_LE(kp.id, 17);
      EXPECT_GE(kp.score, 0.0f);
      EXPECT_LE(kp.score, 1.0f);
      EXPECT_GE(kp.point.x, 0.0);
      EXPECT_LE(kp.point.x, image_.cols);
      EXPECT_GE(kp.point.y, 0.0);
      EXPECT_LE(kp.point.y, image_.rows);
    }
  }
  EXPECT_TRUE(any_keypoints);
}

class ExposedClassify : public YoloClassify {
public:
  using YoloClassify::YoloClassify;
  std::size_t num_class_names() const { return class_names.size(); }
};

class Classify26nTest : public ::testing::Test {
protected:
  void SetUp() override {
    model_path_ = yolo_ros::test::download_model(kRepo, "yolo26n-cls.onnx");
    image_ = yolo_ros::test::download_sample_image();
    if (model_path_.empty() || image_.empty()) {
      GTEST_SKIP() << "yolo26n-cls.onnx or the sample image is unavailable";
    }
    classify_ = std::make_unique<ExposedClassify>(
        yolo_ros::test::make_params(model_path_));
  }
  std::string model_path_;
  cv::Mat image_;
  std::unique_ptr<ExposedClassify> classify_;
};

TEST_F(Classify26nTest, LoadsImageNetVocabulary) {
  EXPECT_EQ(classify_->num_class_names(), 1000u);
}

TEST_F(Classify26nTest, TopKIsAProbabilityDistribution) {
  const auto dets = classify_->detect(image_);
  ASSERT_EQ(dets.size(), 5u);
  float sum = 0.0f;
  for (std::size_t i = 0; i < dets.size(); ++i) {
    EXPECT_GE(dets[i].score, 0.0f);
    EXPECT_LE(dets[i].score, 1.0f);
    EXPECT_GE(dets[i].class_id, 0);
    EXPECT_LT(dets[i].class_id, 1000);
    EXPECT_FALSE(dets[i].class_name.empty());
    EXPECT_NE(dets[i].class_name, "unknown");
    EXPECT_EQ(dets[i].bbox.size.x, 0.0);
    EXPECT_EQ(dets[i].bbox.size.y, 0.0);
    if (i > 0) {
      EXPECT_GE(dets[i - 1].score, dets[i].score);
    }
    sum += dets[i].score;
  }
  EXPECT_LE(sum, 1.0f + 1e-2f);
  // The graph bakes in a softmax, so these are 5 of 1000 probabilities: their
  // sum is well below 1. A double-softmax would flatten the row toward
  // uniform (1/1000 ~ 0.001), so the top class staying above 0.01 guards the
  // documented regression without requiring a specific model confidence.
  EXPECT_GT(dets.front().score, 0.01f);
}

TEST_F(Classify26nTest, TopKParameterIsHonored) {
  auto params = yolo_ros::test::make_params(model_path_);
  params.top_k = 3;
  ExposedClassify top3(params);
  EXPECT_EQ(top3.detect(image_).size(), 3u);
}

} // namespace
} // namespace yolo_ros::yolo
