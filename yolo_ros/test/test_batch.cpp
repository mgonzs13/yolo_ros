// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "test_helpers.hpp"
#include "test_model_helpers.hpp"
#include "yolo_ros/engine/model.hpp"
#include "yolo_ros/yolo/detect.hpp"

namespace yolo_ros {
namespace {

// Splitting a batched output tensor into per-image {1, ...} views is what lets
// the unchanged task postprocessors read one image at a time.
TEST(BatchSlicing, SplitsBatchedTensorPerImage) {
  const std::vector<int64_t> shape{2, 3, 6};
  std::vector<float> data(2 * 3 * 6);
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<float>(i);
  }
  yolo_ros::test::Tensor tensor(shape, data);
  std::vector<Ort::Value> preds;
  preds.push_back(std::move(tensor.value()));

  const auto first = yolo_ros::engine::slice_batch_outputs(
      preds, 0, yolo_ros::test::cpu_memory_info());
  const auto second = yolo_ros::engine::slice_batch_outputs(
      preds, 1, yolo_ros::test::cpu_memory_info());
  ASSERT_EQ(first.size(), 1u);
  EXPECT_EQ(first[0].GetTensorTypeAndShapeInfo().GetShape(),
            std::vector<int64_t>({1, 3, 6}));
  const float *f = first[0].GetTensorData<float>();
  const float *s = second[0].GetTensorData<float>();
  for (std::size_t i = 0; i < 18; ++i) {
    EXPECT_FLOAT_EQ(f[i], data[i]);
    EXPECT_FLOAT_EQ(s[i], data[18 + i]);
  }
}

class BatchDetectTest : public ::testing::Test {
protected:
  void SetUp() override {
    model_path_ =
        yolo_ros::test::download_model("zwh20081/yolo26-onnx", "yolo26n.onnx");
    image_ = yolo_ros::test::download_sample_image();
    if (model_path_.empty() || image_.empty()) {
      GTEST_SKIP() << "yolo26n.onnx or the sample image is unavailable";
    }
    detector_ = std::make_unique<yolo_ros::yolo::YoloDetect>(
        yolo_ros::test::make_params(model_path_));
  }
  std::string model_path_;
  cv::Mat image_;
  std::unique_ptr<yolo_ros::yolo::YoloDetect> detector_;
};

// The stock (dynamic=False) models have a batch-1 input; batching must degrade
// to chunks of the fixed size and still return one result per input image.
TEST_F(BatchDetectTest, BatchedMatchesPerImageOnFixedBatchModel) {
  cv::Mat small;
  cv::resize(image_, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
  const auto a = detector_->detect(image_);
  const auto b = detector_->detect(small);
  const auto batched = detector_->detect_batch({image_, small});
  ASSERT_EQ(batched.size(), 2u);
  ASSERT_EQ(batched[0].size(), a.size());
  ASSERT_EQ(batched[1].size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_FLOAT_EQ(batched[0][i].score, a[i].score);
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    EXPECT_FLOAT_EQ(batched[1][i].score, b[i].score);
  }
}

TEST_F(BatchDetectTest, DifferentImageSizesUnscaleIndependently) {
  cv::Mat small;
  cv::resize(image_, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
  const auto batched = detector_->detect_batch({image_, small});
  ASSERT_EQ(batched.size(), 2u);
  for (const auto &d : batched[1]) {
    EXPECT_LE(d.bbox.center.position.x + d.bbox.size.x / 2.0,
              static_cast<double>(small.cols) + 1.0);
    EXPECT_LE(d.bbox.center.position.y + d.bbox.size.y / 2.0,
              static_cast<double>(small.rows) + 1.0);
  }
}

// True batch > 1 needs a dynamic-batch export; skip when it is not installed.
TEST(DynamicBatchDetect, RunsSeveralImagesInOneBatch) {
  const std::string model = yolo_ros::test::download_model(
      "zwh20081/yolo26-onnx", "yolo26n-dyn.onnx");
  const cv::Mat image = yolo_ros::test::download_sample_image();
  if (model.empty() || image.empty()) {
    GTEST_SKIP() << "yolo26n-dyn.onnx or the sample image is unavailable";
  }
  yolo_ros::yolo::YoloDetect detector(yolo_ros::test::make_params(model));
  cv::Mat small;
  cv::resize(image, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);

  const auto single = detector.detect(image);
  const auto batched = detector.detect_batch({image, image, small});
  ASSERT_EQ(batched.size(), 3u);
  ASSERT_EQ(batched[0].size(), single.size());
  for (std::size_t i = 0; i < single.size(); ++i) {
    EXPECT_EQ(batched[0][i].class_id, single[i].class_id);
    // Different batch sizes can select different cuDNN kernels, so scores are
    // close but not bit-identical across a batch-1 and a batch-3 run.
    EXPECT_NEAR(batched[0][i].score, single[i].score, 1e-2f);
  }
  // Identical inputs in the SAME run must yield identical outputs.
  ASSERT_EQ(batched[0].size(), batched[1].size());
  for (std::size_t i = 0; i < batched[0].size(); ++i) {
    EXPECT_FLOAT_EQ(batched[0][i].score, batched[1][i].score);
  }
}

// A model with a fixed batch > 1 exercises the partial-chunk padding path:
// one image pads up to the chunk, and three images run as chunks 2 + 1.
TEST(FixedBatchPadding, PadsPartialChunksAndDropsPadding) {
  const std::string model =
      yolo_ros::test::download_model("zwh20081/yolo26-onnx", "yolo26n-b2.onnx");
  const cv::Mat image = yolo_ros::test::download_sample_image();
  if (model.empty() || image.empty()) {
    GTEST_SKIP() << "yolo26n-b2.onnx or the sample image is unavailable";
  }
  yolo_ros::yolo::YoloDetect detector(yolo_ros::test::make_params(model));
  const auto single = detector.detect(image);
  ASSERT_FALSE(single.empty());
  const auto three = detector.detect_batch({image, image, image});
  ASSERT_EQ(three.size(), 3u);
  for (const auto &detections : three) {
    ASSERT_EQ(detections.size(), single.size());
  }
}

} // namespace
} // namespace yolo_ros
