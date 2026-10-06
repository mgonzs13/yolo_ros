// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "yolo_ros/yolo/model_factory.hpp"

namespace yolo_ros {
namespace {

TEST(ModelTaskSelection, ExplicitTypeWins) {
  using yolo_ros::yolo::ModelTask;
  EXPECT_EQ(yolo_ros::yolo::select_model_task("Pose", "/x.onnx"),
            ModelTask::Pose);
  EXPECT_EQ(yolo_ros::yolo::select_model_task("Segment", "/x.onnx"),
            ModelTask::Segment);
  EXPECT_EQ(yolo_ros::yolo::select_model_task("OBB", "/x.onnx"),
            ModelTask::Obb);
  EXPECT_EQ(yolo_ros::yolo::select_model_task("Classify", "/x.onnx"),
            ModelTask::Classify);
  EXPECT_EQ(yolo_ros::yolo::select_model_task("YOLO", "/x.onnx"),
            ModelTask::Detect);
}

TEST(ModelTaskSelection, AutoUsesFilename) {
  using yolo_ros::yolo::ModelTask;
  EXPECT_EQ(yolo_ros::yolo::select_model_task("auto", "yolo11n-pose.onnx"),
            ModelTask::Pose);
  EXPECT_EQ(yolo_ros::yolo::select_model_task("auto", "yolo11n-seg.onnx"),
            ModelTask::Segment);
  EXPECT_EQ(yolo_ros::yolo::select_model_task("auto", "yolo26m-obb.onnx"),
            ModelTask::Obb);
  EXPECT_EQ(yolo_ros::yolo::select_model_task("auto", "yolo26s-cls.onnx"),
            ModelTask::Classify);
  EXPECT_EQ(yolo_ros::yolo::select_model_task("auto", "yolo26s.onnx"),
            ModelTask::Detect);
}

TEST(ModelTaskSelection, ExplicitOverridesMisleadingFilename) {
  EXPECT_EQ(yolo_ros::yolo::select_model_task("Segment", "yolo26m-obb.onnx"),
            yolo_ros::yolo::ModelTask::Segment);
}

} // namespace
} // namespace yolo_ros
