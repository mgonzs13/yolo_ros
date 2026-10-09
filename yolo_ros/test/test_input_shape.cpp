// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/engine/input_shape.hpp"

#include <gtest/gtest.h>

TEST(InputShape, StaticGraphWinsOverParams) {
  const auto shape =
      yolo_ros::engine::resolve_input_shape({1, 3, 640, 640}, 640, 480);
  EXPECT_EQ(shape.width, 640);
  EXPECT_EQ(shape.height, 640);
  EXPECT_FALSE(shape.from_params);
  EXPECT_TRUE(shape.params_ignored);
}

TEST(InputShape, StaticGraphMatchingParamsIsNotIgnored) {
  const auto shape =
      yolo_ros::engine::resolve_input_shape({1, 3, 480, 640}, 640, 480);
  EXPECT_EQ(shape.width, 640);
  EXPECT_EQ(shape.height, 480);
  EXPECT_FALSE(shape.from_params);
  EXPECT_FALSE(shape.params_ignored);
}

TEST(InputShape, DynamicGraphUsesParams) {
  const auto shape =
      yolo_ros::engine::resolve_input_shape({-1, 3, -1, -1}, 640, 480);
  EXPECT_EQ(shape.width, 640);
  EXPECT_EQ(shape.height, 480);
  EXPECT_TRUE(shape.from_params);
  EXPECT_FALSE(shape.params_ignored);
}

TEST(InputShape, MixedGraphUsesParamsOnlyForDynamicDims) {
  const auto shape =
      yolo_ros::engine::resolve_input_shape({1, 3, -1, 640}, 960, 544);
  EXPECT_EQ(shape.width, 640); // static graph width wins
  EXPECT_EQ(shape.height, 544);
  EXPECT_TRUE(shape.from_params);
  EXPECT_TRUE(shape.params_ignored); // width param was ignored
}

TEST(InputShape, MixedGraphStaticHeightDynamicWidth) {
  const auto shape =
      yolo_ros::engine::resolve_input_shape({1, 3, 640, -1}, 960, 544);
  EXPECT_EQ(shape.width, 960);  // dynamic graph width, param used
  EXPECT_EQ(shape.height, 640); // static graph height wins
  EXPECT_TRUE(shape.from_params);
  EXPECT_TRUE(shape.params_ignored); // height param was ignored
}

TEST(InputShape, ZeroDimIsTreatedAsDynamic) {
  const auto shape =
      yolo_ros::engine::resolve_input_shape({1, 3, 0, 640}, 960, 544);
  EXPECT_EQ(shape.width, 640); // static graph width wins
  EXPECT_EQ(shape.height, 544);
  EXPECT_TRUE(shape.from_params);
  EXPECT_TRUE(shape.params_ignored); // width param was ignored
}

TEST(InputShape, ShortShapeFallsBackToParams) {
  const auto shape = yolo_ros::engine::resolve_input_shape({1, 3}, 640, 480);
  EXPECT_EQ(shape.width, 640);
  EXPECT_EQ(shape.height, 480);
  EXPECT_TRUE(shape.from_params);
  EXPECT_FALSE(shape.params_ignored);
}
