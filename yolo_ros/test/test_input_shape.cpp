// Copyright (C) 2026 Miguel Ángel González Santamarta
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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

TEST(InputShape, ChannelLastStaticGraphWinsOverParams) {
  const auto shape =
      yolo_ros::engine::resolve_input_shape({1, 640, 640, 3}, 640, 480);
  EXPECT_EQ(shape.width, 640);
  EXPECT_EQ(shape.height, 640);
  EXPECT_TRUE(shape.channel_last);
  EXPECT_FALSE(shape.from_params);
  EXPECT_TRUE(shape.params_ignored);
}

TEST(InputShape, ChannelLastDynamicGraphUsesParams) {
  const auto shape =
      yolo_ros::engine::resolve_input_shape({1, -1, -1, 3}, 640, 480);
  EXPECT_EQ(shape.width, 640);
  EXPECT_EQ(shape.height, 480);
  EXPECT_TRUE(shape.channel_last);
  EXPECT_TRUE(shape.from_params);
  EXPECT_FALSE(shape.params_ignored);
}

TEST(InputShape, ChannelFirstIsNotChannelLast) {
  const auto shape =
      yolo_ros::engine::resolve_input_shape({1, 3, 640, 640}, 640, 480);
  EXPECT_FALSE(shape.channel_last);
}
