// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/blackboard/channel_sync.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>

#include "sensor_msgs/msg/image.hpp"
#include "yolo_msgs/msg/detection_array.hpp"

using namespace std::chrono_literals;

namespace {

sensor_msgs::msg::Image::SharedPtr make_image(int32_t sec, uint32_t nanosec) {
  auto image = std::make_shared<sensor_msgs::msg::Image>();
  image->header.stamp.sec = sec;
  image->header.stamp.nanosec = nanosec;
  image->height = 1;
  image->width = 1;
  image->encoding = "8UC3";
  image->step = 3;
  image->data = {0, 0, 0};
  return image;
}

yolo_msgs::msg::DetectionArray::SharedPtr make_detections(int32_t sec,
                                                          uint32_t nanosec) {
  auto detections = std::make_shared<yolo_msgs::msg::DetectionArray>();
  detections->header.stamp.sec = sec;
  detections->header.stamp.nanosec = nanosec;
  return detections;
}

} // namespace

TEST(ChannelSync, MatchesEqualStamps) {
  yolo_ros::Blackboard bb;
  yolo_ros::ChannelSync<sensor_msgs::msg::Image, yolo_msgs::msg::DetectionArray>
      sync(bb, {"image", "detections"}, 10);

  bb.publish<sensor_msgs::msg::Image>("image", make_image(1, 500000000));
  yolo_ros::ChannelSync<sensor_msgs::msg::Image,
                        yolo_msgs::msg::DetectionArray>::Result result;
  EXPECT_FALSE(sync.next(result, 30ms));

  bb.publish<yolo_msgs::msg::DetectionArray>("detections",
                                             make_detections(1, 500000000));
  ASSERT_TRUE(sync.next(result, 100ms));
  EXPECT_EQ(std::get<0>(result)->header.stamp.sec, 1);
  EXPECT_EQ(std::get<0>(result)->header.stamp.nanosec, 500000000u);
  EXPECT_EQ(std::get<1>(result)->header.stamp.sec, 1);
  EXPECT_EQ(std::get<1>(result)->header.stamp.nanosec, 500000000u);
}

TEST(ChannelSync, MatchesNewestWhenOlderImageWasQueued) {
  yolo_ros::Blackboard bb;
  yolo_ros::ChannelSync<sensor_msgs::msg::Image, yolo_msgs::msg::DetectionArray>
      sync(bb, {"image", "detections"}, 10);

  bb.publish<sensor_msgs::msg::Image>("image", make_image(1, 0));
  bb.publish<sensor_msgs::msg::Image>("image", make_image(2, 0));
  bb.publish<yolo_msgs::msg::DetectionArray>("detections",
                                             make_detections(2, 0));

  yolo_ros::ChannelSync<sensor_msgs::msg::Image,
                        yolo_msgs::msg::DetectionArray>::Result result;
  ASSERT_TRUE(sync.next(result, 100ms));
  EXPECT_EQ(std::get<0>(result)->header.stamp.sec, 2);
}

TEST(ChannelSync, RejectsWrongChannelCount) {
  yolo_ros::Blackboard bb;
  EXPECT_THROW((yolo_ros::ChannelSync<sensor_msgs::msg::Image,
                                      yolo_msgs::msg::DetectionArray>(
                   bb, {"image"}, 10)),
               std::invalid_argument);
}

TEST(ChannelSync, MatchesNearbyStamps) {
  yolo_ros::Blackboard bb;
  yolo_ros::ChannelSync<sensor_msgs::msg::Image, yolo_msgs::msg::DetectionArray>
      sync(bb, {"image", "detections"}, 10);

  bb.publish<sensor_msgs::msg::Image>("image", make_image(1, 0));
  bb.publish<yolo_msgs::msg::DetectionArray>("detections",
                                             make_detections(1, 50000000));
  // ApproximateTime emits a candidate only once it can prove no better match
  // is possible; a newer image makes the (1, 0) + (1, 50 ms) pair provably
  // optimal, so it is the set returned by the next call.
  bb.publish<sensor_msgs::msg::Image>("image", make_image(3, 0));

  yolo_ros::ChannelSync<sensor_msgs::msg::Image,
                        yolo_msgs::msg::DetectionArray>::Result result;
  ASSERT_TRUE(sync.next(result, 100ms));
  EXPECT_EQ(std::get<0>(result)->header.stamp.sec, 1);
  EXPECT_EQ(std::get<0>(result)->header.stamp.nanosec, 0u);
  EXPECT_EQ(std::get<1>(result)->header.stamp.sec, 1);
  EXPECT_EQ(std::get<1>(result)->header.stamp.nanosec, 50000000u);
}

TEST(ChannelSync, ReturnsMostRecentMatch) {
  yolo_ros::Blackboard bb;
  yolo_ros::ChannelSync<sensor_msgs::msg::Image, yolo_msgs::msg::DetectionArray>
      sync(bb, {"image", "detections"}, 10);

  bb.publish<sensor_msgs::msg::Image>("image", make_image(1, 0));
  bb.publish<yolo_msgs::msg::DetectionArray>("detections",
                                             make_detections(1, 0));
  bb.publish<sensor_msgs::msg::Image>("image", make_image(2, 0));
  bb.publish<yolo_msgs::msg::DetectionArray>("detections",
                                             make_detections(2, 0));

  yolo_ros::ChannelSync<sensor_msgs::msg::Image,
                        yolo_msgs::msg::DetectionArray>::Result result;
  ASSERT_TRUE(sync.next(result, 100ms));
  EXPECT_EQ(std::get<0>(result)->header.stamp.sec, 2);
  EXPECT_EQ(std::get<1>(result)->header.stamp.sec, 2);
}

TEST(ChannelSync, StopsWhenBlackboardClosed) {
  yolo_ros::Blackboard bb;
  yolo_ros::ChannelSync<sensor_msgs::msg::Image, yolo_msgs::msg::DetectionArray>
      sync(bb, {"image", "detections"}, 10);
  bb.close_all();

  yolo_ros::ChannelSync<sensor_msgs::msg::Image,
                        yolo_msgs::msg::DetectionArray>::Result result;
  const auto start = std::chrono::steady_clock::now();
  EXPECT_FALSE(sync.next(result, 1000ms));
  const auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_LT(elapsed, 500ms);
}

TEST(ChannelSync, ZeroTimeoutReturnsImmediately) {
  yolo_ros::Blackboard bb;
  yolo_ros::ChannelSync<sensor_msgs::msg::Image, yolo_msgs::msg::DetectionArray>
      sync(bb, {"image", "detections"}, 10);

  yolo_ros::ChannelSync<sensor_msgs::msg::Image,
                        yolo_msgs::msg::DetectionArray>::Result result;
  const auto start = std::chrono::steady_clock::now();
  EXPECT_FALSE(sync.next(result, 0ms));
  const auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_LT(elapsed, 5ms);
}

TEST(ChannelSync, ZeroTimeoutReturnsQueuedMatch) {
  yolo_ros::Blackboard bb;
  yolo_ros::ChannelSync<sensor_msgs::msg::Image, yolo_msgs::msg::DetectionArray>
      sync(bb, {"image", "detections"}, 10);

  bb.publish<sensor_msgs::msg::Image>("image", make_image(1, 0));
  bb.publish<yolo_msgs::msg::DetectionArray>("detections",
                                             make_detections(1, 0));

  yolo_ros::ChannelSync<sensor_msgs::msg::Image,
                        yolo_msgs::msg::DetectionArray>::Result result;
  EXPECT_TRUE(sync.next(result, 0ms));
  EXPECT_EQ(std::get<0>(result)->header.stamp.sec, 1);
}
