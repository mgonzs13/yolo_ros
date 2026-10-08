// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/camera/camera_frame.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <vector>

#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "yolo_msgs/msg/detection_array.hpp"
#include "yolo_ros/blackboard/channel_sync.hpp"

using namespace std::chrono_literals;

namespace {

sensor_msgs::msg::Image::SharedPtr make_rgb8(int32_t sec, uint32_t nanosec) {
  auto image = std::make_shared<sensor_msgs::msg::Image>();
  image->header.stamp.sec = sec;
  image->header.stamp.nanosec = nanosec;
  image->header.frame_id = "camera";
  image->height = 2;
  image->width = 2;
  image->encoding = "rgb8";
  image->is_bigendian = 0;
  image->step = 6;
  image->data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  return image;
}

sensor_msgs::msg::Image::SharedPtr make_depth(int32_t sec, uint32_t nanosec) {
  auto image = std::make_shared<sensor_msgs::msg::Image>();
  image->header.stamp.sec = sec;
  image->header.stamp.nanosec = nanosec;
  image->height = 1;
  image->width = 2;
  image->encoding = "16UC1";
  image->is_bigendian = 0;
  image->step = 4;
  image->data = {0xE8, 0x03, 0xD0, 0x07}; // 1000, 2000
  return image;
}

sensor_msgs::msg::Image::SharedPtr make_depth32f() {
  auto image = std::make_shared<sensor_msgs::msg::Image>();
  image->height = 1;
  image->width = 2;
  image->encoding = "32FC1";
  image->is_bigendian = 0;
  image->step = 8;
  const float values[2] = {1.5f, 2.25f};
  image->data.resize(sizeof(values));
  std::memcpy(image->data.data(), values, sizeof(values));
  return image;
}

} // namespace

TEST(CameraFrame, Bgr8ConvertsAndCaches) {
  yolo_ros::CameraFrame frame;
  frame.rgb = make_rgb8(1, 0);

  const cv::Mat first = frame.bgr8();
  ASSERT_FALSE(first.empty());
  EXPECT_EQ(first.type(), CV_8UC3);
  EXPECT_EQ(first.size(), cv::Size(2, 2));
  // rgb8 -> bgr8 swaps channels.
  EXPECT_EQ(first.at<cv::Vec3b>(0, 0), cv::Vec3b(3, 2, 1));

  const cv::Mat second = frame.bgr8();
  EXPECT_EQ(first.data, second.data); // cached, not recomputed
}

TEST(CameraFrame, Depth16uIsZeroCopyInSourceEncoding) {
  yolo_ros::CameraFrame frame;
  frame.depth = make_depth(1, 0);

  const cv::Mat depth = frame.depth_image();
  ASSERT_FALSE(depth.empty());
  EXPECT_EQ(depth.type(), CV_16UC1);
  EXPECT_EQ(depth.data, frame.depth->data.data()); // view over the message
  EXPECT_EQ(depth.at<uint16_t>(0, 0), 1000);
  EXPECT_EQ(depth.at<uint16_t>(0, 1), 2000);
}

TEST(CameraFrame, Depth32Fc1PreservesValues) {
  yolo_ros::CameraFrame frame;
  frame.depth = make_depth32f();

  const cv::Mat depth = frame.depth_image();
  ASSERT_FALSE(depth.empty());
  EXPECT_EQ(depth.type(), CV_32FC1);
  EXPECT_EQ(depth.data, frame.depth->data.data()); // no conversion copy
  EXPECT_FLOAT_EQ(depth.at<float>(0, 0), 1.5f);
  EXPECT_FLOAT_EQ(depth.at<float>(0, 1), 2.25f);
}

TEST(CameraFrame, NullDepthYieldsEmptyMat) {
  yolo_ros::CameraFrame frame;
  frame.rgb = make_rgb8(1, 0);
  EXPECT_TRUE(frame.depth_image().empty());
}

TEST(CameraFrame, WorksWithChannelSync) {
  yolo_ros::Blackboard blackboard;
  yolo_ros::ChannelSync<yolo_ros::CameraFrame, yolo_msgs::msg::DetectionArray>
      sync(blackboard, {"cam0", "cam0/detections"}, 10);

  auto frame = std::make_shared<yolo_ros::CameraFrame>();
  frame->header.stamp.sec = 5;
  frame->rgb = make_rgb8(5, 0);

  auto detections = std::make_shared<yolo_msgs::msg::DetectionArray>();
  detections->header.stamp.sec = 5;

  blackboard.publish<yolo_ros::CameraFrame>("cam0", frame);
  blackboard.publish<yolo_msgs::msg::DetectionArray>("cam0/detections",
                                                     detections);

  yolo_ros::ChannelSync<yolo_ros::CameraFrame,
                        yolo_msgs::msg::DetectionArray>::Result result;
  ASSERT_TRUE(sync.next(result, 100ms));
  EXPECT_EQ(std::get<0>(result)->header.stamp.sec, 5);
  EXPECT_EQ(std::get<1>(result)->header.stamp.sec, 5);
}
