// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/camera/camera_streams.hpp"

#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "yolo_ros/blackboard/blackboard.hpp"

using namespace std::chrono_literals;

class CameraStreamsTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }

  static rclcpp::NodeOptions
  camera_options(const std::vector<rclcpp::Parameter> &extra) {
    std::vector<rclcpp::Parameter> parameters{
        rclcpp::Parameter("cam0.rgb_topic", "/camera0/image"),
        rclcpp::Parameter("cam0.depth_topic", "/camera0/depth"),
        rclcpp::Parameter("cam0.depth_info_topic", "/camera0/info"),
        rclcpp::Parameter("cam1.rgb_topic", "/camera1/image"),
    };
    parameters.insert(parameters.end(), extra.begin(), extra.end());
    rclcpp::NodeOptions options;
    options.parameter_overrides(parameters);
    return options;
  }
};

namespace {

/// @brief Publish one rgb8 image on @p topic and spin until @p channel yields
/// a frame, or the attempt budget runs out.
/// @return The delivered frame, or nullptr when nothing arrived.
std::shared_ptr<const yolo_ros::CameraFrame>
spin_until_frame(rclcpp_lifecycle::LifecycleNode &node,
                 yolo_ros::Blackboard &blackboard, const std::string &channel,
                 const std::string &topic, int32_t sec) {
  auto reader = blackboard.subscribe<yolo_ros::CameraFrame>(channel, 1);

  auto rgb = std::make_shared<sensor_msgs::msg::Image>();
  rgb->header.stamp.sec = sec;
  rgb->header.frame_id = "camera";
  rgb->height = 1;
  rgb->width = 1;
  rgb->encoding = "rgb8";
  rgb->step = 3;
  rgb->data = {1, 2, 3};

  auto publisher =
      node.create_publisher<sensor_msgs::msg::Image>(topic, rclcpp::QoS(1));
  publisher->on_activate();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node.get_node_base_interface());

  std::shared_ptr<const yolo_ros::CameraFrame> frame;
  for (int i = 0; i < 100 && !frame; ++i) {
    publisher->publish(*rgb);
    executor.spin_some();
    std::this_thread::sleep_for(10ms);
    reader.try_pop(frame);
  }

  executor.remove_node(node.get_node_base_interface());
  return frame;
}

} // namespace

TEST_F(CameraStreamsTest, ConfigureValidatesAndResolves) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "streams_test", camera_options({}));
  yolo_ros::Blackboard blackboard;
  yolo_ros::CameraStreams streams(blackboard, node->get_logger());

  std::string error;
  ASSERT_TRUE(streams.configure(*node, {"cam0", "cam1"}, error)) << error;
  ASSERT_EQ(streams.cameras().size(), 2u);
  EXPECT_EQ(streams.cameras()[0].name, "cam0");
  EXPECT_TRUE(streams.cameras()[0].has_depth());
  EXPECT_FALSE(streams.cameras()[1].has_depth());
  EXPECT_EQ(streams.cameras()[1].rgb_topic, "/camera1/image");
}

TEST_F(CameraStreamsTest, RejectsDepthWithoutInfo) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "streams_test", camera_options({rclcpp::Parameter("cam1.depth_topic",
                                                        "/camera1/depth")}));
  yolo_ros::Blackboard blackboard;
  yolo_ros::CameraStreams streams(blackboard, node->get_logger());

  std::string error;
  EXPECT_FALSE(streams.configure(*node, {"cam1"}, error));
  EXPECT_NE(error.find("cam1"), std::string::npos);
}

TEST_F(CameraStreamsTest, RejectsDuplicateTopics) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "streams_test",
      camera_options({rclcpp::Parameter("cam1.rgb_topic", "/camera0/image")}));
  yolo_ros::Blackboard blackboard;
  yolo_ros::CameraStreams streams(blackboard, node->get_logger());

  std::string error;
  EXPECT_FALSE(streams.configure(*node, {"cam0", "cam1"}, error));
  EXPECT_NE(error.find("cam1"), std::string::npos);
}

TEST_F(CameraStreamsTest, RejectsMissingRgbTopic) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "streams_test", camera_options({}));
  yolo_ros::Blackboard blackboard;
  yolo_ros::CameraStreams streams(blackboard, node->get_logger());

  std::string error;
  EXPECT_FALSE(streams.configure(*node, {"cam2"}, error));
  EXPECT_NE(error.find("cam2"), std::string::npos);
  EXPECT_NE(error.find("rgb_topic"), std::string::npos);
}

TEST_F(CameraStreamsTest, FailedConfigureLeavesNoPartialState) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "streams_test",
      camera_options({rclcpp::Parameter("cam1.rgb_topic", "")}));
  yolo_ros::Blackboard blackboard;
  yolo_ros::CameraStreams streams(blackboard, node->get_logger());

  std::string error;
  EXPECT_FALSE(streams.configure(*node, {"cam0", "cam1"}, error));
  EXPECT_TRUE(streams.cameras().empty());
  EXPECT_FALSE(blackboard.has_producer("cam0"));
  EXPECT_FALSE(blackboard.has_producer("cam1"));
}

TEST_F(CameraStreamsTest, RejectsBadNames) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "streams_test", camera_options({}));
  yolo_ros::Blackboard blackboard;
  yolo_ros::CameraStreams streams(blackboard, node->get_logger());

  std::string error;
  EXPECT_FALSE(streams.configure(*node, {"bad.name"}, error));
  EXPECT_NE(error.find("invalid camera name"), std::string::npos);
  EXPECT_NE(error.find("bad.name"), std::string::npos);

  EXPECT_FALSE(streams.configure(*node, {"cam0", "cam0"}, error));
  EXPECT_NE(error.find("duplicate camera name"), std::string::npos);
  EXPECT_NE(error.find("cam0"), std::string::npos);

  EXPECT_FALSE(streams.configure(*node, {""}, error));
  EXPECT_NE(error.find("invalid camera name"), std::string::npos);

  EXPECT_FALSE(streams.configure(*node, {"cam:0"}, error));
  EXPECT_NE(error.find("invalid camera name"), std::string::npos);
  EXPECT_NE(error.find("cam:0"), std::string::npos);

  EXPECT_FALSE(streams.configure(*node, {"cam/0"}, error));
  EXPECT_NE(error.find("invalid camera name"), std::string::npos);
  EXPECT_NE(error.find("cam/0"), std::string::npos);
}

TEST_F(CameraStreamsTest, DeliversSynchronizedFrame) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "streams_test", camera_options({}));
  yolo_ros::Blackboard blackboard;
  yolo_ros::CameraStreams streams(blackboard, node->get_logger());

  std::string error;
  ASSERT_TRUE(streams.configure(*node, {"cam0"}, error)) << error;
  ASSERT_TRUE(streams.activate(*node, error)) << error;

  auto reader = blackboard.subscribe<yolo_ros::CameraFrame>("cam0", 1);

  auto rgb = std::make_shared<sensor_msgs::msg::Image>();
  rgb->header.stamp.sec = 7;
  rgb->header.frame_id = "camera";
  rgb->height = 1;
  rgb->width = 1;
  rgb->encoding = "rgb8";
  rgb->step = 3;
  rgb->data = {1, 2, 3};

  auto depth = std::make_shared<sensor_msgs::msg::Image>();
  depth->header = rgb->header;
  depth->height = 1;
  depth->width = 1;
  depth->encoding = "16UC1";
  depth->step = 2;
  depth->data = {0xE8, 0x03};

  auto info = std::make_shared<sensor_msgs::msg::CameraInfo>();
  info->header = rgb->header;

  auto rgb_pub = node->create_publisher<sensor_msgs::msg::Image>(
      "/camera0/image", rclcpp::QoS(1));
  rgb_pub->on_activate();
  auto depth_pub = node->create_publisher<sensor_msgs::msg::Image>(
      "/camera0/depth", rclcpp::QoS(1));
  depth_pub->on_activate();
  auto info_pub = node->create_publisher<sensor_msgs::msg::CameraInfo>(
      "/camera0/info", rclcpp::QoS(1));
  info_pub->on_activate();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node->get_node_base_interface());

  std::shared_ptr<const yolo_ros::CameraFrame> frame;
  for (int i = 0; i < 100 && !frame; ++i) {
    rgb_pub->publish(*rgb);
    depth_pub->publish(*depth);
    info_pub->publish(*info);
    executor.spin_some();
    std::this_thread::sleep_for(10ms);
    reader.try_pop(frame);
  }

  ASSERT_TRUE(frame);
  EXPECT_EQ(frame->header.stamp.sec, 7);
  ASSERT_TRUE(frame->depth);
  ASSERT_TRUE(frame->depth_info);

  streams.deactivate();
  executor.remove_node(node->get_node_base_interface());
}

TEST_F(CameraStreamsTest, DeliversRgbOnlyFrame) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "streams_test", camera_options({}));
  yolo_ros::Blackboard blackboard;
  yolo_ros::CameraStreams streams(blackboard, node->get_logger());

  std::string error;
  ASSERT_TRUE(streams.configure(*node, {"cam1"}, error)) << error;
  ASSERT_TRUE(streams.activate(*node, error)) << error;

  const auto frame =
      spin_until_frame(*node, blackboard, "cam1", "/camera1/image", 9);
  ASSERT_TRUE(frame);
  EXPECT_EQ(frame->header.stamp.sec, 9);
  ASSERT_TRUE(frame->rgb);
  EXPECT_FALSE(frame->depth);
  EXPECT_FALSE(frame->depth_info);

  streams.deactivate();
}

TEST_F(CameraStreamsTest, DeactivateThenActivateAgain) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "streams_test", camera_options({}));
  yolo_ros::Blackboard blackboard;
  yolo_ros::CameraStreams streams(blackboard, node->get_logger());

  std::string error;
  ASSERT_TRUE(streams.configure(*node, {"cam1"}, error)) << error;
  ASSERT_TRUE(streams.activate(*node, error)) << error;
  streams.deactivate();

  ASSERT_TRUE(streams.activate(*node, error)) << error;
  const auto frame =
      spin_until_frame(*node, blackboard, "cam1", "/camera1/image", 11);
  ASSERT_TRUE(frame);
  EXPECT_EQ(frame->header.stamp.sec, 11);

  streams.deactivate();
}

TEST_F(CameraStreamsTest, ActivateInvalidTopicRollsBack) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "streams_test",
      camera_options({rclcpp::Parameter("cam0.depth_topic", ""),
                      rclcpp::Parameter("cam0.depth_info_topic", ""),
                      rclcpp::Parameter("cam1.rgb_topic", "bad topic!")}));
  yolo_ros::Blackboard blackboard;
  yolo_ros::CameraStreams streams(blackboard, node->get_logger());

  std::string error;
  ASSERT_TRUE(streams.configure(*node, {"cam0", "cam1"}, error)) << error;

  EXPECT_FALSE(streams.activate(*node, error));
  EXPECT_NE(error.find("cam1"), std::string::npos);

  // cam0 was subscribed before cam1 failed: the rollback must have removed it,
  // so publishing on its topic delivers no frame.
  const auto frame =
      spin_until_frame(*node, blackboard, "cam0", "/camera0/image", 3);
  EXPECT_FALSE(frame);
}
