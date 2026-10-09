// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/plugins/detection_plugin.hpp"

#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <memory>
#include <string>
#include <vector>

#include "yolo_msgs/msg/detection_array.hpp"

class DetectionPluginTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }
};

TEST_F(DetectionPluginTest, SingleCameraOutputsDetectionsChannel) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("detection_test");
  yolo_ros::Blackboard blackboard;
  yolo_ros::TopicRegistry topics;

  yolo_ros::DetectionPlugin plugin;
  plugin.declare_params(*node, "det.");
  plugin.get_params(*node, "det.");

  yolo_ros::PluginContext context{blackboard,
                                  topics,
                                  node->get_logger(),
                                  node->get_clock(),
                                  nullptr,
                                  "det",
                                  {{"cam0", "cam0", "cam0", false}}};
  ASSERT_TRUE(plugin.setup(context));
  EXPECT_TRUE(blackboard.has_producer("cam0/detections"));
}

TEST_F(DetectionPluginTest, MultiCameraUsesCameraPrefixes) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("detection_test");
  yolo_ros::Blackboard blackboard;
  yolo_ros::TopicRegistry topics;

  yolo_ros::DetectionPlugin plugin;
  plugin.declare_params(*node, "det.");
  plugin.get_params(*node, "det.");

  yolo_ros::PluginContext context{
      blackboard,
      topics,
      node->get_logger(),
      node->get_clock(),
      nullptr,
      "det",
      {{"cam0", "cam0", "cam0", false}, {"cam1", "cam1", "cam1", false}}};
  ASSERT_TRUE(plugin.setup(context));
  EXPECT_TRUE(blackboard.has_producer("cam0/detections"));
  EXPECT_TRUE(blackboard.has_producer("cam1/detections"));
}

TEST_F(DetectionPluginTest, NegativeMaxBatchSizeIsClamped) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("detection_test");

  yolo_ros::DetectionPlugin plugin;
  plugin.declare_params(*node, "det.");
  node->set_parameter(rclcpp::Parameter("det.max_batch_size", -1));
  plugin.get_params(*node, "det.");

  EXPECT_EQ(plugin.max_batch_size(), 1u);
}

TEST_F(DetectionPluginTest, NegativeMaxDetIsClamped) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("detection_test");

  yolo_ros::DetectionPlugin plugin;
  plugin.declare_params(*node, "det.");
  node->set_parameter(rclcpp::Parameter("det.max_det", -1));
  plugin.get_params(*node, "det.");

  EXPECT_EQ(plugin.max_det(), 0);
}

TEST_F(DetectionPluginTest, ReadsInputSizeParams) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("detection_test");
  yolo_ros::DetectionPlugin plugin;
  plugin.declare_params(*node, "det.");
  node->set_parameter(rclcpp::Parameter("det.img_width", 960));
  node->set_parameter(rclcpp::Parameter("det.img_height", 544));
  plugin.get_params(*node, "det.");

  EXPECT_EQ(plugin.img_width(), 960);
  EXPECT_EQ(plugin.img_height(), 544);
}

TEST_F(DetectionPluginTest, RejectsNonPositiveInputSize) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("detection_test");
  yolo_ros::DetectionPlugin plugin;
  plugin.declare_params(*node, "det.");
  node->set_parameter(rclcpp::Parameter("det.img_width", 0));
  EXPECT_THROW(plugin.get_params(*node, "det."), std::runtime_error);
}

TEST_F(DetectionPluginTest, RejectsNegativeInputSize) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("detection_test");
  yolo_ros::DetectionPlugin plugin;
  plugin.declare_params(*node, "det.");
  node->set_parameter(rclcpp::Parameter("det.img_height", -1));
  EXPECT_THROW(plugin.get_params(*node, "det."), std::runtime_error);
}

TEST_F(DetectionPluginTest, AcceptsNonMultipleOf32) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("detection_test");
  yolo_ros::DetectionPlugin plugin;
  plugin.declare_params(*node, "det.");
  node->set_parameter(rclcpp::Parameter("det.img_width", 650));
  node->set_parameter(rclcpp::Parameter("det.img_height", 480));
  plugin.get_params(*node, "det.");

  EXPECT_EQ(plugin.img_width(), 650);
}
