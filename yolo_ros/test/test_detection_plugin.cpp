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
  static void SetUpTestCase() {
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
  node->set_parameter(rclcpp::Parameter("det.batch.max_size", -1));
  plugin.get_params(*node, "det.");

  EXPECT_EQ(plugin.max_batch_size(), 1u);
}

TEST_F(DetectionPluginTest, ReadsPadToBatchParam) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("detection_test");

  yolo_ros::DetectionPlugin plugin;
  plugin.declare_params(*node, "det.");
  node->set_parameter(rclcpp::Parameter("det.batch.pad_to", 4));
  plugin.get_params(*node, "det.");

  EXPECT_EQ(plugin.pad_to_batch(), 4);
}

TEST_F(DetectionPluginTest, PadToBatchIsClampedToMaxBatchSize) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("detection_test");

  yolo_ros::DetectionPlugin plugin;
  plugin.declare_params(*node, "det.");
  node->set_parameter(rclcpp::Parameter("det.batch.max_size", 2));
  node->set_parameter(rclcpp::Parameter("det.batch.pad_to", 4));
  plugin.get_params(*node, "det.");

  EXPECT_EQ(plugin.pad_to_batch(), 2);
}

TEST_F(DetectionPluginTest, NegativePadToBatchDisablesPadding) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("detection_test");

  yolo_ros::DetectionPlugin plugin;
  plugin.declare_params(*node, "det.");
  node->set_parameter(rclcpp::Parameter("det.batch.pad_to", -3));
  plugin.get_params(*node, "det.");

  EXPECT_EQ(plugin.pad_to_batch(), 0);
}

TEST_F(DetectionPluginTest, NegativeMaxDetIsClamped) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("detection_test");

  yolo_ros::DetectionPlugin plugin;
  plugin.declare_params(*node, "det.");
  node->set_parameter(rclcpp::Parameter("det.inference.max_det", -1));
  plugin.get_params(*node, "det.");

  EXPECT_EQ(plugin.max_det(), 0);
}

TEST_F(DetectionPluginTest, ReadsInputSizeParams) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("detection_test");
  yolo_ros::DetectionPlugin plugin;
  plugin.declare_params(*node, "det.");
  node->set_parameter(rclcpp::Parameter("det.input.width", 960));
  node->set_parameter(rclcpp::Parameter("det.input.height", 544));
  plugin.get_params(*node, "det.");

  EXPECT_EQ(plugin.img_width(), 960);
  EXPECT_EQ(plugin.img_height(), 544);
}

TEST_F(DetectionPluginTest, RejectsNonPositiveInputSize) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("detection_test");
  yolo_ros::DetectionPlugin plugin;
  plugin.declare_params(*node, "det.");
  node->set_parameter(rclcpp::Parameter("det.input.width", 0));
  EXPECT_THROW(plugin.get_params(*node, "det."), std::runtime_error);
}

TEST_F(DetectionPluginTest, RejectsNegativeInputSize) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("detection_test");
  yolo_ros::DetectionPlugin plugin;
  plugin.declare_params(*node, "det.");
  node->set_parameter(rclcpp::Parameter("det.input.height", -1));
  EXPECT_THROW(plugin.get_params(*node, "det."), std::runtime_error);
}

TEST_F(DetectionPluginTest, AcceptsNonMultipleOf32) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("detection_test");
  yolo_ros::DetectionPlugin plugin;
  plugin.declare_params(*node, "det.");
  node->set_parameter(rclcpp::Parameter("det.input.width", 650));
  node->set_parameter(rclcpp::Parameter("det.input.height", 480));
  plugin.get_params(*node, "det.");

  EXPECT_EQ(plugin.img_width(), 650);
}

TEST_F(DetectionPluginTest, DeclaresQnnParams) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("detection_test");
  yolo_ros::DetectionPlugin plugin;
  plugin.declare_params(*node, "det.");
  EXPECT_TRUE(node->has_parameter("det.qnn.htp_performance_mode"));
  EXPECT_TRUE(node->has_parameter("det.qnn.htp_arch"));
  EXPECT_TRUE(node->has_parameter("det.qnn.soc_model"));
  EXPECT_TRUE(node->has_parameter("det.qnn.backend_path"));
  EXPECT_TRUE(node->has_parameter("det.qnn.offload_graph_io_quantization"));
  EXPECT_TRUE(node->has_parameter("det.qnn.enable_htp_fp16_precision"));
  EXPECT_TRUE(node->has_parameter("det.qnn.context_cache_enable"));
  EXPECT_TRUE(node->has_parameter("det.qnn.context_cache_path"));
}

TEST_F(DetectionPluginTest, QnnClampsMaxBatchSizeToOne) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("detection_test");
  yolo_ros::DetectionPlugin plugin;
  plugin.declare_params(*node, "det.");
  node->set_parameter(rclcpp::Parameter("det.provider", "qnn"));
  node->set_parameter(rclcpp::Parameter("det.batch.max_size", 8));
  plugin.get_params(*node, "det.");
  EXPECT_EQ(plugin.max_batch_size(), 1u);
}

TEST_F(DetectionPluginTest, NonQnnKeepsMaxBatchSize) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("detection_test");
  yolo_ros::DetectionPlugin plugin;
  plugin.declare_params(*node, "det.");
  node->set_parameter(rclcpp::Parameter("det.provider", "cpu"));
  node->set_parameter(rclcpp::Parameter("det.batch.max_size", 8));
  plugin.get_params(*node, "det.");
  EXPECT_EQ(plugin.max_batch_size(), 8u);
}
