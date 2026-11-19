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

#include "yolo_ros/plugin/topic_registry.hpp"

#include <gtest/gtest.h>

#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>

#include "sensor_msgs/msg/image.hpp"
#include "yolo_msgs/msg/detection_array.hpp"

using namespace std::chrono_literals;

class TopicRegistryTest : public ::testing::Test {
protected:
  static void SetUpTestCase() {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }
};

TEST_F(TopicRegistryTest, PublishesExposedChannels) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("registry_test");
  yolo_ros::TopicRegistry registry;
  yolo_ros::Blackboard blackboard;

  blackboard.declare_channel<yolo_msgs::msg::DetectionArray>("detections");
  registry.expose<yolo_msgs::msg::DetectionArray>("detections", "detections",
                                                  rclcpp::QoS(10));
  registry.validate();
  registry.create_entities(*node, blackboard);
  EXPECT_EQ(registry.publisher_entity_count(), 1u);
  EXPECT_TRUE(registry.has_publisher("detections"));
  EXPECT_FALSE(registry.has_publisher("missing"));
  EXPECT_FALSE(registry.has_subscribers("detections"));
  blackboard.publish<yolo_msgs::msg::DetectionArray>(
      "detections", std::make_shared<yolo_msgs::msg::DetectionArray>());
  registry.destroy_entities();
}

TEST_F(TopicRegistryTest, SharesPublisherAcrossChannels) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("registry_test");
  yolo_ros::TopicRegistry registry;
  yolo_ros::Blackboard blackboard;

  registry.expose<sensor_msgs::msg::Image>("chan", "topic", rclcpp::QoS(1));
  registry.expose<sensor_msgs::msg::Image>("other", "topic", rclcpp::QoS(1));
  registry.validate();
  registry.create_entities(*node, blackboard);
  EXPECT_EQ(registry.publisher_entity_count(), 1u);
  registry.destroy_entities();
}

TEST_F(TopicRegistryTest, RejectsExposeSameTopicDifferentType) {
  yolo_ros::TopicRegistry registry;
  registry.expose<sensor_msgs::msg::Image>("chan", "topic", rclcpp::QoS(1));
  registry.expose<yolo_msgs::msg::DetectionArray>("chan", "topic",
                                                  rclcpp::QoS(1));
  EXPECT_THROW(registry.validate(), std::runtime_error);
}

TEST_F(TopicRegistryTest, RejectsExposeSameTopicDifferentQos) {
  yolo_ros::TopicRegistry registry;
  registry.expose<sensor_msgs::msg::Image>("chan", "topic", rclcpp::QoS(1));
  registry.expose<sensor_msgs::msg::Image>("other", "topic",
                                           rclcpp::QoS(1).best_effort());
  EXPECT_THROW(registry.validate(), std::runtime_error);
}

TEST_F(TopicRegistryTest, DeliversExposedChannelToRosSubscriber) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("registry_test");
  yolo_ros::TopicRegistry registry;
  yolo_ros::Blackboard blackboard;
  std::shared_ptr<const sensor_msgs::msg::Image> received;

  auto subscription = node->create_subscription<sensor_msgs::msg::Image>(
      "topic_delivers_exposed_channel", rclcpp::QoS(1),
      [&received](std::shared_ptr<const sensor_msgs::msg::Image> msg) {
        received = std::move(msg);
      });
  registry.expose<sensor_msgs::msg::Image>(
      "chan", "topic_delivers_exposed_channel", rclcpp::QoS(1));
  registry.validate();
  registry.create_entities(*node, blackboard);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node->get_node_base_interface());

  const auto deadline = std::chrono::steady_clock::now() + 1s;
  while (std::chrono::steady_clock::now() < deadline && !received) {
    blackboard.publish<sensor_msgs::msg::Image>(
        "chan", std::make_shared<sensor_msgs::msg::Image>());
    executor.spin_some();
    std::this_thread::sleep_for(5ms);
  }
  EXPECT_TRUE(received != nullptr);
  EXPECT_TRUE(registry.has_subscribers("chan"));

  executor.remove_node(node->get_node_base_interface());
  registry.destroy_entities();
  (void)subscription;
}

TEST_F(TopicRegistryTest, ResetClearsRequests) {
  yolo_ros::TopicRegistry registry;
  registry.expose<sensor_msgs::msg::Image>("reset_a", "reset_topic",
                                           rclcpp::QoS(1));
  registry.reset();
  EXPECT_EQ(registry.publisher_entity_count(), 0u);
}
