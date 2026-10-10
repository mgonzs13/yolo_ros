// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/plugins/detect_3d_plugin.hpp"

#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "yolo_msgs/msg/detection.hpp"
#include "yolo_msgs/msg/detection_array.hpp"
#include "yolo_ros/camera/camera_frame.hpp"

using namespace std::chrono_literals;

class Detect3DPluginTest : public ::testing::Test {
protected:
  static void SetUpTestCase() {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }
};

namespace {

std::shared_ptr<sensor_msgs::msg::Image>
make_depth16u(int32_t sec, const std::string &frame, uint16_t value) {
  auto depth = std::make_shared<sensor_msgs::msg::Image>();
  depth->header.stamp.sec = sec;
  depth->header.frame_id = frame;
  depth->height = 4;
  depth->width = 4;
  depth->encoding = "16UC1";
  depth->step = 8;
  depth->data.assign(4 * 8, 0);

  for (std::size_t i = 0; i < 4 * 4; ++i) {
    depth->data[i * 2] = static_cast<uint8_t>(value & 0xFF);
    depth->data[i * 2 + 1] = static_cast<uint8_t>(value >> 8);
  }

  return depth;
}

std::shared_ptr<sensor_msgs::msg::Image>
make_depth32f(int32_t sec, const std::string &frame, float metres) {
  auto depth = std::make_shared<sensor_msgs::msg::Image>();
  depth->header.stamp.sec = sec;
  depth->header.frame_id = frame;
  depth->height = 4;
  depth->width = 4;
  depth->encoding = "32FC1";
  depth->step = 16;
  depth->data.assign(4 * 16, 0);

  for (std::size_t i = 0; i < 4 * 4; ++i) {
    std::memcpy(depth->data.data() + i * sizeof(float), &metres, sizeof(float));
  }

  return depth;
}

std::shared_ptr<sensor_msgs::msg::CameraInfo>
make_info(int32_t sec, const std::string &frame) {
  auto info = std::make_shared<sensor_msgs::msg::CameraInfo>();
  info->header.stamp.sec = sec;
  info->header.frame_id = frame;
  info->height = 4;
  info->width = 4;
  info->k[0] = 50.0;
  info->k[4] = 50.0;
  info->k[2] = 2.0;
  info->k[5] = 2.0;
  return info;
}

std::shared_ptr<yolo_msgs::msg::DetectionArray>
make_detections(int32_t sec, const std::string &frame) {
  auto detections = std::make_shared<yolo_msgs::msg::DetectionArray>();
  detections->header.stamp.sec = sec;
  detections->header.frame_id = frame;
  yolo_msgs::msg::Detection detection;
  detection.score = 0.9;
  detection.bbox.center.position.x = 2.0;
  detection.bbox.center.position.y = 2.0;
  detection.bbox.size.x = 3.0;
  detection.bbox.size.y = 3.0;
  detections->detections.push_back(detection);
  return detections;
}

} // namespace

TEST_F(Detect3DPluginTest, LiftsDetectionUsingDepth) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("detect3d_test");
  auto clock = node->get_clock();
  yolo_ros::Blackboard blackboard;
  yolo_ros::TopicRegistry topics;
  tf2_ros::Buffer tf_buffer(clock);

  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = clock->now();
  transform.header.frame_id = "base_link";
  transform.child_frame_id = "camera";
  transform.transform.rotation.w = 1.0;
  transform.transform.translation.x = 0.5;
  tf_buffer.setTransform(transform, "test", true);

  yolo_ros::Detect3DPlugin plugin;
  plugin.declare_params(*node, "d3d.");
  plugin.get_params(*node, "d3d.");

  yolo_ros::PluginContext context{blackboard,
                                  topics,
                                  node->get_logger(),
                                  clock,
                                  &tf_buffer,
                                  "d3d",
                                  {{"cam0", "cam0", "cam0/detections", true}}};
  ASSERT_TRUE(plugin.setup(context));
  ASSERT_TRUE(plugin.activate());

  auto output_reader = blackboard.subscribe<yolo_msgs::msg::DetectionArray>(
      "cam0/detections_3d", 4);

  auto depth = std::make_shared<sensor_msgs::msg::Image>();
  depth->header.stamp.sec = 4;
  depth->header.frame_id = "camera";
  depth->height = 4;
  depth->width = 4;
  depth->encoding = "16UC1";
  depth->step = 8;
  depth->data.assign(4 * 8, 0);
  for (std::size_t i = 0; i < 4 * 4; ++i) {
    const uint16_t value = 1000; // 1 m at divisor 1000
    depth->data[i * 2] = static_cast<uint8_t>(value & 0xFF);
    depth->data[i * 2 + 1] = static_cast<uint8_t>(value >> 8);
  }

  auto info = std::make_shared<sensor_msgs::msg::CameraInfo>();
  info->header.stamp.sec = 4;
  info->header.frame_id = "camera";
  info->height = 4;
  info->width = 4;
  info->k[0] = 50.0;
  info->k[4] = 50.0;
  info->k[2] = 2.0;
  info->k[5] = 2.0;

  auto frame = std::make_shared<yolo_ros::CameraFrame>();
  frame->header = depth->header;
  frame->depth = depth;
  frame->depth_info = info;

  auto detections = std::make_shared<yolo_msgs::msg::DetectionArray>();
  detections->header.stamp.sec = 4;
  detections->header.frame_id = "camera";
  yolo_msgs::msg::Detection detection;
  detection.score = 0.9;
  detection.bbox.center.position.x = 2.0;
  detection.bbox.center.position.y = 2.0;
  detection.bbox.size.x = 3.0;
  detection.bbox.size.y = 3.0;
  detections->detections.push_back(detection);

  std::atomic<bool> stop{false};
  std::thread worker([&plugin, &stop] { plugin.run(stop); });
  std::this_thread::sleep_for(20ms);
  blackboard.publish<yolo_ros::CameraFrame>("cam0", frame);
  blackboard.publish<yolo_msgs::msg::DetectionArray>("cam0/detections",
                                                     detections);

  std::shared_ptr<const yolo_msgs::msg::DetectionArray> out;
  const bool received = output_reader.wait(out, 1000ms);
  stop = true;
  blackboard.wake_all();
  worker.join();
  plugin.deactivate();

  ASSERT_TRUE(received);
  ASSERT_EQ(out->detections.size(), 1u);
  ASSERT_EQ(out->detections[0].bbox3d.frame_id, "base_link");
  EXPECT_LT(std::abs(out->detections[0].bbox3d.center.position.z - 1.0), 0.05);
  EXPECT_LT(std::abs(out->detections[0].bbox3d.center.position.x - 0.5), 0.05);
}

TEST_F(Detect3DPluginTest, DepthlessCameraFailsSetup) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "detect3d_depthless_test");
  yolo_ros::Blackboard blackboard;
  yolo_ros::TopicRegistry topics;
  tf2_ros::Buffer tf_buffer(node->get_clock());

  yolo_ros::Detect3DPlugin plugin;
  plugin.declare_params(*node, "d3d.");
  plugin.get_params(*node, "d3d.");

  yolo_ros::PluginContext context{blackboard,
                                  topics,
                                  node->get_logger(),
                                  node->get_clock(),
                                  &tf_buffer,
                                  "d3d",
                                  {{"cam0", "cam0", "cam0/detections", false}}};
  EXPECT_FALSE(plugin.setup(context));
}

TEST_F(Detect3DPluginTest, LiftsTwoCamerasIndependently) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "detect3d_two_cameras_test");
  auto clock = node->get_clock();
  yolo_ros::Blackboard blackboard;
  yolo_ros::TopicRegistry topics;
  tf2_ros::Buffer tf_buffer(clock);

  geometry_msgs::msg::TransformStamped transform0;
  transform0.header.stamp = clock->now();
  transform0.header.frame_id = "base_link";
  transform0.child_frame_id = "camera0";
  transform0.transform.rotation.w = 1.0;
  transform0.transform.translation.x = 0.5;
  tf_buffer.setTransform(transform0, "test", true);

  geometry_msgs::msg::TransformStamped transform1;
  transform1.header.stamp = clock->now();
  transform1.header.frame_id = "base_link";
  transform1.child_frame_id = "camera1";
  transform1.transform.rotation.w = 1.0;
  transform1.transform.translation.x = 2.0;
  tf_buffer.setTransform(transform1, "test", true);

  yolo_ros::Detect3DPlugin plugin;
  plugin.declare_params(*node, "d3d.");
  plugin.get_params(*node, "d3d.");

  yolo_ros::PluginContext context{blackboard,
                                  topics,
                                  node->get_logger(),
                                  clock,
                                  &tf_buffer,
                                  "d3d",
                                  {{"cam0", "cam0", "cam0/detections", true},
                                   {"cam1", "cam1", "cam1/detections", true}}};
  ASSERT_TRUE(plugin.setup(context));
  ASSERT_TRUE(plugin.activate());

  auto reader0 = blackboard.subscribe<yolo_msgs::msg::DetectionArray>(
      "cam0/detections_3d", 4);
  auto reader1 = blackboard.subscribe<yolo_msgs::msg::DetectionArray>(
      "cam1/detections_3d", 4);

  auto frame0 = std::make_shared<yolo_ros::CameraFrame>();
  frame0->header.stamp.sec = 30;
  frame0->header.frame_id = "camera0";
  frame0->depth = make_depth16u(30, "camera0", 1000); // 1 m
  frame0->depth_info = make_info(30, "camera0");

  auto frame1 = std::make_shared<yolo_ros::CameraFrame>();
  frame1->header.stamp.sec = 40;
  frame1->header.frame_id = "camera1";
  frame1->depth = make_depth16u(40, "camera1", 2000); // 2 m
  frame1->depth_info = make_info(40, "camera1");

  std::atomic<bool> stop{false};
  std::thread worker([&plugin, &stop] { plugin.run(stop); });
  std::this_thread::sleep_for(20ms);
  blackboard.publish<yolo_ros::CameraFrame>("cam0", frame0);
  blackboard.publish<yolo_msgs::msg::DetectionArray>(
      "cam0/detections", make_detections(30, "camera0"));
  blackboard.publish<yolo_ros::CameraFrame>("cam1", frame1);
  blackboard.publish<yolo_msgs::msg::DetectionArray>(
      "cam1/detections", make_detections(40, "camera1"));

  std::shared_ptr<const yolo_msgs::msg::DetectionArray> out0;
  std::shared_ptr<const yolo_msgs::msg::DetectionArray> out1;
  const bool received0 = reader0.wait(out0, 1000ms);
  const bool received1 = reader1.wait(out1, 1000ms);

  stop = true;
  blackboard.wake_all();
  worker.join();
  plugin.deactivate();

  ASSERT_TRUE(received0);
  ASSERT_TRUE(received1);
  ASSERT_EQ(out0->detections.size(), 1u);
  ASSERT_EQ(out1->detections.size(), 1u);
  ASSERT_EQ(out0->detections[0].bbox3d.frame_id, "base_link");
  ASSERT_EQ(out1->detections[0].bbox3d.frame_id, "base_link");
  EXPECT_LT(std::abs(out0->detections[0].bbox3d.center.position.z - 1.0), 0.05);
  EXPECT_LT(std::abs(out0->detections[0].bbox3d.center.position.x - 0.5), 0.05);
  EXPECT_LT(std::abs(out1->detections[0].bbox3d.center.position.z - 2.0), 0.05);
  EXPECT_LT(std::abs(out1->detections[0].bbox3d.center.position.x - 2.0), 0.05);
}

TEST_F(Detect3DPluginTest, Depth32Fc1LiftsInMetres) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "detect3d_depth32f_test");
  auto clock = node->get_clock();
  yolo_ros::Blackboard blackboard;
  yolo_ros::TopicRegistry topics;
  tf2_ros::Buffer tf_buffer(clock);

  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = clock->now();
  transform.header.frame_id = "base_link";
  transform.child_frame_id = "camera";
  transform.transform.rotation.w = 1.0;
  transform.transform.translation.x = 0.5;
  tf_buffer.setTransform(transform, "test", true);

  yolo_ros::Detect3DPlugin plugin;
  plugin.declare_params(*node, "d3d.");
  plugin.get_params(*node, "d3d.");

  yolo_ros::PluginContext context{blackboard,
                                  topics,
                                  node->get_logger(),
                                  clock,
                                  &tf_buffer,
                                  "d3d",
                                  {{"cam0", "cam0", "cam0/detections", true}}};
  ASSERT_TRUE(plugin.setup(context));
  ASSERT_TRUE(plugin.activate());

  auto output_reader = blackboard.subscribe<yolo_msgs::msg::DetectionArray>(
      "cam0/detections_3d", 4);

  auto frame = std::make_shared<yolo_ros::CameraFrame>();
  frame->header.stamp.sec = 50;
  frame->header.frame_id = "camera";
  // 32FC1 depths are already in metres: depth_utils ignores
  // depth_image_units_divisor (default 1000) for float images, so 1.5 must
  // stay 1.5 instead of collapsing to 1.5 / 1000.
  frame->depth = make_depth32f(50, "camera", 1.5f);
  frame->depth_info = make_info(50, "camera");

  std::atomic<bool> stop{false};
  std::thread worker([&plugin, &stop] { plugin.run(stop); });
  std::this_thread::sleep_for(20ms);
  blackboard.publish<yolo_ros::CameraFrame>("cam0", frame);
  blackboard.publish<yolo_msgs::msg::DetectionArray>(
      "cam0/detections", make_detections(50, "camera"));

  std::shared_ptr<const yolo_msgs::msg::DetectionArray> out;
  const bool received = output_reader.wait(out, 1000ms);
  stop = true;
  blackboard.wake_all();
  worker.join();
  plugin.deactivate();

  ASSERT_TRUE(received);
  ASSERT_EQ(out->detections.size(), 1u);
  ASSERT_EQ(out->detections[0].bbox3d.frame_id, "base_link");
  EXPECT_LT(std::abs(out->detections[0].bbox3d.center.position.z - 1.5), 0.05);
  EXPECT_LT(std::abs(out->detections[0].bbox3d.center.position.x - 0.5), 0.05);
}
