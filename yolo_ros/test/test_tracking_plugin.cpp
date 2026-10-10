// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/plugins/tracking_plugin.hpp"

#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "sensor_msgs/msg/image.hpp"
#include "yolo_msgs/msg/detection.hpp"
#include "yolo_msgs/msg/detection_array.hpp"
#include "yolo_ros/camera/camera_frame.hpp"

using namespace std::chrono_literals;

class TrackingPluginTest : public ::testing::Test {
protected:
  static void SetUpTestCase() {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }
};

namespace {

std::shared_ptr<yolo_ros::CameraFrame> make_frame(int32_t sec) {
  auto frame = std::make_shared<yolo_ros::CameraFrame>();
  frame->header.stamp.sec = sec;
  auto image = std::make_shared<sensor_msgs::msg::Image>();
  image->header = frame->header;
  image->height = 32;
  image->width = 32;
  image->encoding = "bgr8";
  image->step = 96;
  image->data.assign(32 * 96, 0);
  frame->rgb = image;
  return frame;
}

std::shared_ptr<yolo_msgs::msg::DetectionArray>
make_detections(int32_t sec, const std::string &class_name, float score,
                double center_x) {
  auto detections = std::make_shared<yolo_msgs::msg::DetectionArray>();
  detections->header.stamp.sec = sec;
  yolo_msgs::msg::Detection detection;
  detection.class_id = 0;
  detection.class_name = class_name;
  detection.score = score;
  detection.bbox.center.position.x = center_x;
  detection.bbox.center.position.y = 16.0;
  detection.bbox.size.x = 10.0;
  detection.bbox.size.y = 20.0;
  detections->detections.push_back(detection);
  return detections;
}

} // namespace

TEST_F(TrackingPluginTest, TracksDetectionsFromBlackboard) {
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("tracking_test");
  yolo_ros::Blackboard blackboard;
  yolo_ros::TopicRegistry topics;

  yolo_ros::TrackingPlugin plugin;
  plugin.declare_params(*node, "track.");
  plugin.get_params(*node, "track.");

  yolo_ros::PluginContext context{blackboard,
                                  topics,
                                  node->get_logger(),
                                  node->get_clock(),
                                  nullptr,
                                  "track",
                                  {{"cam0", "cam0", "cam0/detections", false}}};
  ASSERT_TRUE(plugin.setup(context));
  ASSERT_TRUE(plugin.activate());

  auto output_reader =
      blackboard.subscribe<yolo_msgs::msg::DetectionArray>("cam0/tracking", 4);

  auto frame = std::make_shared<yolo_ros::CameraFrame>();
  frame->header.stamp.sec = 3;
  auto image = std::make_shared<sensor_msgs::msg::Image>();
  image->header = frame->header;
  image->height = 32;
  image->width = 32;
  image->encoding = "bgr8";
  image->step = 96;
  image->data.assign(32 * 96, 0);
  frame->rgb = image;

  auto detections = std::make_shared<yolo_msgs::msg::DetectionArray>();
  detections->header.stamp.sec = 3;
  yolo_msgs::msg::Detection detection;
  detection.class_id = 0;
  detection.class_name = "person";
  detection.score = 0.9;
  detection.bbox.center.position.x = 16.0;
  detection.bbox.center.position.y = 16.0;
  detection.bbox.size.x = 10.0;
  detection.bbox.size.y = 20.0;
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
  EXPECT_FALSE(out->detections[0].id.empty());
}

TEST_F(TrackingPluginTest, UnknownTrackerPassesDetectionsThrough) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "tracking_unknown_test");
  yolo_ros::Blackboard blackboard;
  yolo_ros::TopicRegistry topics;

  yolo_ros::TrackingPlugin plugin;
  plugin.declare_params(*node, "track.");
  node->set_parameter(rclcpp::Parameter("track.tracker_type", "unknown"));
  plugin.get_params(*node, "track.");

  yolo_ros::PluginContext context{blackboard,
                                  topics,
                                  node->get_logger(),
                                  node->get_clock(),
                                  nullptr,
                                  "track",
                                  {{"cam0", "cam0", "cam0/detections", false}}};
  ASSERT_TRUE(plugin.setup(context));
  ASSERT_TRUE(plugin.activate());

  auto output_reader =
      blackboard.subscribe<yolo_msgs::msg::DetectionArray>("cam0/tracking", 4);

  auto frame = std::make_shared<yolo_ros::CameraFrame>();
  frame->header.stamp.sec = 5;
  auto image = std::make_shared<sensor_msgs::msg::Image>();
  image->header = frame->header;
  image->height = 32;
  image->width = 32;
  image->encoding = "bgr8";
  image->step = 96;
  image->data.assign(32 * 96, 0);
  frame->rgb = image;

  auto detections = std::make_shared<yolo_msgs::msg::DetectionArray>();
  detections->header.stamp.sec = 5;
  yolo_msgs::msg::Detection detection;
  detection.class_id = 0;
  detection.class_name = "person";
  detection.score = 0.9;
  detection.bbox.center.position.x = 16.0;
  detection.bbox.center.position.y = 16.0;
  detection.bbox.size.x = 10.0;
  detection.bbox.size.y = 20.0;
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
  EXPECT_TRUE(out->detections[0].id.empty());
}

TEST_F(TrackingPluginTest, TracksTwoCamerasIndependently) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "tracking_two_cameras_test");
  yolo_ros::Blackboard blackboard;
  yolo_ros::TopicRegistry topics;

  yolo_ros::TrackingPlugin plugin;
  plugin.declare_params(*node, "track.");
  plugin.get_params(*node, "track.");

  yolo_ros::PluginContext context{blackboard,
                                  topics,
                                  node->get_logger(),
                                  node->get_clock(),
                                  nullptr,
                                  "track",
                                  {{"cam0", "cam0", "cam0/detections", false},
                                   {"cam1", "cam1", "cam1/detections", false}}};
  ASSERT_TRUE(plugin.setup(context));
  ASSERT_TRUE(plugin.activate());

  auto reader0 =
      blackboard.subscribe<yolo_msgs::msg::DetectionArray>("cam0/tracking", 4);
  auto reader1 =
      blackboard.subscribe<yolo_msgs::msg::DetectionArray>("cam1/tracking", 4);

  std::atomic<bool> stop{false};
  std::thread worker([&plugin, &stop] { plugin.run(stop); });
  std::this_thread::sleep_for(20ms);

  // Nearly identical boxes on both cameras: a single shared tracker would
  // associate cam1's detection with cam0's fresh track and reuse its id, so
  // distinct ids prove each camera owns independent tracker state.
  blackboard.publish<yolo_ros::CameraFrame>("cam0", make_frame(10));
  blackboard.publish<yolo_msgs::msg::DetectionArray>(
      "cam0/detections", make_detections(10, "person", 0.9, 16.0));
  blackboard.publish<yolo_ros::CameraFrame>("cam1", make_frame(20));
  blackboard.publish<yolo_msgs::msg::DetectionArray>(
      "cam1/detections", make_detections(20, "bicycle", 0.8, 17.0));

  std::shared_ptr<const yolo_msgs::msg::DetectionArray> out0;
  std::shared_ptr<const yolo_msgs::msg::DetectionArray> out1;
  const bool received0 = reader0.wait(out0, 1000ms);
  const bool received1 = reader1.wait(out1, 1000ms);

  // Second sweep: each camera's own track must keep its id.
  blackboard.publish<yolo_ros::CameraFrame>("cam0", make_frame(11));
  blackboard.publish<yolo_msgs::msg::DetectionArray>(
      "cam0/detections", make_detections(11, "person", 0.9, 16.0));
  blackboard.publish<yolo_ros::CameraFrame>("cam1", make_frame(21));
  blackboard.publish<yolo_msgs::msg::DetectionArray>(
      "cam1/detections", make_detections(21, "bicycle", 0.8, 17.0));

  std::shared_ptr<const yolo_msgs::msg::DetectionArray> out0_again;
  std::shared_ptr<const yolo_msgs::msg::DetectionArray> out1_again;
  const bool received0_again = reader0.wait(out0_again, 1000ms);
  const bool received1_again = reader1.wait(out1_again, 1000ms);

  stop = true;
  blackboard.wake_all();
  worker.join();
  plugin.deactivate();

  ASSERT_TRUE(received0);
  ASSERT_TRUE(received1);
  ASSERT_EQ(out0->detections.size(), 1u);
  ASSERT_EQ(out1->detections.size(), 1u);
  EXPECT_EQ(out0->detections[0].class_name, "person");
  EXPECT_EQ(out1->detections[0].class_name, "bicycle");
  EXPECT_FALSE(out0->detections[0].id.empty());
  EXPECT_FALSE(out1->detections[0].id.empty());
  EXPECT_NE(out0->detections[0].id, out1->detections[0].id);

  ASSERT_TRUE(received0_again);
  ASSERT_TRUE(received1_again);
  ASSERT_EQ(out0_again->detections.size(), 1u);
  ASSERT_EQ(out1_again->detections.size(), 1u);
  EXPECT_EQ(out0_again->detections[0].id, out0->detections[0].id);
  EXPECT_EQ(out1_again->detections[0].id, out1->detections[0].id);
}

TEST_F(TrackingPluginTest, ReconfigureSwitchesTrackerType) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
      "tracking_reconfigure_test");

  yolo_ros::TrackingPlugin plugin;
  plugin.declare_params(*node, "track.");
  node->set_parameter(rclcpp::Parameter("track.tracker_type", "botsort"));
  plugin.declare_params(*node, "track."); // must not throw
  plugin.get_params(*node, "track.");     // must read botsort knobs
  EXPECT_TRUE(node->has_parameter("track.gmc_method"));
  EXPECT_TRUE(node->has_parameter("track.with_reid"));
}
