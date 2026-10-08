// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/plugins/debug_plugin.hpp"

#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <thread>

#include "sensor_msgs/msg/image.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "yolo_msgs/msg/detection.hpp"
#include "yolo_msgs/msg/detection_array.hpp"
#include "yolo_msgs/msg/key_point3_d.hpp"
#include "yolo_ros/camera/camera_frame.hpp"

using namespace std::chrono_literals;

class DebugPluginTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }
};

namespace {
std::shared_ptr<yolo_ros::CameraFrame> make_black_frame() {
  auto frame = std::make_shared<yolo_ros::CameraFrame>();
  frame->header.stamp.sec = 1;
  frame->header.stamp.nanosec = 7;
  frame->header.frame_id = "camera";

  auto image = std::make_shared<sensor_msgs::msg::Image>();
  image->header = frame->header;
  image->height = 8;
  image->width = 8;
  image->encoding = "bgr8";
  image->is_bigendian = 0;
  image->step = 24;
  image->data.assign(8 * 24, 0);
  frame->rgb = image;
  return frame;
}
} // namespace

TEST_F(DebugPluginTest, DrawsCameraFrameAndMarkers) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("debug_test");
  yolo_ros::Blackboard blackboard;
  yolo_ros::TopicRegistry topics;

  yolo_ros::DebugPlugin plugin;
  plugin.declare_params(*node, "dbg.");
  plugin.get_params(*node, "dbg.");

  yolo_ros::PluginContext context{blackboard,
                                  topics,
                                  node->get_logger(),
                                  node->get_clock(),
                                  nullptr,
                                  "dbg",
                                  {{"cam0", "cam0", "cam0/detections", false}}};
  ASSERT_TRUE(plugin.setup(context));
  EXPECT_TRUE(blackboard.has_producer("cam0/debug_image"));
  EXPECT_TRUE(blackboard.has_producer("cam0/debug_bb_markers"));
  EXPECT_TRUE(blackboard.has_producer("cam0/debug_kp_markers"));

  auto debug_reader =
      blackboard.subscribe<sensor_msgs::msg::Image>("cam0/debug_image", 1);
  auto bb_marker_reader =
      blackboard.subscribe<visualization_msgs::msg::MarkerArray>(
          "cam0/debug_bb_markers", 1);
  auto kp_marker_reader =
      blackboard.subscribe<visualization_msgs::msg::MarkerArray>(
          "cam0/debug_kp_markers", 1);

  std::atomic<bool> stop{false};
  std::thread worker([&plugin, &stop] { plugin.run(stop); });
  std::this_thread::sleep_for(20ms);

  // Phase 1: one 4x4 px detection centered at (4, 4) on an all-black 8x8 BGR
  // camera frame. The box border sits at x/y in [2, 5], so drawing must change
  // at least one pixel there.
  auto frame = make_black_frame();
  auto detections = std::make_shared<yolo_msgs::msg::DetectionArray>();
  detections->header = frame->header;
  yolo_msgs::msg::Detection detection;
  detection.class_name = "person";
  detection.score = 0.9;
  detection.bbox.center.position.x = 4.0;
  detection.bbox.center.position.y = 4.0;
  detection.bbox.size.x = 4.0;
  detection.bbox.size.y = 4.0;
  detections->detections.push_back(detection);

  blackboard.publish<yolo_ros::CameraFrame>("cam0", frame);
  blackboard.publish<yolo_msgs::msg::DetectionArray>("cam0/detections",
                                                     detections);

  std::shared_ptr<const sensor_msgs::msg::Image> out;
  const bool image_received = debug_reader.wait(out, 1000ms);

  // Phase 2: a 3D detection with a box and a keypoint pair matching skeleton
  // limb {1, 2}, published on the same detections channel. The input carries
  // bbox3d/keypoints3d, so box, keypoint and limb markers must be built from
  // it without any separate markers input.
  auto detections_3d = std::make_shared<yolo_msgs::msg::DetectionArray>();
  detections_3d->header = frame->header;
  yolo_msgs::msg::Detection detection_3d;
  detection_3d.class_name = "person";
  detection_3d.score = 0.9;
  detection_3d.bbox3d.frame_id = "camera";
  detection_3d.bbox3d.center.position.x = 1.0;
  detection_3d.bbox3d.center.position.y = 2.0;
  detection_3d.bbox3d.center.position.z = 3.0;
  detection_3d.bbox3d.center.orientation.w = 1.0;
  detection_3d.bbox3d.size.x = 0.5;
  detection_3d.bbox3d.size.y = 0.6;
  detection_3d.bbox3d.size.z = 1.7;
  detection_3d.keypoints3d.frame_id = "camera";
  yolo_msgs::msg::KeyPoint3D kp1;
  kp1.id = 1;
  kp1.point.x = 1.0;
  kp1.point.y = 2.0;
  kp1.point.z = 3.0;
  kp1.score = 0.8;
  yolo_msgs::msg::KeyPoint3D kp2;
  kp2.id = 2;
  kp2.point.x = 1.1;
  kp2.point.y = 2.1;
  kp2.point.z = 3.1;
  kp2.score = 0.7;
  detection_3d.keypoints3d.data = {kp1, kp2};
  detections_3d->detections.push_back(detection_3d);

  blackboard.publish<yolo_ros::CameraFrame>("cam0", make_black_frame());
  blackboard.publish<yolo_msgs::msg::DetectionArray>("cam0/detections",
                                                     detections_3d);

  std::shared_ptr<const visualization_msgs::msg::MarkerArray> bb_markers;
  std::shared_ptr<const visualization_msgs::msg::MarkerArray> kp_markers;
  const bool bb_received = bb_marker_reader.wait(bb_markers, 1000ms);
  const bool kp_received = kp_marker_reader.wait(kp_markers, 1000ms);

  stop = true;
  blackboard.wake_all();
  worker.join();

  // Phase 1 assertions.
  ASSERT_TRUE(image_received);
  ASSERT_EQ(out->height, 8u);
  ASSERT_EQ(out->width, 8u);
  EXPECT_EQ(out->encoding, "bgr8");
  EXPECT_EQ(out->header.stamp.sec, frame->header.stamp.sec);
  EXPECT_EQ(out->header.stamp.nanosec, frame->header.stamp.nanosec);

  int changed_pixels = 0;
  for (int y = 2; y <= 5; ++y) {
    for (int x = 2; x <= 5; ++x) {
      const std::size_t offset = static_cast<std::size_t>(y) * out->step +
                                 static_cast<std::size_t>(x) * 3;
      if (out->data[offset] != 0 || out->data[offset + 1] != 0 ||
          out->data[offset + 2] != 0) {
        ++changed_pixels;
      }
    }
  }
  EXPECT_GT(changed_pixels, 0);

  // Phase 2 assertions.
  ASSERT_TRUE(bb_received);
  ASSERT_TRUE(kp_received);
  ASSERT_FALSE(bb_markers->markers.empty());
  ASSERT_FALSE(kp_markers->markers.empty());
  for (const auto &marker : bb_markers->markers) {
    EXPECT_EQ(marker.ns, "yolo_3d");
    EXPECT_TRUE(marker.lifetime.sec > 0 || marker.lifetime.nanosec > 0);
  }
  bool has_keypoint = false;
  bool has_limb = false;
  for (const auto &marker : kp_markers->markers) {
    EXPECT_TRUE(marker.lifetime.sec > 0 || marker.lifetime.nanosec > 0);
    has_keypoint = has_keypoint || marker.ns == "yolo_3d";
    has_limb = has_limb || marker.ns == "yolo_3d_limbs";
  }
  EXPECT_TRUE(has_keypoint);
  EXPECT_TRUE(has_limb);
}
