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

#include "yolo_ros/plugins/debug_plugin.hpp"

#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "yolo_msgs/msg/detection.hpp"
#include "yolo_msgs/msg/detection_array.hpp"
#include "yolo_msgs/msg/key_point3_d.hpp"
#include "yolo_ros/camera/camera_frame.hpp"

using namespace std::chrono_literals;

class DebugPluginTest : public ::testing::Test {
protected:
  static void SetUpTestCase() {
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
  EXPECT_TRUE(blackboard.has_producer("cam0/debug_image/compressed"));
  EXPECT_TRUE(blackboard.has_producer("cam0/debug_bb_markers"));
  EXPECT_TRUE(blackboard.has_producer("cam0/debug_kp_markers"));

  auto debug_reader =
      blackboard.subscribe<sensor_msgs::msg::Image>("cam0/debug_image", 1);
  auto compressed_reader =
      blackboard.subscribe<sensor_msgs::msg::CompressedImage>(
          "cam0/debug_image/compressed", 1);
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
  std::shared_ptr<const sensor_msgs::msg::CompressedImage> compressed;
  const bool compressed_received = compressed_reader.wait(compressed, 1000ms);

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

  ASSERT_TRUE(compressed_received);
  EXPECT_EQ(compressed->format, "jpeg");
  EXPECT_EQ(compressed->header.stamp.sec, frame->header.stamp.sec);
  EXPECT_EQ(compressed->header.stamp.nanosec, frame->header.stamp.nanosec);
  ASSERT_GE(compressed->data.size(), 2u);
  EXPECT_EQ(compressed->data[0], 0xFF); // JPEG SOI marker
  EXPECT_EQ(compressed->data[1], 0xD8);

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

TEST_F(DebugPluginTest, DrawsFreshestUpstreamWhileMarkersUseChainInput) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("debug_test");
  yolo_ros::Blackboard blackboard;
  yolo_ros::TopicRegistry topics;

  yolo_ros::DebugPlugin plugin;
  plugin.declare_params(*node, "dbg.");
  plugin.get_params(*node, "dbg.");

  // Chain: detection -> tracking -> detection3d -> debug. The chain input is
  // cam0/detections_3d; the freshest non-3D upstream is cam0/tracking.
  yolo_ros::PluginContext context{
      blackboard,
      topics,
      node->get_logger(),
      node->get_clock(),
      nullptr,
      "dbg",
      {{"cam0",
        "cam0",
        "cam0/detections_3d",
        true,
        {"cam0/detections", "cam0/tracking", "cam0/detections_3d"}}}};
  ASSERT_TRUE(plugin.setup(context));

  auto debug_reader =
      blackboard.subscribe<sensor_msgs::msg::Image>("cam0/debug_image", 1);
  auto bb_marker_reader =
      blackboard.subscribe<visualization_msgs::msg::MarkerArray>(
          "cam0/debug_bb_markers", 1);

  std::atomic<bool> stop{false};
  std::thread worker([&plugin, &stop] { plugin.run(stop); });
  std::this_thread::sleep_for(20ms);

  // 2D detections arrive on the tracking stream, not on the chain input:
  // debug_image must still be drawn from them.
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
  blackboard.publish<yolo_msgs::msg::DetectionArray>("cam0/tracking",
                                                     detections);

  std::shared_ptr<const sensor_msgs::msg::Image> out;
  const bool image_received = debug_reader.wait(out, 1000ms);

  // The 3D-enriched stream (the chain input) drives the markers.
  auto detections_3d = std::make_shared<yolo_msgs::msg::DetectionArray>();
  detections_3d->header = frame->header;
  yolo_msgs::msg::Detection detection_3d;
  detection_3d.class_name = "person";
  detection_3d.bbox3d.frame_id = "camera";
  detection_3d.bbox3d.center.position.z = 3.0;
  detection_3d.bbox3d.center.orientation.w = 1.0;
  detection_3d.bbox3d.size.x = 0.5;
  detection_3d.bbox3d.size.y = 0.6;
  detection_3d.bbox3d.size.z = 1.7;
  detections_3d->detections.push_back(detection_3d);

  blackboard.publish<yolo_msgs::msg::DetectionArray>("cam0/detections_3d",
                                                     detections_3d);

  std::shared_ptr<const visualization_msgs::msg::MarkerArray> bb_markers;
  const bool bb_received = bb_marker_reader.wait(bb_markers, 1000ms);

  stop = true;
  blackboard.wake_all();
  worker.join();

  ASSERT_TRUE(image_received);
  ASSERT_EQ(out->height, 8u);
  ASSERT_TRUE(bb_received);
  ASSERT_FALSE(bb_markers->markers.empty());
}

TEST_F(DebugPluginTest, CompressedTopicCanBeDisabled) {
  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>("debug_test");
  yolo_ros::Blackboard blackboard;
  yolo_ros::TopicRegistry topics;

  yolo_ros::DebugPlugin plugin;
  plugin.declare_params(*node, "dbg.");
  node->set_parameter(rclcpp::Parameter("dbg.compressed.enable", false));
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
  EXPECT_FALSE(blackboard.has_producer("cam0/debug_image/compressed"));
}

TEST_F(DebugPluginTest, CompressedImageDecodesToAnnotatedDebugImage) {
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

  auto debug_reader =
      blackboard.subscribe<sensor_msgs::msg::Image>("cam0/debug_image", 1);
  auto compressed_reader =
      blackboard.subscribe<sensor_msgs::msg::CompressedImage>(
          "cam0/debug_image/compressed", 1);

  std::atomic<bool> stop{false};
  std::thread worker([&plugin, &stop] { plugin.run(stop); });
  std::this_thread::sleep_for(20ms);

  // 64x64 mid-gray frame: large enough that JPEG noise stays low, and flat
  // enough that an unchanged (unannotated) picture is easy to detect.
  auto frame = std::make_shared<yolo_ros::CameraFrame>();
  frame->header.stamp.sec = 2;
  frame->header.stamp.nanosec = 11;
  frame->header.frame_id = "camera";
  auto image = std::make_shared<sensor_msgs::msg::Image>();
  image->header = frame->header;
  image->height = 64;
  image->width = 64;
  image->encoding = "bgr8";
  image->is_bigendian = 0;
  image->step = 64 * 3;
  image->data.assign(64 * 64 * 3, 96);
  frame->rgb = image;

  auto detections = std::make_shared<yolo_msgs::msg::DetectionArray>();
  detections->header = frame->header;
  yolo_msgs::msg::Detection detection;
  detection.class_name = "person";
  detection.score = 0.9;
  detection.bbox.center.position.x = 32.0;
  detection.bbox.center.position.y = 32.0;
  detection.bbox.size.x = 20.0;
  detection.bbox.size.y = 20.0;
  detections->detections.push_back(detection);

  blackboard.publish<yolo_ros::CameraFrame>("cam0", frame);
  blackboard.publish<yolo_msgs::msg::DetectionArray>("cam0/detections",
                                                     detections);

  std::shared_ptr<const sensor_msgs::msg::Image> out;
  std::shared_ptr<const sensor_msgs::msg::CompressedImage> compressed;
  const bool image_received = debug_reader.wait(out, 1000ms);
  const bool compressed_received = compressed_reader.wait(compressed, 1000ms);

  stop = true;
  blackboard.wake_all();
  worker.join();

  ASSERT_TRUE(image_received);
  ASSERT_TRUE(compressed_received);

  // The payload must decode to the annotated image the plugin published, not
  // to the untouched camera frame: compare against the raw debug image with a
  // small tolerance for JPEG loss and require the drawn box to be present.
  // Lyrical's sensor_msgs uses rosidl::Buffer for the payload, which is not
  // directly convertible to cv::InputArray. Copy the bytes into a plain vector
  // so the test works across ROS 2 distros.
  const std::vector<std::uint8_t> encoded(compressed->data.data(),
                                          compressed->data.data() +
                                              compressed->data.size());
  const cv::Mat decoded = cv::imdecode(encoded, cv::IMREAD_COLOR);
  ASSERT_FALSE(decoded.empty());
  ASSERT_EQ(decoded.cols, static_cast<int>(out->width));
  ASSERT_EQ(decoded.rows, static_cast<int>(out->height));
  ASSERT_EQ(decoded.channels(), 3);

  const cv::Mat published(static_cast<int>(out->height),
                          static_cast<int>(out->width), CV_8UC3,
                          const_cast<std::uint8_t *>(out->data.data()),
                          static_cast<std::size_t>(out->step));
  cv::Mat annotated_diff;
  cv::absdiff(decoded, published, annotated_diff);
  const cv::Scalar mean_diff = cv::mean(annotated_diff);
  EXPECT_LT((mean_diff[0] + mean_diff[1] + mean_diff[2]) / 3.0, 5.0);

  const cv::Mat camera_frame(64, 64, CV_8UC3, cv::Scalar(96, 96, 96));
  cv::Mat frame_diff;
  cv::absdiff(decoded, camera_frame, frame_diff);
  EXPECT_GT(cv::norm(frame_diff, cv::NORM_INF), 80.0);
}
