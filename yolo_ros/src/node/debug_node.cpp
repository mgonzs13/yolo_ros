// Copyright (c) 2025 Alejandro González Cantón
// Portions Copyright (c) 2023-2025 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#if defined(CV_BRIDGE_H)
#include <cv_bridge/cv_bridge.h>
#else
#include <cv_bridge/cv_bridge.hpp>
#endif

#include "yolo_ros/node/debug_node.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace yolo_ros::node {

namespace {
/// @brief Inset (px) between a label box and the top-left corner it is anchored
/// to (a detection's box corner, or the frame for image-level labels).
constexpr int kLabelInset = 2;

/// @brief Map a hue (OpenCV 8-bit H, 0..179) to a fully saturated, fully bright
/// BGR color.
///
/// The old palette derived colors straight from the hash bytes, which produced
/// muddy, low-contrast colors for many classes. Pinning saturation/value and
/// only varying the hue guarantees every color is vivid while remaining
/// deterministic for a given class name.
cv::Scalar vivid_bgr(int hue) {
  const int h = ((hue % 180) + 180) % 180;
  cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(h, 235, 255));
  cv::Mat bgr;
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  const cv::Vec3b pixel = bgr.at<cv::Vec3b>(0, 0);
  return cv::Scalar(pixel[0], pixel[1], pixel[2]);
}

/// @brief Number of limbs in @ref kSkeleton.
constexpr int kNumSkeletonLimbs = 19;

/// @brief COCO human pose skeleton, using the 1-based keypoint ids published by
/// yolo_node. Shared by the 2D debug image and the 3D RViz keypoint markers so
/// both draw the same skeleton.
constexpr int kSkeleton[kNumSkeletonLimbs][2] = {
    {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13},
    {6, 7},   {6, 8},   {7, 9},   {8, 10},  {9, 11},  {2, 3},  {1, 2},
    {1, 3},   {2, 4},   {3, 5},   {4, 6},   {5, 7}};

/// @brief Generate a stable, vivid BGR color directly from a keypoint or limb
/// index. The 47 stride is coprime with 180, so adjacent indices land on
/// well-separated hues (a borrowed palette is deliberately avoided).
cv::Scalar indexed_color(int index) { return vivid_bgr(index * 47 + 11); }

/// @brief Convert a lifetime in (fractional) seconds to a ROS duration.
///
/// Negative inputs are clamped to zero so a misconfigured parameter degrades to
/// the persistent-marker behaviour instead of wrapping the unsigned
/// nanoseconds.
builtin_interfaces::msg::Duration duration_from_seconds(double seconds) {
  builtin_interfaces::msg::Duration duration;
  const double clamped = std::max(0.0, seconds);
  duration.sec = static_cast<int32_t>(clamped);
  duration.nanosec = static_cast<uint32_t>(
      (clamped - static_cast<double>(duration.sec)) * 1e9);
  return duration;
}
} // namespace

DebugNode::DebugNode()
    : rclcpp_lifecycle::LifecycleNode("yolo_node"), image_qos_profile(1),
      class_to_color() {
  this->declare_parameter("image_reliability", 2);
  this->declare_parameter("image_topic", "image");
  this->declare_parameter("detections_topic", "detections");
  this->declare_parameter("markers_topic", "detections_3d");
  this->declare_parameter<double>("marker_lifetime", 0.5);
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
DebugNode::on_configure(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "[%s] Configuring...", this->get_name());

  this->image_topic_ = this->get_parameter("image_topic").as_string();
  this->detections_topic_ = this->get_parameter("detections_topic").as_string();
  this->markers_topic_ = this->get_parameter("markers_topic").as_string();
  this->marker_lifetime_ = this->get_parameter("marker_lifetime").as_double();

  int image_reliability = this->get_parameter("image_reliability").as_int();
  rclcpp::ReliabilityPolicy qos_reliability_policy;
  if (image_reliability == 0) {
    qos_reliability_policy = rclcpp::ReliabilityPolicy::SystemDefault;
  } else if (image_reliability == 1) {
    qos_reliability_policy = rclcpp::ReliabilityPolicy::Reliable;
  } else {
    qos_reliability_policy = rclcpp::ReliabilityPolicy::BestEffort;
  }
  this->image_qos_profile = rclcpp::QoS(1)
                                .reliability(qos_reliability_policy)
                                .durability_volatile()
                                .keep_last(1);

  this->debug_publisher =
      this->create_publisher<sensor_msgs::msg::Image>("debug_image", 10);

  this->bb_markers_publisher =
      this->create_publisher<visualization_msgs::msg::MarkerArray>(
          "debug_bb_markers", 10);

  this->kp_markers_publisher =
      this->create_publisher<visualization_msgs::msg::MarkerArray>(
          "debug_kp_markers", 10);

  RCLCPP_INFO(get_logger(), "[%s] Configured", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
DebugNode::on_activate(const rclcpp_lifecycle::State &) {
  this->image_subscription.subscribe(this->shared_from_this(),
                                     this->image_topic_, image_qos_profile);
  this->detection_subscription.subscribe(
      this->shared_from_this(), this->detections_topic_, image_qos_profile);

  uint32_t queue_size = 10;

  this->synchronizer =
      std::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy>>(
          queue_size);
  this->synchronizer->connectInput(this->image_subscription,
                                   this->detection_subscription);
  this->synchronizer->registerCallback(std::bind(&DebugNode::recieve_callback,
                                                 this, std::placeholders::_1,
                                                 std::placeholders::_2));

  // Separate subscription to the 3D-enriched stream purely for the RViz
  // markers. It is NOT part of the image sync, so a slow 3D/depth stream only
  // throttles the markers, never the debug image.
  this->markers_subscription_ =
      this->create_subscription<yolo_msgs::msg::DetectionArray>(
          this->markers_topic_, rclcpp::QoS(1),
          std::bind(&DebugNode::markers_callback, this, std::placeholders::_1));

  RCLCPP_INFO(get_logger(), "[%s] Activated", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
DebugNode::on_deactivate(const rclcpp_lifecycle::State &) {
  this->markers_subscription_.reset();
  RCLCPP_INFO(get_logger(), "[%s] Deactivated", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
DebugNode::on_cleanup(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "[%s] Cleaned up", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
DebugNode::on_shutdown(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "[%s] Shutting down", this->get_name());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

void DebugNode::recieve_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr &msg_image,
    const yolo_msgs::msg::DetectionArray::ConstSharedPtr &msg_detections) {
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg_image, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception &e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  cv::Mat image = cv_ptr->image;

  // Draw ALL detections onto one image, then publish ONCE (not per-detection).
  // Masks are drawn onto a single overlay layer and blended once, avoiding a
  // full image.clone() per detection.
  cv::Mat overlay = image.clone();
  // Image-level (classification) detections carry an empty bbox: draw them as
  // one vertical list, sorted by descending score (top-1 first) so the best
  // class sits at the top of the stack. Spatial detections (non-empty bbox)
  // keep their per-box anchors and are drawn below.
  std::vector<const yolo_msgs::msg::Detection *> image_level;
  for (const auto &detection : msg_detections->detections) {
    if (detection.bbox.size.x <= 0.0 && detection.bbox.size.y <= 0.0) {
      image_level.push_back(&detection);
    }
  }
  std::sort(
      image_level.begin(), image_level.end(),
      [](const yolo_msgs::msg::Detection *a,
         const yolo_msgs::msg::Detection *b) { return a->score > b->score; });
  int image_label_y = 0;
  for (const auto *detection : image_level) {
    const auto color = color_for_class(detection->class_name);
    image_label_y +=
        draw_label(image, label_text(*detection),
                   cv::Point(kLabelInset, kLabelInset + image_label_y), color);
  }

  for (const auto &detection : msg_detections->detections) {
    if (detection.bbox.size.x <= 0.0 && detection.bbox.size.y <= 0.0) {
      continue; // image-level label already drawn above
    }
    auto color = color_for_class(detection.class_name);
    image = draw_box(image, detection, color);
    draw_mask(overlay, image, detection, color);
    image = draw_keypoints(image, detection);
  }
  cv::addWeighted(overlay, 0.4, image, 0.6, 0, image);

  // The debug image is only gated by the image<->2D-detections sync, so it
  // publishes at the full 2D detection rate regardless of the 3D stream.
  // The Mat is always BGR8 (toCvCopy above forced BGR8 and OpenCV draws in
  // BGR), so advertise BGR8 — reusing the *original* encoding here mislabels
  // e.g. an rgb8 camera stream as rgb8 while the pixels are BGR, which makes
  // RViz swap red/blue and the image look blue-tainted.
  auto return_image =
      cv_bridge::CvImage(msg_image->header, sensor_msgs::image_encodings::BGR8,
                         image)
          .toImageMsg();
  this->debug_publisher->publish(*return_image.get());
}

// Drives the RViz 3D box / keypoint markers from the 3D-enriched stream. This
// runs on its OWN subscription (markers_topic_), so its publish rate is the 3D
// stream's rate (min(debug image, 3D detections)) and does not throttle the
// debug image.
void DebugNode::markers_callback(
    const yolo_msgs::msg::DetectionArray::ConstSharedPtr &msg_detections) {
  visualization_msgs::msg::MarkerArray bb_marker_array;
  visualization_msgs::msg::MarkerArray kp_marker_array;

  for (const auto &detection : msg_detections->detections) {
    auto color = color_for_class(detection.class_name);

    // RViz markers for the 3D boxes (emitted only when a detect_3d node has
    // enriched the stream with bbox3d).
    if (!detection.bbox3d.frame_id.empty()) {
      auto marker = create_bb_marker(detection, color);
      marker.header.stamp = msg_detections->header.stamp;
      marker.id = bb_marker_array.markers.size();
      bb_marker_array.markers.push_back(marker);
    }

    // RViz markers for the 3D keypoints (pose output from a detect_3d node).
    if (!detection.keypoints3d.frame_id.empty()) {
      std::map<int, const yolo_msgs::msg::KeyPoint3D *> points;
      for (const auto &keypoint : detection.keypoints3d.data) {
        points[keypoint.id] = &keypoint;

        auto marker = create_kp_marker(keypoint);
        marker.header.frame_id = detection.keypoints3d.frame_id;
        marker.header.stamp = msg_detections->header.stamp;
        marker.id = kp_marker_array.markers.size();
        kp_marker_array.markers.push_back(marker);
      }

      // Connect the keypoints with the same per-limb COCO skeleton (and
      // colors) the 2D debug image draws, whenever both endpoints are present
      // in the 3D stream.
      for (int i = 0; i < kNumSkeletonLimbs; ++i) {
        auto it1 = points.find(kSkeleton[i][0]);
        auto it2 = points.find(kSkeleton[i][1]);
        if (it1 == points.end() || it2 == points.end()) {
          continue;
        }
        auto marker = create_limb_marker(*it1->second, *it2->second,
                                         indexed_color(i + 32));
        marker.header.frame_id = detection.keypoints3d.frame_id;
        marker.header.stamp = msg_detections->header.stamp;
        marker.id = kp_marker_array.markers.size();
        kp_marker_array.markers.push_back(marker);
      }
    }
  }

  this->bb_markers_publisher->publish(bb_marker_array);
  this->kp_markers_publisher->publish(kp_marker_array);
}

cv::Scalar DebugNode::color_for_class(const std::string &class_name) {
  auto color_it = class_to_color.find(class_name);
  if (color_it == class_to_color.end()) {
    // Deterministic FNV-1a hash of the class name so the same class always
    // gets the same color across runs (rand() made colors change every run).
    uint32_t hash = 2166136261u;
    for (const char c : class_name) {
      hash ^= static_cast<uint8_t>(c);
      hash *= 16777619u;
    }
    class_to_color[class_name] = vivid_bgr(static_cast<int>(hash % 180));
    color_it = class_to_color.find(class_name);
  }
  return color_it->second;
}

std::string
DebugNode::label_text(const yolo_msgs::msg::Detection &detection) const {
  std::string text = detection.class_name;
  if (detection.id != "") {
    text += " " + detection.id;
  }
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(3) << detection.score;
  text += " " + ss.str();
  return text;
}

cv::Mat DebugNode::draw_box(const cv::Mat &image,
                            const yolo_msgs::msg::Detection &detection,
                            const cv::Scalar &color) {
  const auto &center = detection.bbox.center.position;
  const double theta = detection.bbox.center.theta;

  // Label anchor, filled in below (top-left INSIDE the drawn box so a large
  // detection near the frame edge cannot push the label off screen).
  cv::Point label_anchor(0, 0);

  if (std::abs(theta) > 0.0) {
    // Oriented bounding box (OBB): the rotation angle rides in center.theta
    // (radians, ultralytics xywhr convention). Draw the rotated quad with the
    // same corner geometry as the OBB postprocessor.
    const float c = static_cast<float>(std::cos(theta));
    const float s = static_cast<float>(std::sin(theta));
    const cv::Point2f ctr(static_cast<float>(center.x),
                          static_cast<float>(center.y));
    const float w = static_cast<float>(detection.bbox.size.x);
    const float h = static_cast<float>(detection.bbox.size.y);
    const cv::Point2f vec1(w / 2 * c, w / 2 * s);
    const cv::Point2f vec2(-h / 2 * s, h / 2 * c);
    const std::array<cv::Point2f, 4> corners = {
        ctr + vec1 + vec2, ctr + vec1 - vec2, ctr - vec1 - vec2,
        ctr - vec1 + vec2};

    float min_x = corners[0].x, min_y = corners[0].y;
    std::vector<cv::Point> quad;
    quad.reserve(4);
    for (const auto &p : corners) {
      quad.emplace_back(cvRound(p.x), cvRound(p.y));
      min_x = std::min(min_x, p.x);
      min_y = std::min(min_y, p.y);
    }
    cv::polylines(image, quad, true, color, 2, cv::LINE_AA);
    label_anchor =
        cv::Point(cvRound(min_x) + kLabelInset, cvRound(min_y) + kLabelInset);
  } else {
    cv::Rect box(center.x - detection.bbox.size.x / 2,
                 center.y - detection.bbox.size.y / 2, detection.bbox.size.x,
                 detection.bbox.size.y);
    cv::rectangle(image, box, color, 2);
    label_anchor = cv::Point(box.x + kLabelInset, box.y + kLabelInset);
  }

  // Text
  draw_label(image, label_text(detection), label_anchor, color);

  return image;
}

int DebugNode::draw_label(const cv::Mat &image, const std::string &text,
                          const cv::Point &anchor, const cv::Scalar &background,
                          double font_scale, int thickness) {
  constexpr int font = cv::FONT_HERSHEY_SIMPLEX;
  constexpr int pad = 3;

  int baseline = 0;
  const cv::Size size =
      cv::getTextSize(text, font, font_scale, thickness, &baseline);
  const int label_w = size.width + 2 * pad;
  const int label_h = size.height + baseline + 2 * pad;

  // `anchor` is the desired top-left corner. Slide the label back inside the
  // frame when the requested corner would push it off screen (e.g. a detection
  // touching the border), so the text is never clipped.
  int x0 = anchor.x;
  int y0 = anchor.y;
  if (x0 + label_w > image.cols) {
    x0 = image.cols - label_w;
  }
  if (y0 + label_h > image.rows) {
    y0 = image.rows - label_h;
  }
  x0 = std::max(0, x0);
  y0 = std::max(0, y0);
  const int x1 = std::min(image.cols - 1, x0 + label_w - 1);
  const int y1 = std::min(image.rows - 1, y0 + label_h - 1);
  if (x1 <= x0 || y1 <= y0) {
    return 0; // degenerate or fully off-frame
  }

  cv::rectangle(image, cv::Point(x0, y0), cv::Point(x1, y1), background,
                cv::FILLED);

  // Pick black or white glyphs by WCAG contrast ratio (relative luminance)
  // against the background, so the label is readable for every class color.
  auto relative_luminance = [](const cv::Scalar &bgr) {
    auto channel = [](double value) {
      value /= 255.0;
      return value <= 0.03928 ? value / 12.92
                              : std::pow((value + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * channel(bgr[2]) + 0.7152 * channel(bgr[1]) +
           0.0722 * channel(bgr[0]);
  };
  const double luminance = relative_luminance(background);
  const double contrast_black = (luminance + 0.05) / 0.05;
  const double contrast_white = 1.05 / (luminance + 0.05);
  const cv::Scalar text_color = contrast_black >= contrast_white
                                    ? cv::Scalar(0, 0, 0)
                                    : cv::Scalar(255, 255, 255);

  cv::putText(image, text, cv::Point(x0 + pad, y0 + pad + size.height), font,
              font_scale, text_color, thickness, cv::LINE_AA);
  return label_h;
}

void DebugNode::draw_mask(cv::Mat &overlay, cv::Mat &image,
                          const yolo_msgs::msg::Detection &detection,
                          const cv::Scalar &color) {
  if (detection.mask.data.size() == 0) {
    return;
  }
  // Convert ROS Point2D (float64) mask boundary points to OpenCV points
  std::vector<std::vector<cv::Point>> contours(1);
  contours[0].reserve(detection.mask.data.size());
  for (const auto &p : detection.mask.data) {
    contours[0].emplace_back(cvRound(p.x), cvRound(p.y));
  }
  // Fill the mask onto the shared overlay layer (blended once by the caller)
  // and draw the crisp outline directly on the final image.
  cv::fillPoly(overlay, contours, color, cv::LINE_AA);
  cv::polylines(image, contours, true, color, 2, cv::LINE_AA);
}

cv::Mat DebugNode::draw_keypoints(const cv::Mat &image,
                                  const yolo_msgs::msg::Detection &detection) {
  if (detection.keypoints.data.size() == 0) {
    return image;
  }

  std::map<int, cv::Point> points;
  for (const auto &kp : detection.keypoints.data) {
    const auto pt = cv::Point(cvRound(kp.point.x), cvRound(kp.point.y));
    points[kp.id] = pt;
    const auto color_k = indexed_color(kp.id);
    cv::circle(image, pt, 5, color_k, -1, cv::LINE_AA);
    draw_label(image, std::to_string(kp.id), pt + cv::Point(6, -7), color_k,
               0.5, 1);
  }

  // Draw the skeleton limbs on top (per-limb colors, only when both
  // endpoints of the limb are present in the detection).
  for (int i = 0; i < kNumSkeletonLimbs; ++i) {
    auto it1 = points.find(kSkeleton[i][0]);
    auto it2 = points.find(kSkeleton[i][1]);
    if (it1 != points.end() && it2 != points.end()) {
      cv::line(image, it1->second, it2->second, indexed_color(i + 32), 2,
               cv::LINE_AA);
    }
  }
  return image;
}

visualization_msgs::msg::Marker
DebugNode::create_bb_marker(const yolo_msgs::msg::Detection &detection,
                            const cv::Scalar &color) {
  visualization_msgs::msg::Marker marker;

  marker.header.frame_id = detection.bbox3d.frame_id;
  marker.ns = "yolo_3d";
  marker.type = visualization_msgs::msg::Marker::CUBE;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.frame_locked = false;

  marker.pose.position.x = detection.bbox3d.center.position.x;
  marker.pose.position.y = detection.bbox3d.center.position.y;
  marker.pose.position.z = detection.bbox3d.center.position.z;
  marker.pose.orientation.x = detection.bbox3d.center.orientation.x;
  marker.pose.orientation.y = detection.bbox3d.center.orientation.y;
  marker.pose.orientation.z = detection.bbox3d.center.orientation.z;
  marker.pose.orientation.w = detection.bbox3d.center.orientation.w;

  marker.scale.x = detection.bbox3d.size.x;
  marker.scale.y = detection.bbox3d.size.y;
  marker.scale.z = detection.bbox3d.size.z;

  // The per-class color is stored as an OpenCV BGR scalar -> convert to RGB.
  marker.color.r = color[2] / 255.0;
  marker.color.g = color[1] / 255.0;
  marker.color.b = color[0] / 255.0;
  marker.color.a = 0.4;

  // Finite lifetime so a marker whose detection disappears from the 3D stream
  // (e.g. convert_bb_to_3d dropped it for invalid depth, or the stream stalled)
  // expires in RViz instead of lingering forever: RViz never removes a marker
  // just because it is missing from a later MarkerArray. The markers are
  // re-published on every 3D message, so a still-valid box keeps refreshing its
  // lifetime. 0 keeps the old persistent behaviour.
  marker.lifetime = duration_from_seconds(this->marker_lifetime_);
  marker.text = detection.class_name;

  return marker;
}

visualization_msgs::msg::Marker
DebugNode::create_kp_marker(const yolo_msgs::msg::KeyPoint3D &keypoint) {
  visualization_msgs::msg::Marker marker;

  marker.ns = "yolo_3d";
  marker.type = visualization_msgs::msg::Marker::SPHERE;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.frame_locked = false;

  marker.pose.position.x = keypoint.point.x;
  marker.pose.position.y = keypoint.point.y;
  marker.pose.position.z = keypoint.point.z;
  marker.pose.orientation.w = 1.0;

  marker.scale.x = 0.05;
  marker.scale.y = 0.05;
  marker.scale.z = 0.05;

  // Confidence gradient from red (low score) to blue (high score).
  marker.color.r = 1.0 - keypoint.score;
  marker.color.g = 0.0;
  marker.color.b = keypoint.score;
  marker.color.a = 0.4;

  // Same finite lifetime as the 3D box markers (see create_bb_marker): a
  // keypoint that vanishes from the stream must expire in RViz too.
  marker.lifetime = duration_from_seconds(this->marker_lifetime_);
  marker.text = std::to_string(keypoint.id);

  return marker;
}

visualization_msgs::msg::Marker
DebugNode::create_limb_marker(const yolo_msgs::msg::KeyPoint3D &from,
                              const yolo_msgs::msg::KeyPoint3D &to,
                              const cv::Scalar &color) {
  visualization_msgs::msg::Marker marker;

  marker.ns = "yolo_3d_limbs";
  marker.type = visualization_msgs::msg::Marker::LINE_LIST;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.frame_locked = false;

  // LINE_LIST consumes points in pairs; one limb is a single pair.
  geometry_msgs::msg::Point p_from;
  p_from.x = from.point.x;
  p_from.y = from.point.y;
  p_from.z = from.point.z;
  geometry_msgs::msg::Point p_to;
  p_to.x = to.point.x;
  p_to.y = to.point.y;
  p_to.z = to.point.z;
  marker.points = {p_from, p_to};

  marker.pose.orientation.w = 1.0;

  // For LINE_LIST, scale.x is the line width in metres.
  marker.scale.x = 0.02;

  // The per-limb color is stored as an OpenCV BGR scalar -> convert to RGB.
  marker.color.r = color[2] / 255.0;
  marker.color.g = color[1] / 255.0;
  marker.color.b = color[0] / 255.0;
  marker.color.a = 0.8;

  // Same finite lifetime as the other 3D markers (see create_bb_marker): a
  // limb that vanishes from the stream must expire in RViz too.
  marker.lifetime = duration_from_seconds(this->marker_lifetime_);

  return marker;
}

} // namespace yolo_ros::node
