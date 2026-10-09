// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/camera/camera_streams.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rclcpp/qos.hpp"
#include "yolo_ros/utils/qos_compat.hpp"

namespace yolo_ros {

namespace {

/// @brief Whether @p name is a valid camera name.
bool valid_name(const std::string &name) {
  return !name.empty() && name.find('.') == std::string::npos &&
         name.find(':') == std::string::npos &&
         name.find('/') == std::string::npos;
}

} // namespace

CameraStreams::CameraStreams(Blackboard &blackboard, rclcpp::Logger /*logger*/)
    : blackboard_(blackboard) {}

bool CameraStreams::configure(rclcpp_lifecycle::LifecycleNode &node,
                              const std::vector<std::string> &names,
                              std::string &error) {
  this->cameras_.clear();
  std::vector<CameraConfig> validated;
  validated.reserve(names.size());
  std::unordered_set<std::string> seen_names;
  std::unordered_set<std::string> seen_topics;

  for (const auto &name : names) {
    if (!valid_name(name)) {
      error = "invalid camera name '" + name + "'";
      return false;
    }

    if (!seen_names.insert(name).second) {
      error = "duplicate camera name '" + name + "'";
      return false;
    }

    CameraConfig camera;
    camera.name = name;

    if (!node.has_parameter(name + ".rgb_topic")) {
      node.declare_parameter<std::string>(name + ".rgb_topic", "");
    }

    if (!node.has_parameter(name + ".depth_topic")) {
      node.declare_parameter<std::string>(name + ".depth_topic", "");
    }

    if (!node.has_parameter(name + ".depth_info_topic")) {
      node.declare_parameter<std::string>(name + ".depth_info_topic", "");
    }

    if (!node.has_parameter(name + ".image_reliability")) {
      node.declare_parameter<int>(name + ".image_reliability", 2);
    }

    if (!node.has_parameter(name + ".depth_reliability")) {
      node.declare_parameter<int>(name + ".depth_reliability", 2);
    }

    node.get_parameter(name + ".rgb_topic", camera.rgb_topic);
    node.get_parameter(name + ".depth_topic", camera.depth_topic);
    node.get_parameter(name + ".depth_info_topic", camera.depth_info_topic);
    node.get_parameter(name + ".image_reliability", camera.image_reliability);
    node.get_parameter(name + ".depth_reliability", camera.depth_reliability);

    if (camera.rgb_topic.empty()) {
      error = "camera '" + name + "': missing '" + name + ".rgb_topic'";
      return false;
    }

    if (camera.depth_topic.empty() != camera.depth_info_topic.empty()) {
      error = "camera '" + name +
              "': depth_topic and depth_info_topic must be set together";
      return false;
    }

    if (!seen_topics.insert(camera.rgb_topic).second) {
      error = "camera '" + name + "': topic '" + camera.rgb_topic +
              "' is already used by another camera";
      return false;
    }

    if (camera.has_depth() && !seen_topics.insert(camera.depth_topic).second) {
      error = "camera '" + name + "': topic '" + camera.depth_topic +
              "' is already used by another camera";
      return false;
    }

    if (camera.has_depth() &&
        !seen_topics.insert(camera.depth_info_topic).second) {
      error = "camera '" + name + "': topic '" + camera.depth_info_topic +
              "' is already used by another camera";
      return false;
    }

    validated.push_back(std::move(camera));
  }

  // Commit only after every camera validated, so a failure leaves no channels
  // declared and cameras_ empty.
  for (const auto &camera : validated) {
    this->blackboard_.declare_channel<CameraFrame>(camera.name);
  }

  this->cameras_ = std::move(validated);
  return true;
}

bool CameraStreams::activate(rclcpp_lifecycle::LifecycleNode &node,
                             std::string &error) {
  this->streams_.clear();
  this->streams_.reserve(this->cameras_.size());

  for (std::size_t i = 0; i < this->cameras_.size(); ++i) {
    const auto &camera = this->cameras_[i];
    this->streams_.push_back(std::make_unique<Stream>());
    auto &stream = *this->streams_[i];
    const std::string name = camera.name;

    try {
      const auto image_qos = rclcpp::QoS(1).reliability(
          yolo_ros::utils::reliability_policy_from_int(
              camera.image_reliability));

      if (!camera.has_depth()) {
        stream.rgb_only_subscription =
            node.create_subscription<sensor_msgs::msg::Image>(
                camera.rgb_topic, image_qos,
                [this, name](sensor_msgs::msg::Image::ConstSharedPtr msg) {
                  this->publish_frame(name, msg, nullptr, nullptr);
                });
        continue;
      }

      const auto depth_qos = rclcpp::QoS(1).reliability(
          yolo_ros::utils::reliability_policy_from_int(
              camera.depth_reliability));
      stream.rgb_subscription.subscribe(node.shared_from_this(),
                                        camera.rgb_topic, image_qos);
      stream.depth_subscription.subscribe(node.shared_from_this(),
                                          camera.depth_topic, depth_qos);
      stream.info_subscription.subscribe(node.shared_from_this(),
                                         camera.depth_info_topic, depth_qos);

      stream.synchronizer =
          std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
              SyncPolicy(10), stream.rgb_subscription,
              stream.depth_subscription, stream.info_subscription);
      auto callback =
          [this,
           name](const sensor_msgs::msg::Image::ConstSharedPtr &rgb,
                 const sensor_msgs::msg::Image::ConstSharedPtr &depth,
                 const sensor_msgs::msg::CameraInfo::ConstSharedPtr &info) {
            this->publish_frame(name, rgb, depth, info);
          };
#if defined(MESSAGE_FILTERS_LEGACY_API) || defined(MESSAGE_FILTERS_OLD_API)
      // OLD-API Signal9 registers callbacks through a nine-slot std::bind
      // padded with NullType, so a plain three-argument lambda is not
      // invocable. Binding it to the matching placeholders lets std::bind
      // discard the padding; the wrapper is only needed on distros that take
      // that padded path.
      stream.synchronizer->registerCallback(
          std::bind(callback, std::placeholders::_1, std::placeholders::_2,
                    std::placeholders::_3));
#else
      stream.synchronizer->registerCallback(callback);
#endif
    } catch (const std::exception &e) {
      error = "camera '" + name + "': \"" + e.what() + "\"";
      this->deactivate();
      return false;
    }
  }

  return true;
}

void CameraStreams::deactivate() {
  for (auto &stream_ptr : this->streams_) {
    auto &stream = *stream_ptr;
    stream.rgb_only_subscription.reset();
    stream.rgb_subscription.unsubscribe();
    stream.depth_subscription.unsubscribe();
    stream.info_subscription.unsubscribe();
    stream.synchronizer.reset();
  }

  this->streams_.clear();
}

const CameraConfig *CameraStreams::find(const std::string &name) const {
  for (const auto &camera : this->cameras_) {
    if (camera.name == name) {
      return &camera;
    }
  }

  return nullptr;
}

void CameraStreams::publish_frame(
    const std::string &name, const sensor_msgs::msg::Image::ConstSharedPtr &rgb,
    const sensor_msgs::msg::Image::ConstSharedPtr &depth,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr &info) {
  auto frame = std::make_shared<CameraFrame>();
  frame->header = rgb->header;
  frame->rgb = rgb;
  frame->depth = depth;
  frame->depth_info = info;
  this->blackboard_.publish<CameraFrame>(name, frame);
}

} // namespace yolo_ros
