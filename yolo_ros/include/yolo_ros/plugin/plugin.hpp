// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief Plugin base class and shared context.

#ifndef YOLO_ROS__PLUGIN__PLUGIN_HPP_
#define YOLO_ROS__PLUGIN__PLUGIN_HPP_

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/clock.hpp"
#include "rclcpp/logger.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "tf2_ros/buffer.h"
#include "yolo_ros/blackboard/blackboard.hpp"
#include "yolo_ros/plugin/topic_registry.hpp"

namespace yolo_ros {

/// @brief One camera resolved for a plugin instance.
struct CameraInput {
  /// @brief Camera name ("cam0").
  std::string name;
  /// @brief Channel carrying this camera's CameraFrame ("cam0").
  std::string frame_channel;
  /// @brief Upstream DetectionArray channel for this camera (the previous
  /// plugin's output, or the frame channel for the first plugin).
  std::string input_channel;
  /// @brief Whether the camera has a depth stream.
  bool has_depth = false;
  /// @brief Output channels of every earlier plugin in the chain for this
  /// camera, in chain order (empty for the first plugin).
  std::vector<std::string> upstream_channels{};
};

/// @brief Services handed to a plugin during setup().
struct PluginContext {
  /// @brief Shared typed channel bus.
  Blackboard &blackboard;
  /// @brief Registry that owns the plugin's ROS topic I/O.
  TopicRegistry &topics;
  /// @brief Logger of the owning lifecycle node.
  rclcpp::Logger logger;
  /// @brief Clock of the owning lifecycle node.
  rclcpp::Clock::SharedPtr clock;
  /// @brief TF buffer of the owning node, or nullptr when unavailable.
  tf2_ros::Buffer *tf_buffer = nullptr;
  /// @brief Plugin instance name, reported as the owner in topic conflicts.
  std::string name{};
  /// @brief Cameras resolved for this instance, in configuration order.
  std::vector<CameraInput> cameras;
};

/// @brief Base class for every YoloNode plugin.
///
/// Lifecycle: declare_params() for all instances, then get_params() for all,
/// then setup() during on_configure; activate() then run() on a dedicated
/// thread during on_activate; deactivate() after the thread joined.
class Plugin {
public:
  /// @brief Destroy the plugin.
  virtual ~Plugin() = default;

  /// @brief Declare the instance parameters under @p prefix.
  /// @param node Lifecycle node owning the parameters.
  /// @param prefix Instance prefix including the trailing dot, e.g. "det.".
  virtual void declare_params(rclcpp_lifecycle::LifecycleNode &node,
                              const std::string &prefix) = 0;
  /// @brief Read the declared instance parameters under @p prefix.
  /// @param node Lifecycle node owning the parameters.
  /// @param prefix Instance prefix including the trailing dot, e.g. "det.".
  virtual void get_params(const rclcpp_lifecycle::LifecycleNode &node,
                          const std::string &prefix) = 0;

  /// @brief Channel this plugin produces for @p camera.
  /// @param camera Camera name.
  /// @return The output channel name (also exposed as a ROS topic).
  virtual std::string output_channel(const std::string &camera) const = 0;

  /// @brief Register blackboard channels and external topic I/O.
  /// @return False to fail on_configure.
  virtual bool setup(PluginContext &ctx) {
    (void)ctx;
    return true;
  }

  /// @brief Build heavy resources (model, tracker). Runs on the executor.
  /// @return False to fail on_activate.
  virtual bool activate() { return true; }

  /// @brief Release resources after the worker thread has joined.
  virtual void deactivate() {}

  /// @brief Worker loop; must poll @p stop and return promptly.
  virtual void run(const std::atomic<bool> &stop) = 0;
};

} // namespace yolo_ros

#endif // YOLO_ROS__PLUGIN__PLUGIN_HPP_
