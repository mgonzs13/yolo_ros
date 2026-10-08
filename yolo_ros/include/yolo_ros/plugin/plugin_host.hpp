// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief Loads plugins, runs their lifecycle and owns their worker threads.

#ifndef YOLO_ROS__PLUGIN__PLUGIN_HOST_HPP_
#define YOLO_ROS__PLUGIN__PLUGIN_HOST_HPP_

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "pluginlib/class_loader.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "tf2_ros/buffer.h"
#include "yolo_ros/blackboard/blackboard.hpp"
#include "yolo_ros/camera/camera_streams.hpp"
#include "yolo_ros/plugin/plugin.hpp"
#include "yolo_ros/plugin/topic_registry.hpp"

namespace yolo_ros {

/// @brief One loaded plugin instance.
struct PluginInstance {
  /// @brief Instance name from the `plugins` parameter list.
  std::string name;
  /// @brief Pluginlib type name.
  std::string type;
  /// @brief Loaded plugin object.
  std::shared_ptr<Plugin> plugin;
  /// @brief Worker thread running the plugin; empty until activate().
  std::thread thread;
};

/// @brief Owns plugin instances and their worker threads.
///
/// The factory is injectable so tests can use in-process fakes; the default
/// factory loads pluginlib classes from the `yolo_ros` package.
class PluginHost {
public:
  /// @brief Loads a plugin of @p type, reporting failures in @p error.
  using Factory = std::function<std::shared_ptr<Plugin>(const std::string &type,
                                                        std::string &error)>;

  /// @brief Bind the host to its node services and plugin factory.
  /// @param node Lifecycle node owning the plugin parameters.
  /// @param blackboard Shared channel bus handed to plugins.
  /// @param topics Topic registry handed to plugins.
  /// @param camera_streams Validated node cameras resolved for each plugin.
  /// @param tf_buffer TF buffer handed to plugins, or nullptr.
  /// @param factory Custom plugin factory; the default uses pluginlib.
  PluginHost(rclcpp_lifecycle::LifecycleNode &node, Blackboard &blackboard,
             TopicRegistry &topics, CameraStreams &camera_streams,
             tf2_ros::Buffer *tf_buffer = nullptr, Factory factory = Factory());

  /// @brief Stops and joins any running worker threads before destruction.
  ~PluginHost();

  /// @brief Parse instance names, resolve each `<name>.plugin` class, then
  /// declare/get plugin params and setup() every instance in order.
  /// @param names Plugin instance names from the `plugins` parameter; each must
  /// be non-empty and contain none of ':', '.' or '/'.
  /// @param error Receives a failure description when configuration fails.
  /// @return True when every instance configured successfully.
  bool configure(const std::vector<std::string> &names, std::string &error);

  /// @brief activate() every plugin, then start one thread per instance.
  bool activate(std::string &error);

  /// @brief Stop and join every thread, then deactivate() every plugin.
  void deactivate();

  /// @brief Drop all instances (after deactivate()).
  void cleanup();

  /// @brief Currently loaded plugin instances.
  /// @return Read-only view of the loaded instances.
  const std::vector<PluginInstance> &instances() const {
    return this->instances_;
  }

private:
  /// @brief Instantiate a plugin through the factory or pluginlib loader.
  /// @param type Pluginlib type name.
  /// @param error Receives a failure description when loading fails.
  /// @return Loaded plugin, or nullptr on failure.
  std::shared_ptr<Plugin> create(const std::string &type, std::string &error);

  /// @brief Worker entry point running plugin @p index until stopped.
  /// @param index Index into instances_.
  void run_plugin(std::size_t index);

  /// @brief Stop and join any started threads, then deactivate every plugin.
  void rollback_activation();

  /// @brief Set every declared instance parameter to its node override.
  ///
  /// Statically typed parameters cannot be undeclared on Jazzy, so a retry
  /// after a failed configure reuses the parameters declared by the previous
  /// cycle; this reapplies the NodeOptions/YAML overrides so changed values
  /// still take effect.
  /// @param error Receives a failure description when an override cannot be
  /// applied.
  /// @return True when every override applied or none exists.
  bool apply_overrides(std::string &error);

  /// @brief Validate the order-based chain rules.
  /// @param error Receives a failure description when the chain is invalid.
  /// @return True when the chain is valid.
  bool validate_chain(std::string &error) const;

  /// @brief Resolve the camera list of @p index from its `cameras` parameter.
  /// @param index Index into instances_.
  /// @param previous Cameras resolved for instance @p index - 1; ignored when
  /// @p index is 0.
  /// @param cameras Receives the resolved camera inputs.
  /// @param error Receives a failure description when a camera is unknown,
  /// duplicated in the list, or not produced by the previous plugin.
  /// @return True when every camera resolves.
  bool resolve_cameras(std::size_t index,
                       const std::vector<std::string> &previous,
                       std::vector<CameraInput> &cameras,
                       std::string &error) const;

  /// @brief Lifecycle node owning the plugin parameters.
  rclcpp_lifecycle::LifecycleNode &node_;
  /// @brief Shared channel bus handed to plugins.
  Blackboard &blackboard_;
  /// @brief Topic registry handed to plugins.
  TopicRegistry &topics_;
  /// @brief Validated node cameras resolved for each plugin.
  CameraStreams &camera_streams_;
  /// @brief TF buffer handed to plugins, or nullptr when unavailable.
  tf2_ros::Buffer *tf_buffer_;
  /// @brief Factory used to build plugin instances.
  Factory factory_;
  /// @brief Lazily created pluginlib loader for the default factory.
  std::unique_ptr<pluginlib::ClassLoader<Plugin>> loader_;
  /// @brief Loaded plugin instances, in configure order.
  std::vector<PluginInstance> instances_;
  /// @brief True after activate() fully succeeds, until deactivate() runs.
  bool activated_ = false;
  /// @brief Set to stop every worker thread; read by running plugins.
  std::atomic<bool> stop_{false};
};

} // namespace yolo_ros

#endif // YOLO_ROS__PLUGIN__PLUGIN_HOST_HPP_
