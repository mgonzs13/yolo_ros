// Copyright (c) 2026 Alejandro González Cantón
// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief The single lifecycle node hosting every pipeline plugin.

#ifndef YOLO_ROS__YOLO_NODE_HPP_
#define YOLO_ROS__YOLO_NODE_HPP_

#include <memory>
#include <string>
#include <vector>

#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "yolo_ros/blackboard/blackboard.hpp"
#include "yolo_ros/camera/camera_streams.hpp"
#include "yolo_ros/plugin/plugin_host.hpp"
#include "yolo_ros/plugin/topic_registry.hpp"

namespace yolo_ros {

/// @brief Loads plugin instances listed in the `plugins` parameter and drives
/// their lifecycle.
///
/// `plugins` holds plain instance names. For each name, the `<name>.plugin`
/// parameter selects the pluginlib class, and all other plugin parameters live
/// under the instance prefix (e.g. `det.threshold`).
class YoloNode : public rclcpp_lifecycle::LifecycleNode {
public:
  /// @brief Construct the node and declare the `plugins` parameter.
  /// @param options Node options forwarded to the lifecycle base.
  /// @param factory Plugin factory used by the host; defaults to pluginlib.
  explicit YoloNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions(),
                    PluginHost::Factory factory = PluginHost::Factory());

  /// @brief Tear down any still-active plugin host before destruction.
  ~YoloNode() override;

  /// @brief Create the TF listener and configure every plugin instance.
  /// @param state Current lifecycle state (unused).
  /// @return SUCCESS when all plugins configure, FAILURE otherwise.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State &state) override;

  /// @brief Create topic entities and activate all plugins.
  /// @param state Current lifecycle state (unused).
  /// @return SUCCESS when activation succeeds, FAILURE otherwise.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State &state) override;

  /// @brief Deactivate all plugins and destroy the topic entities.
  /// @param state Current lifecycle state (unused).
  /// @return Always SUCCESS.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State &state) override;

  /// @brief Drop the plugin host, topics, blackboard and TF listener.
  /// @param state Current lifecycle state (unused).
  /// @return Always SUCCESS.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_cleanup(const rclcpp_lifecycle::State &state) override;

  /// @brief Fully deactivate and clean up before final shutdown.
  /// @param state Current lifecycle state (unused).
  /// @return Always SUCCESS.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_shutdown(const rclcpp_lifecycle::State &state) override;

  /// @brief Access the pipeline blackboard.
  /// @return Mutable reference to the blackboard.
  Blackboard &blackboard() { return this->blackboard_; }

  /// @brief Access the node's topic registry.
  /// @return Mutable reference to the registry.
  TopicRegistry &topics() { return this->topics_; }

  /// @brief Access the configured plugin host.
  /// @return Pointer to the host, or nullptr when not configured.
  PluginHost *host() { return this->host_.get(); }

private:
  /// @brief Shared state exchanged between plugin instances.
  Blackboard blackboard_;

  /// @brief Node-level cameras publishing synchronized frames on the
  /// blackboard.
  CameraStreams camera_streams_;

  /// @brief Topics owned by the node on behalf of the plugins.
  TopicRegistry topics_;

  /// @brief TF buffer owned by the node; outlives the listener.
  tf2_ros::Buffer tf_buffer_;

  /// @brief Listener filling `tf_buffer_` from the node's own executor.
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  /// @brief Factory used to create plugin instances.
  PluginHost::Factory factory_;

  /// @brief Configured plugin host, null until on_configure() succeeds.
  std::unique_ptr<PluginHost> host_;

  /// @brief True between a successful activate and the next deactivate.
  bool active_ = false;
};

} // namespace yolo_ros

#endif // YOLO_ROS__YOLO_NODE_HPP_
