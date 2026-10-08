// Copyright (c) 2026 Alejandro González Cantón
// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/yolo_node.hpp"

#include <exception>
#include <string>
#include <vector>

#include "rcl_interfaces/msg/parameter_descriptor.hpp"
#include "rclcpp/logging.hpp"

namespace yolo_ros {

YoloNode::YoloNode(const rclcpp::NodeOptions &options,
                   PluginHost::Factory factory)
    : rclcpp_lifecycle::LifecycleNode("yolo_node", options),
      camera_streams_(this->blackboard_, this->get_logger()),
      tf_buffer_(this->get_clock()), factory_(std::move(factory)) {
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.description =
      "Plugin instance names; each '<name>.plugin' selects the pluginlib class";
  this->declare_parameter<std::vector<std::string>>(
      "plugins", std::vector<std::string>{}, descriptor);
  rcl_interfaces::msg::ParameterDescriptor cameras_descriptor;
  cameras_descriptor.description =
      "Camera names; each <name> has rgb_topic/depth_topic/depth_info_topic";
  this->declare_parameter<std::vector<std::string>>(
      "cameras", std::vector<std::string>{}, cameras_descriptor);
}

YoloNode::~YoloNode() {
  if (this->active_ && this->host_) {
    this->host_->deactivate();
    this->topics_.destroy_entities();
  }
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
YoloNode::on_configure(const rclcpp_lifecycle::State &) {
  // Attach the listener to this node and let the node's own executor spin its
  // /tf subscriptions. Do NOT use the default spin_thread dedicated-thread
  // listener: its teardown joins a spinning executor and can race/deadlock on
  // rapid configure/cleanup cycles (hit by the gtests).
  this->tf_listener_ = std::make_shared<tf2_ros::TransformListener>(
      this->tf_buffer_, this->get_node_base_interface(),
      this->get_node_logging_interface(), this->get_node_parameters_interface(),
      this->get_node_topics_interface(), false);

  const auto specs = this->get_parameter("plugins").as_string_array();
  const auto cameras = this->get_parameter("cameras").as_string_array();
  std::string camera_error;

  if (!this->camera_streams_.configure(*this, cameras, camera_error)) {
    RCLCPP_ERROR(this->get_logger(), "camera configuration failed: %s",
                 camera_error.c_str());
    this->camera_streams_.deactivate();
    this->blackboard_.reset();
    this->tf_listener_.reset();
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
        CallbackReturn::FAILURE;
  }

  this->host_ = std::make_unique<PluginHost>(
      *this, this->blackboard_, this->topics_, this->camera_streams_,
      &this->tf_buffer_, this->factory_);

  std::string error;

  if (!this->host_->configure(specs, error)) {
    RCLCPP_ERROR(this->get_logger(), "on_configure failed: %s", error.c_str());
    this->camera_streams_.deactivate();
    this->host_->cleanup();
    this->host_.reset();
    this->topics_.reset();
    this->blackboard_.reset();
    this->tf_listener_.reset();
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
        CallbackReturn::FAILURE;
  }

  try {
    this->topics_.validate();
  } catch (const std::exception &e) {
    RCLCPP_ERROR(this->get_logger(), "topic validation failed: %s", e.what());
    this->camera_streams_.deactivate();
    this->host_->cleanup();
    this->host_.reset();
    this->topics_.reset();
    this->blackboard_.reset();
    this->tf_listener_.reset();
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
        CallbackReturn::FAILURE;
  }

  RCLCPP_INFO(this->get_logger(), "Configured %zu plugin(s)",
              this->host_->instances().size());
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
YoloNode::on_activate(const rclcpp_lifecycle::State &) {
  if (!this->host_) {
    RCLCPP_ERROR(this->get_logger(),
                 "on_activate called without a configured plugin host");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
        CallbackReturn::FAILURE;
  }

  std::string error;
  bool activated = false;

  try {
    if (this->camera_streams_.activate(*this, error)) {
      this->topics_.create_entities(*this, this->blackboard_);
      activated = this->host_->activate(error);
    }
  } catch (const std::exception &e) {
    error = std::string("exception during activate: ") + e.what();
  }

  if (activated) {
    this->active_ = true;
    RCLCPP_INFO(this->get_logger(), "Activated");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
        CallbackReturn::SUCCESS;
  }

  RCLCPP_ERROR(this->get_logger(), "on_activate failed: %s", error.c_str());
  // A failed activation does not necessarily run on_cleanup (Jazzy returns to
  // inactive), so perform the full cleanup here. Otherwise stale blackboard
  // channels and topic requests would leak into the next configure().
  this->camera_streams_.deactivate();
  this->topics_.destroy_entities();
  this->host_->cleanup();
  this->host_.reset();
  this->topics_.reset();
  this->blackboard_.reset();
  this->tf_listener_.reset();
  this->active_ = false;
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::FAILURE;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
YoloNode::on_deactivate(const rclcpp_lifecycle::State &) {
  if (this->host_) {
    this->host_->deactivate();
  }

  this->camera_streams_.deactivate();
  this->topics_.destroy_entities();
  this->active_ = false;
  RCLCPP_INFO(this->get_logger(), "Deactivated");
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
YoloNode::on_cleanup(const rclcpp_lifecycle::State &) {
  this->camera_streams_.deactivate();

  if (this->host_) {
    this->host_->cleanup();
    this->host_.reset();
  }

  this->topics_.reset();
  this->blackboard_.reset();
  this->tf_listener_.reset();
  this->active_ = false;
  RCLCPP_INFO(this->get_logger(), "Cleaned up");
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
YoloNode::on_shutdown(const rclcpp_lifecycle::State &) {
  if (this->active_ && this->host_) {
    this->host_->deactivate();
    this->topics_.destroy_entities();
    this->active_ = false;
  }

  this->camera_streams_.deactivate();

  if (this->host_) {
    this->host_->cleanup();
    this->host_.reset();
  }

  this->topics_.reset();
  this->blackboard_.reset();
  this->tf_listener_.reset();
  RCLCPP_INFO(this->get_logger(), "Shutting down");
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

} // namespace yolo_ros
