// Copyright (c) 2026 Alejandro González Cantón
// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief Entry point of the single YoloNode executable.

#include <exception>
#include <memory>

#include "lifecycle_msgs/msg/state.hpp"
#include "rclcpp/executors/single_threaded_executor.hpp"
#include "rclcpp/rclcpp.hpp"
#include "yolo_ros/node/yolo_node.hpp"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  try {
    auto node = std::make_shared<yolo_ros::node::YoloNode>();
    const auto configured = node->configure();

    if (configured.id() != lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE) {
      RCLCPP_ERROR(node->get_logger(), "Failed to configure YoloNode");
      node.reset();
      rclcpp::shutdown();
      return 1;
    }

    const auto activated = node->activate();

    if (activated.id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
      RCLCPP_ERROR(node->get_logger(), "Failed to activate YoloNode");
      node->cleanup();
      node.reset();
      rclcpp::shutdown();
      return 1;
    }

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node->get_node_base_interface());
    executor.spin();
    executor.remove_node(node->get_node_base_interface());

    node->deactivate();
    node->cleanup();
    node.reset();
  } catch (const std::exception &e) {
    fprintf(stderr, "yolo_node exception: %s\n", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
