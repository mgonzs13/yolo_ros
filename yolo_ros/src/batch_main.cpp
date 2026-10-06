// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include <memory>
#include <rclcpp/executors.hpp>

#include <lifecycle_msgs/msg/state.hpp>

#include "yolo_ros/node/batch_node.hpp"

using namespace yolo_ros::node;

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<BatchNode>();
  if (node->configure().id() !=
      lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE) {
    RCLCPP_ERROR(node->get_logger(), "Failed to configure %s",
                 node->get_name());
    rclcpp::shutdown();
    return 1;
  }
  if (node->activate().id() !=
      lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
    RCLCPP_ERROR(node->get_logger(), "Failed to activate %s", node->get_name());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node->get_node_base_interface());

  executor.spin();

  executor.remove_node(node->get_node_base_interface());
  node.reset();

  rclcpp::shutdown();

  return 0;
}
