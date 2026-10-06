// Copyright (c) 2025 Alejandro González Cantón
// Portions Copyright (c) 2023-2025 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include <memory>
#include <rclcpp/executors.hpp>

#include "yolo_ros/node/yolo_node.hpp"

using namespace yolo_ros::node;

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<YoloNode>();
  node->configure();
  node->activate();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node->get_node_base_interface());

  executor.spin();

  executor.remove_node(node->get_node_base_interface());
  node.reset();

  rclcpp::shutdown();

  return 0;
}