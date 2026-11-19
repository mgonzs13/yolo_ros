// Copyright (C) 2026 Alejandro González Cantón
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

/// @file
/// @brief Entry point of the single YoloNode executable.

#include <exception>
#include <memory>

#include "lifecycle_msgs/msg/state.hpp"
#include "rclcpp/executors/single_threaded_executor.hpp"
#include "rclcpp/rclcpp.hpp"
#include "yolo_ros/node/yolo_node.hpp"

/// @brief Initialize ROS 2, configure the YoloNode and spin until shutdown.
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
