// Copyright (c) 2025 Alejandro González Cantón
// Portions Copyright (c) 2023-2025 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief Lifecycle node running a single YOLO model over an image topic.

#ifndef YOLO_ROS__NODE__YOLO_NODE_HPP_
#define YOLO_ROS__NODE__YOLO_NODE_HPP_

#include "rclcpp_lifecycle/lifecycle_node.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <set>
#include <string>

#include "sensor_msgs/msg/image.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "yolo_msgs/msg/detection_array.hpp"
#include "yolo_msgs/srv/set_classes.hpp"
#include "yolo_ros/engine/model.hpp"
#include "yolo_ros/yolo/utils.hpp"

/// @addtogroup yolo_nodes
/// @{
namespace yolo_ros::node {
/// @brief Lifecycle node that creates the requested YOLO model on configure,
/// runs inference on the image topic while active and publishes a
/// DetectionArray.
///
/// The task (detect/segment/pose/OBB/classify) is selected by the `model_type`
/// parameter. Inference can be enabled/disabled at runtime through the
/// `enable` SetBool service, the published classes can be restricted through
/// the `set_classes` service, and the publish rate is capped by `max_fps`.
class YoloNode : public rclcpp_lifecycle::LifecycleNode {
public:
  /// @brief Construct the node and declare the parameters.
  YoloNode();

  /// @brief Create the model from the parameters and set up publishers,
  /// subscriptions and the enable service.
  /// @param[in] state Previous lifecycle state.
  /// @return SUCCESS on success, FAILURE otherwise.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State &state);
  /// @brief Activate the publishers and start processing images.
  /// @param[in] state Previous lifecycle state.
  /// @return SUCCESS on success, FAILURE otherwise.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State &state);
  /// @brief Deactivate the publishers and stop processing images.
  /// @param[in] state Previous lifecycle state.
  /// @return SUCCESS on success, FAILURE otherwise.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State &state);
  /// @brief Destroy the model and release publishers/subscriptions.
  /// @param[in] state Previous lifecycle state.
  /// @return SUCCESS on success, FAILURE otherwise.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_cleanup(const rclcpp_lifecycle::State &state);
  /// @brief Tear everything down on shutdown.
  /// @param[in] state Previous lifecycle state.
  /// @return SUCCESS on success, FAILURE otherwise.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_shutdown(const rclcpp_lifecycle::State &state);

protected:
  /// @brief The loaded YOLO model (null until configured).
  std::unique_ptr<yolo_ros::engine::Model> yolo_model;

  /// @brief Publisher of the detection results.
  rclcpp::Publisher<yolo_msgs::msg::DetectionArray>::SharedPtr
      detection_publisher;
  /// @brief Subscription to the input image topic.
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription;

  /// @brief Declare all ROS parameters with their defaults.
  void declare_params();
  /// @brief Read the declared parameters into a YoloParams struct.
  /// @return The populated parameters.
  yolo_ros::yolo::utils::YoloParams get_params();

  /// @brief Cached parameters used to build the model.
  yolo_ros::yolo::utils::YoloParams yolo_params;
  /// @brief Whether the parameters have already been declared.
  bool params_declared = false;

  // Runtime inference gate, toggled by the `enable` service (SetBool); the
  // `enable` parameter only provides the initial value. Atomic because the
  // image subscription and the service may run on different executor threads.
  /// @brief Runtime inference gate, toggled by the enable service.
  std::atomic<bool> enable_inference_{true};

  // Published-class filter, set through the set_classes service. Empty means
  // every class is published. Guarded because the service and the image
  // callback may run on different executor threads.
  /// @brief Class names to publish; empty publishes every class.
  std::set<std::string> allowed_classes_;
  /// @brief Guards allowed_classes_.
  std::mutex classes_mutex_;

  // Timestamp of the last processed frame, used by the max_fps frequency cap
  // to decide whether the current frame should be dropped.
  /// @brief Timestamp of the last processed frame (max_fps cap).
  std::chrono::steady_clock::time_point last_inference_time_{};

  /// @brief Build the model selected by @p params.
  /// @param params Model and task configuration.
  void create_yolo(yolo_ros::yolo::utils::YoloParams params);
  /// @brief Destroy the current model, if any.
  void destroy_yolo();

private:
  /// @brief Image subscription callback: preprocess, infer and publish.
  /// @param[in] msg Incoming image message.
  void recieve_image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  /// @brief Enable/disable inference at runtime.
  /// @param[in] request Request with the desired enabled flag.
  /// @param[out] response Response reporting success and the new state.
  void enable_service_callback(
      const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
      std::shared_ptr<std_srvs::srv::SetBool::Response> response);
  /// @brief Service toggling the inference gate.
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_service_;

  /// @brief Restrict the published classes to the requested names (empty
  /// publishes every class).
  /// @param[in] request Request with the class names to keep.
  /// @param[out] response Response reporting success.
  void set_classes_callback(
      const std::shared_ptr<yolo_msgs::srv::SetClasses::Request> request,
      std::shared_ptr<yolo_msgs::srv::SetClasses::Response> response);
  /// @brief Service restricting the published classes.
  rclcpp::Service<yolo_msgs::srv::SetClasses>::SharedPtr set_classes_service_;
};
} // namespace yolo_ros::node
/// @}

#endif // YOLO_ROS__NODE__YOLO_NODE_HPP_
