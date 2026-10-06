// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

/// @file
/// @brief Lifecycle node running one batched YOLO model over several cameras.

#ifndef YOLO_ROS__NODE__BATCH_NODE_HPP_
#define YOLO_ROS__NODE__BATCH_NODE_HPP_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "rclcpp/timer.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "yolo_msgs/msg/detection_array.hpp"
#include "yolo_msgs/srv/set_classes.hpp"
#include "yolo_ros/engine/batch_scheduler.hpp"
#include "yolo_ros/engine/model.hpp"
#include "yolo_ros/yolo/utils.hpp"

/// @addtogroup yolo_nodes
/// @{
namespace yolo_ros::node {

/// @brief One source frame awaiting its batch slot.
struct PendingFrame {
  /// @brief Source image message, decoded on the worker thread.
  sensor_msgs::msg::Image::ConstSharedPtr image;
};

/// @brief Inference node that batches frames from several cameras.
///
/// Subscribes to one image topic per camera, stores the latest frame per
/// camera, and on a worker thread runs `Model::detect_batch` over whatever is
/// ready. Each camera's detections are published on `<name>/detections`, so a
/// downstream `tracking_node`/`detect_3d_node`/`debug_node` running under the
/// `/<ns>/<name>` namespace consumes them via its relative `detections` topic.
class BatchNode : public rclcpp_lifecycle::LifecycleNode {
public:
  /// @brief Construct the node and declare the parameters.
  BatchNode();
  /// @brief Stop the worker and release resources.
  ~BatchNode() override;

  /// @brief Declare parameters and load the model configuration.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State &state);
  /// @brief Build the model, subscribe to every camera and start the worker.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State &state);
  /// @brief Stop the worker and release subscriptions/publishers.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State &state);
  /// @brief Reset the model.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_cleanup(const rclcpp_lifecycle::State &state);
  /// @brief Tear everything down on shutdown.
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_shutdown(const rclcpp_lifecycle::State &state);

private:
  /// @brief Declare every ROS parameter with its default.
  void declare_params();
  /// @brief Read the parameters into the members.
  /// @return False when the camera/topic arrays are inconsistent.
  bool load_params();
  /// @brief Release the model, scheduler, services and topics.
  void teardown();

  /// @brief One image subscription callback per camera.
  /// @param[in] camera Camera index.
  /// @param[in] msg Incoming image.
  void image_callback(std::size_t camera,
                      const sensor_msgs::msg::Image::SharedPtr msg);
  /// @brief Worker callback: run the batch and publish per camera.
  /// @param[in] batch Drained (camera index, frame) pairs.
  void process_batch(std::vector<std::pair<std::size_t, PendingFrame>> batch);
  /// @brief Enable/disable inference at runtime.
  void enable_service_callback(
      const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
      std::shared_ptr<std_srvs::srv::SetBool::Response> response);
  /// @brief Restrict the published classes.
  void set_classes_callback(
      const std::shared_ptr<yolo_msgs::srv::SetClasses::Request> request,
      std::shared_ptr<yolo_msgs::srv::SetClasses::Response> response);
  /// @brief Log a one-line inference summary for the interval just elapsed.
  void report_stats();

  /// @brief Camera names (also the published topic prefixes).
  std::vector<std::string> camera_names_;
  /// @brief Image topic per camera (parallel to camera_names_).
  std::vector<std::string> image_topics_;
  /// @brief Maximum frames per session run.
  std::size_t max_batch_size_ = 8;
  /// @brief Cached inference parameters.
  yolo_ros::yolo::utils::YoloParams yolo_params_;
  /// @brief Whether the parameters have been declared.
  bool params_declared_ = false;

  /// @brief The shared detector.
  std::unique_ptr<yolo_ros::engine::Model> yolo_model_;
  /// @brief Latest-frame batch scheduler.
  std::unique_ptr<yolo_ros::engine::BatchScheduler<PendingFrame>> scheduler_;
  /// @brief Per-camera image subscriptions.
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr>
      image_subscriptions_;
  /// @brief Per-camera detection publishers.
  std::vector<rclcpp::Publisher<yolo_msgs::msg::DetectionArray>::SharedPtr>
      detection_publishers_;

  /// @brief Runtime inference gate (`enable` service).
  std::atomic<bool> enable_inference_{true};
  /// @brief Published-class filter (`set_classes` service); empty = all.
  std::set<std::string> allowed_classes_;
  /// @brief Guards allowed_classes_.
  std::mutex classes_mutex_;
  /// @brief Last processed frame time (`max_fps` cap).
  std::chrono::steady_clock::time_point last_inference_time_{};

  /// @brief Service toggling inference.
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_service_;
  /// @brief Service restricting published classes.
  rclcpp::Service<yolo_msgs::srv::SetClasses>::SharedPtr set_classes_service_;

  /// @brief Inference counters, written by the worker and read by the reporter.
  struct Stats {
    /// @brief Cumulative images inferred since activation.
    uint64_t processed_total = 0;
    /// @brief Images inferred in the current interval.
    uint64_t interval_images = 0;
    /// @brief Batches run in the current interval.
    uint64_t interval_batches = 0;
    /// @brief Summed detect_batch() wall time (ms) in the current interval.
    double interval_inference_ms = 0.0;
    /// @brief Per-camera images inferred in the current interval.
    std::vector<uint64_t> per_camera;
    /// @brief Distinct class names published in the current interval.
    std::set<std::string> interval_classes;
    /// @brief Start of the current reporting interval.
    std::chrono::steady_clock::time_point interval_start{};
  };
  /// @brief Guards stats_.
  std::mutex stats_mutex_;
  /// @brief Inference counters reported every 5 s.
  Stats stats_;
  /// @brief Periodic stats reporter.
  rclcpp::TimerBase::SharedPtr stats_timer_;
};

} // namespace yolo_ros::node
/// @}

#endif // YOLO_ROS__NODE__BATCH_NODE_HPP_
