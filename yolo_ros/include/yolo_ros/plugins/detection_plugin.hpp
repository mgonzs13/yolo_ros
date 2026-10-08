// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief Merged single/multi-camera YOLO inference plugin.

#ifndef YOLO_ROS__PLUGINS__DETECTION_PLUGIN_HPP_
#define YOLO_ROS__PLUGINS__DETECTION_PLUGIN_HPP_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "yolo_msgs/msg/detection_array.hpp"
#include "yolo_ros/blackboard/blackboard.hpp"
#include "yolo_ros/camera/camera_frame.hpp"
#include "yolo_ros/engine/batch_scheduler.hpp"
#include "yolo_ros/engine/model.hpp"
#include "yolo_ros/plugin/plugin.hpp"
#include "yolo_ros/yolo/utils.hpp"

namespace yolo_ros {

/// @brief Runs one YOLO model over one or more camera topics.
///
/// One camera: decode + detect on the worker thread. Several cameras: the
/// worker thread feeds a BatchScheduler (latest frame wins per camera) and
/// decode + detect_batch run on the scheduler thread.
class DetectionPlugin : public Plugin {
public:
  /// @brief Received camera frame waiting to be decoded and inferred.
  using PendingFrame = std::shared_ptr<const CameraFrame>;

  /// @brief Declare the model, camera and inference parameters.
  /// @param[in,out] node Lifecycle node owning the parameters.
  /// @param[in] prefix Instance prefix including the trailing dot, e.g. "det.".
  void declare_params(rclcpp_lifecycle::LifecycleNode &node,
                      const std::string &prefix) override;

  /// @brief Read the declared model, camera and inference parameters.
  /// @param[in] node Lifecycle node owning the parameters.
  /// @param[in] prefix Instance prefix including the trailing dot, e.g. "det.".
  void get_params(const rclcpp_lifecycle::LifecycleNode &node,
                  const std::string &prefix) override;

  /// @brief Detection channel produced for @p camera.
  /// @param[in] camera Camera name.
  /// @return `<camera>/detections`.
  std::string output_channel(const std::string &camera) const override {
    return camera + "/detections";
  }

  /// @brief Subscribe every camera channel and expose the detection topics.
  /// @param[in] ctx Plugin context carrying the blackboard and topic registry.
  /// @return False when the camera configuration is invalid.
  bool setup(PluginContext &ctx) override;

  /// @brief Create the YOLO model and, with several cameras, start the batch
  /// scheduler.
  /// @return False when the model cannot be created.
  bool activate() override;

  /// @brief Stop the scheduler and release the model.
  void deactivate() override;

  /// @brief Feed camera frames to the model (or scheduler) until @p stop.
  /// @param[in] stop Flag polled to leave the loop.
  void run(const std::atomic<bool> &stop) override;

  /// @brief Effective max batch size after clamping (diagnostics).
  /// @return The clamped maximum batch size.
  std::size_t max_batch_size() const { return this->max_batch_size_; }

  /// @brief Effective max_det after clamping in get_params() (diagnostics).
  /// @return The clamped cap on published detections (0 drops all of them).
  int max_det() const { return this->yolo_params_.max_det; }

private:
  /// @brief One selected camera and its frame reader.
  struct CameraStream {
    /// @brief Camera name.
    std::string name;
    /// @brief Output channel/topic for this camera (`<cam>/detections`).
    std::string output_channel;
    /// @brief Keep-latest reader of synchronized camera frames.
    ChannelReader<CameraFrame> reader;
  };

  /// @brief Run the model on one frame and cap the detections to max_det.
  /// @param[in] frame Source camera frame.
  /// @return The detection array (empty on decode/inference failure).
  yolo_msgs::msg::DetectionArray detect_and_filter(const CameraFrame &frame);

  /// @brief Publish @p message on the @p channel blackboard topic.
  /// @param[in] channel Output channel name.
  /// @param[in] message Detection array to publish.
  void publish(const std::string &channel,
               const yolo_msgs::msg::DetectionArray &message);

  /// @brief Create the YOLO model, downloading from Hugging Face when
  /// configured.
  /// @param[out] error Failure description when the model cannot be created.
  /// @return True when the model is ready.
  bool create_model(std::string &error);

  /// @brief Decode, infer and publish one scheduler batch.
  /// @param[in] batch Batch of (camera index, pending frame) pairs.
  void process_batch(const engine::BatchScheduler<PendingFrame>::Batch &batch);

  /// @brief Shared plugin context (blackboard, topics, logger).
  std::unique_ptr<PluginContext> context_;
  /// @brief Model path, task and inference parameters.
  yolo_ros::yolo::utils::YoloParams yolo_params_;
  /// @brief Configured camera streams.
  std::vector<CameraStream> cameras_;
  /// @brief Loaded YOLO model, null until activate().
  std::unique_ptr<yolo_ros::engine::Model> model_;
  /// @brief Batch scheduler used with several cameras, null otherwise.
  std::unique_ptr<engine::BatchScheduler<PendingFrame>> scheduler_;
  /// @brief Maximum batch size after clamping.
  std::size_t max_batch_size_ = 8;
  /// @brief Timestamp of the last processed frame (max_fps throttling).
  std::chrono::steady_clock::time_point last_frame_;
};

} // namespace yolo_ros

#endif // YOLO_ROS__PLUGINS__DETECTION_PLUGIN_HPP_
