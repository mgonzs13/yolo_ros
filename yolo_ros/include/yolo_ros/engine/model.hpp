// Copyright (c) 2025 Alejandro González Cantón
// SPDX-License-Identifier: MIT

/// @file
/// @brief Base ONNX Runtime model: preprocessing, inference and the
/// postprocessing hook overridden by each YOLO task.

#ifndef YOLO_ROS__ENGINE__MODEL_HPP_
#define YOLO_ROS__ENGINE__MODEL_HPP_

#include "yolo_msgs/msg/detection.hpp"
#include "yolo_ros/yolo/utils.hpp"

#if defined(CV_BRIDGE_H)
#include <cv_bridge/cv_bridge.h>
#else
#include <cv_bridge/cv_bridge.hpp>
#endif

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "yolo_ros/engine/ort_compat.hpp"

#include <cstddef>
#include <string>
#include <vector>

/// @addtogroup yolo_engine
/// @{
namespace yolo_ros::engine {
/// @brief Base class for a single ONNX Runtime YOLO model.
///
/// Owns the ONNX Runtime environment, session and the reusable input buffer.
/// The pipeline is fixed: preprocess() letterboxes the image into the model's
/// input tensor, inference() runs the session and postprocess() (virtual,
/// overridden by the concrete task classes) decodes the raw output tensors
/// into yolo_msgs::msg::Detection messages.
class Model {
public:
  /// @brief Load @p params.model_path (or download it from the Hugging Face
  /// Hub) and create the ONNX Runtime session, then read the class vocabulary.
  /// @param params Model, task and preprocessing configuration.
  /// @param task Resolved task name (e.g. "detect", "segment") reported in the
  /// load log; the base class cannot derive it from @p params alone.
  Model(yolo_ros::yolo::utils::YoloParams params, const std::string &task);
  /// @brief Destroy the model and release the ONNX Runtime session.
  ~Model();

  /// @brief Run the full pipeline (preprocess, inference, postprocess) on one
  /// image.
  /// @param image BGR image in original (un-letterboxed) coordinates.
  /// @return One detection per kept object, in original-image coordinates.
  std::vector<yolo_msgs::msg::Detection> detect(const cv::Mat &image);

  /// @brief Run the full pipeline on a batch of images with one session run.
  /// @param images BGR images in original (un-letterboxed) coordinates; they
  /// may have different original sizes.
  /// @return One detection vector per input image, in input order.
  std::vector<std::vector<yolo_msgs::msg::Detection>>
  detect_batch(const std::vector<cv::Mat> &images);

  /// @brief Name of the execution provider that initialized the session.
  /// @return "cpu", "cuda" or "tensorrt".
  const std::string &active_provider() const { return this->active_provider_; }

  /// @brief Whether inference runs through a captured CUDA graph.
  /// @return true when the session uses CUDA graph replay.
  bool using_cuda_graph() const { return this->cuda_graph_; }

  /// @brief Confidence threshold for detections, in [0, 1]. @see detect()
  float conf_threshold{0.5}; // Confidence threshold for detections
  /// @brief IoU threshold used by the C++ NMS, in [0, 1]. @see detect()
  float iou_threshold{0.5}; // Intersection over union threshold for detections

protected:
  /// @brief Class vocabulary indexed by class id. Populated from the ONNX
  /// graph metadata, or from the coco.names fallback. @see load_class_names()
  std::vector<std::string>
      class_names; // Vector of class names loaded from file
  /// @brief Whether class_names came from the ONNX graph metadata (true) or
  /// from the coco.names fallback (false). @see load_class_names()
  bool class_names_from_metadata_{false};

private:
  /// @brief Fixed batch size read from the graph; 0 when the axis is dynamic.
  int64_t fixed_batch_ = 0;
  /// @brief Letterbox @p image into slice @p index of the reused batch buffer.
  /// @param[in] image BGR image to preprocess.
  /// @param[in] input_tensor_shape Batch shape to write into.
  /// @param[in] index Batch slot to write.
  void preprocess_into(const cv::Mat &image,
                       const std::vector<int64_t> &input_tensor_shape,
                       std::size_t index);
  /// @brief Run exactly `images.size()` images as one session call.
  /// @param[in] images Images forming the batch (all same size after
  /// letterboxing, possibly different original sizes).
  /// @return One detection vector per image, in order.
  std::vector<std::vector<yolo_msgs::msg::Detection>>
  run_batch(const std::vector<cv::Mat> &images);
  /// @brief Run the ONNX Runtime session on the preprocessed input buffer.
  /// @param[in] input_tensor_shape Shape of the input tensor.
  /// @return The raw output tensors produced by the model.
  std::vector<Ort::Value> inference(std::vector<int64_t> &input_tensor_shape);
  /// @brief Decode the raw output tensors into detections. Overridden by each
  /// YOLO task; the base implementation is the identity.
  /// @param[in] original_image_size Size of the original camera image.
  /// @param[in] resized_image_size Size of the letterboxed network input.
  /// @param[in] preds Raw output tensors from inference().
  /// @return Detections in original-image coordinates.
  virtual std::vector<yolo_msgs::msg::Detection>
  postprocess(const cv::Size &original_image_size,
              const cv::Size &resized_image_size,
              const std::vector<Ort::Value> &preds);
  /// @brief Populate class_names from the ONNX graph metadata ("names"),
  /// falling back to the coco.names file when the model carries no vocabulary.
  void load_class_names();

  /// @brief ONNX Runtime environment.
  Ort::Env env{nullptr}; // ONNX Runtime environment
  /// @brief Session options for ONNX Runtime.
  Ort::SessionOptions session_options{
      nullptr}; // Session options for ONNX Runtime
  /// @brief ONNX Runtime inference session.
  Ort::Session session{nullptr}; // ONNX Runtime session for running inference
  /// @brief Primary execution provider that initialized the session. ORT may
  /// still fall back node-by-node to CUDA within a TensorRT session.
  std::string active_provider_;
  /// @brief Expected input image shape for the model.
  cv::Size input_image_shape; // Expected input image shape for the model

  // Persistent input buffer (CHW, 1*3*H*W floats) reused every inference to
  // avoid re-allocating + copying the full blob per frame.
  /// @brief Reusable CHW input blob (1*3*H*W floats).
  std::vector<float> input_buffer_;

  /// @brief Resolved input channel order (true = RGB, false = BGR).
  bool input_is_rgb_{false};

  // Input and output node names. The storage vectors are filled once and never
  // resized afterwards, so the const char* views in inputNames/outputNames stay
  // valid.
  /// @brief Owned input node name storage.
  std::vector<std::string> input_node_name_storage;
  /// @brief Input node names passed to the ONNX Runtime session.
  std::vector<const char *> inputNames;
  /// @brief Owned output node name storage.
  std::vector<std::string> output_node_name_storage;
  /// @brief Output node names requested from the ONNX Runtime session.
  std::vector<const char *> outputNames;

  /// @brief Number of input nodes in the model.
  size_t num_input_nodes;
  /// @brief Number of output nodes in the model.
  size_t num_output_nodes;
  /// @brief Memory information for ONNX Runtime tensor creation.
  Ort::MemoryInfo memory_info; // Memory information for ONNX Runtime

  /// @brief True when inference goes through CUDA graph replay.
  bool cuda_graph_{false};
#if !YOLO_ORT_LEGACY
  /// @brief CUDA device ordinal of the graph input buffer.
  int device_id_{0};
  /// @brief CUDA memory info for the device-resident input tensor.
  Ort::MemoryInfo cuda_memory_info_{nullptr};
  /// @brief Allocator owning the device-resident input tensor.
  Ort::Allocator cuda_allocator_{nullptr};
  /// @brief Persistent device input tensor bound to the session.
  Ort::Value device_input_{nullptr};
  /// @brief I/O binding (device input, CPU outputs) reused every inference.
  Ort::IoBinding io_binding_{nullptr};
#endif
};

/// @brief Slice one batch element out of a batched output tensor set.
///
/// The returned values are non-owning views into @p preds; they must not
/// outlive it. This lets the per-image postprocessors run unchanged on a
/// `{1, ...}` shape.
/// @param[in] preds Output tensors with a leading batch axis.
/// @param[in] batch_index Element to extract.
/// @param[in] memory_info Allocator description used to build the views.
/// @return One view per output tensor, shaped `{1, ...}`.
std::vector<Ort::Value>
slice_batch_outputs(const std::vector<Ort::Value> &preds,
                    std::size_t batch_index,
                    const Ort::MemoryInfo &memory_info);
} // namespace yolo_ros::engine
/// @}

#endif // YOLO_ROS__ENGINE__MODEL_HPP_
