// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2018 Kaiyang Zhou
// SPDX-License-Identifier: MIT

/// @file
/// @brief ONNX Runtime ReID embedding extractor for appearance association.

#ifndef YOLO_ROS__ENGINE__REID_ENCODER_HPP_
#define YOLO_ROS__ENGINE__REID_ENCODER_HPP_

#include <array>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

/// @addtogroup yolo_engine
/// @{
namespace yolo_ros::engine {

/// @brief Batched ReID embedding extractor.
///
/// Input contract: a batched NCHW RGB float32 tensor in [0, 255]; the model
/// bakes its own normalization and L2 normalization into the graph. The input
/// height/width are read from the session, so any ReID export (OSNet,
/// FastReID) works unchanged.
class ReIDEncoder {
public:
  /// @brief Load @p model_path and create the ONNX Runtime session.
  /// @param[in] model_path ONNX ReID model path.
  /// @param[in] provider "auto" | "cuda" | "cpu" (case-insensitive).
  /// @param[in] device Ultralytics-style device string; the ordinal is parsed.
  /// @throws std::runtime_error when no execution provider can create the
  /// session (missing model, unavailable provider, ...).
  ReIDEncoder(const std::string &model_path, const std::string &provider,
              const std::string &device);
  /// @brief Destroy the encoder and release the session.
  ~ReIDEncoder();

  /// @brief Fill @p out with the NCHW RGB [0, 255] batch for @p boxes.
  ///
  /// Static and buffer-injected so it is unit-testable without a session.
  /// Boxes are tlbr in frame pixels and are clamped to @p frame; degenerate
  /// boxes leave a zero patch.
  /// @param[in] frame BGR frame.
  /// @param[in] boxes Detection boxes (tlbr) to crop.
  /// @param[in] width Model input width.
  /// @param[in] height Model input height.
  /// @param[out] out Resized batch buffer, overwritten.
  static void make_batch(const cv::Mat &frame,
                         const std::vector<std::array<float, 4>> &boxes,
                         int width, int height, std::vector<float> &out);

  /// @brief Encode @p boxes (tlbr, frame pixels) in one batched session call.
  /// @param[in] frame BGR frame.
  /// @param[in] boxes Detection boxes (tlbr); clamped to the frame.
  /// @return One L2-normalized feature per box; empty when @p boxes (or the
  /// frame) is empty.
  std::vector<std::vector<float>>
  inference(const cv::Mat &frame,
            const std::vector<std::array<float, 4>> &boxes);

private:
  /// @brief ONNX Runtime environment.
  Ort::Env env_{nullptr};
  /// @brief Session options for ONNX Runtime.
  Ort::SessionOptions session_options_{nullptr};
  /// @brief ONNX Runtime session.
  Ort::Session session_{nullptr};
  /// @brief Memory information for tensor creation.
  Ort::MemoryInfo memory_info_{nullptr};
  /// @brief Primary execution provider that accepted the session.
  std::string active_provider_;
  /// @brief Model input width.
  int input_width_ = 0;
  /// @brief Model input height.
  int input_height_ = 0;
  /// @brief Storage backing input_names_.
  std::vector<Ort::AllocatedStringPtr> input_name_alloc_;
  /// @brief Input node names.
  std::vector<const char *> input_names_;
  /// @brief Storage backing output_names_.
  std::vector<Ort::AllocatedStringPtr> output_name_alloc_;
  /// @brief Output node names.
  std::vector<const char *> output_names_;
  /// @brief Reusable input batch buffer.
  std::vector<float> blob_;
};

} // namespace yolo_ros::engine
/// @}

#endif // YOLO_ROS__ENGINE__REID_ENCODER_HPP_
