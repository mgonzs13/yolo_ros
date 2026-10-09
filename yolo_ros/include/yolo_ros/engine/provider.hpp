// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

/// @file
/// @brief Execution-provider selection and ONNX Runtime session-option
/// building for the YOLO engine (CPU / CUDA / TensorRT with fallback).

#ifndef YOLO_ROS__ENGINE__PROVIDER_HPP_
#define YOLO_ROS__ENGINE__PROVIDER_HPP_

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

/// @addtogroup yolo_engine
/// @{
namespace yolo_ros::engine {
/// @brief Inference execution provider.
enum class Provider { Cpu, Cuda, TensorRt };

/// @brief Configuration used to build ONNX Runtime session options.
struct ProviderConfig {
  /// @brief CUDA/TensorRT device ordinal.
  int device_id{0};
  /// @brief Intra-op thread count for the CPU EP (GPU primaries use 1).
  int n_threads{0};
  /// @brief TensorRT: enable FP16 precision.
  bool trt_fp16_enable{true};
  /// @brief TensorRT: persist the built engine across sessions.
  bool trt_engine_cache_enable{true};
  /// @brief TensorRT engine cache directory (empty disables caching).
  std::string trt_engine_cache_path;
  /// @brief CUDA primary: capture the model as a CUDA graph (fixed shapes).
  bool cuda_graph_enable{false};
};

/// @brief Execution providers available in the linked ONNX Runtime build.
/// @return The available providers, in ONNX Runtime's reported order.
std::vector<Provider> available_providers();

/// @brief Resolve @p requested into an ordered, availability-filtered fallback
/// chain. Unknown values are treated as "auto" (CUDA -> CPU).
/// @param[in] requested "auto" (CUDA -> CPU), "cuda" (CUDA -> CPU),
/// "tensorrt"/"trt" (TensorRT -> CUDA -> CPU) or "cpu" (case-insensitive).
/// @param[in] available Providers present in the ONNX Runtime build.
/// @return The ordered chain; CPU is always kept.
std::vector<Provider> provider_chain(const std::string &requested,
                                     const std::vector<Provider> &available);

/// @brief Lower-case name of @p provider.
/// @param[in] provider Provider to name.
/// @return "cpu", "cuda" or "tensorrt".
const char *provider_name(Provider provider);

/// @brief Parse the device ordinal from an ultralytics-style `device` string.
/// @param[in] device e.g. "cuda:0", "trt:1", "1", "cpu".
/// @return The ordinal, or 0 when absent, negative or invalid.
int parse_device_id(const std::string &device);

/// @brief Provider named by the prefix of a device string ("cuda", "tensorrt",
/// "trt" or "cpu"); empty when the device names no provider (e.g. "0", "1").
/// @param[in] device Ultralytics-style device string.
/// @return "cpu", "cuda", "tensorrt", or "" when absent/unrecognized.
const char *device_provider_name(const std::string &device);

/// @brief Build session options for @p primary, appending the execution
/// provider(s). A TensorRT primary also appends CUDA as node-level fallback.
/// @param[in] primary Provider to append.
/// @param[in] config Provider configuration; @c n_threads must already be
/// resolved (no -1 sentinel).
/// @return The configured session options.
/// @throws Ort::Exception if the provider cannot be appended (modern builds).
/// @throws std::runtime_error on legacy builds (ONNX Runtime < 1.12) when
/// TensorRT is requested: the legacy TensorRT refusal falls through the
/// provider chain to CUDA/CPU.
Ort::SessionOptions build_session_options(Provider primary,
                                          const ProviderConfig &config);

/// @brief TensorRT engine cache directory for @p model_path, created
/// recursively.
/// @param[in] base Configured cache base (empty ->
/// ~/.cache/yolo_ros/trt_engines).
/// @param[in] model_path Model path; its stem is appended to @p base.
/// @return The created directory, or "" if it could not be created.
std::string engine_cache_dir(const std::string &base,
                             const std::string &model_path);
} // namespace yolo_ros::engine
/// @}

#endif // YOLO_ROS__ENGINE__PROVIDER_HPP_
