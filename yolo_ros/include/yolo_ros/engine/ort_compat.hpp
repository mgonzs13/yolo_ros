// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#ifndef YOLO_ROS__ENGINE__ORT_COMPAT_HPP_
#define YOLO_ROS__ENGINE__ORT_COMPAT_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "yolo_ros/engine/provider.hpp"

// ONNX Runtime < 1.12 (the CUDA 10 / legacy path) lacks the V2 provider-option
// APIs and the allocated string/metadata APIs. Everything else in the engine
// compiles unchanged against both generations.
#if ORT_API_VERSION < 12
#define YOLO_ORT_LEGACY 1
#else
#define YOLO_ORT_LEGACY 0
#endif

namespace yolo_ros::engine::compat {

/// @brief Input node names of a session, in session order.
inline std::vector<std::string> input_names(const Ort::Session &session) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> names;
  names.reserve(session.GetInputCount());

  for (std::size_t i = 0; i < session.GetInputCount(); ++i) {
#if YOLO_ORT_LEGACY
    char *name = session.GetInputName(i, allocator);
    names.emplace_back(name == nullptr ? "" : name);
    allocator.Free(name);
#else
    names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
#endif
  }

  return names;
}

/// @brief Output node names of a session, in session order.
inline std::vector<std::string> output_names(const Ort::Session &session) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> names;
  names.reserve(session.GetOutputCount());

  for (std::size_t i = 0; i < session.GetOutputCount(); ++i) {
#if YOLO_ORT_LEGACY
    char *name = session.GetOutputName(i, allocator);
    names.emplace_back(name == nullptr ? "" : name);
    allocator.Free(name);
#else
    names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
#endif
  }

  return names;
}

/// @brief Look up an ONNX custom metadata value; nullopt when absent.
inline std::optional<std::string>
lookup_metadata(const Ort::ModelMetadata &metadata, const char *key) {
  Ort::AllocatorWithDefaultOptions allocator;
#if YOLO_ORT_LEGACY
  char *value = metadata.LookupCustomMetadataMap(key, allocator);

  if (value == nullptr) {
    return std::nullopt;
  }

  std::string result(value);
  allocator.Free(value);
  return result;
#else
  auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);

  if (!value) {
    return std::nullopt;
  }

  return std::string(value.get());
#endif
}

/// @brief Keys of the ONNX custom metadata map.
inline std::vector<std::string>
metadata_keys(const Ort::ModelMetadata &metadata) {
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> keys;
#if YOLO_ORT_LEGACY
  int64_t count = 0;
  char **raw = metadata.GetCustomMetadataMapKeys(allocator, count);

  if (raw == nullptr) {
    return keys;
  }

  keys.reserve(static_cast<std::size_t>(count));

  for (int64_t i = 0; i < count; ++i) {
    keys.emplace_back(raw[i] == nullptr ? "" : raw[i]);
    allocator.Free(raw[i]);
  }

  allocator.Free(raw);
#else
  for (const auto &key :
       metadata.GetCustomMetadataMapKeysAllocated(allocator)) {
    keys.emplace_back(key.get());
  }
#endif
  return keys;
}

/// @brief ONNX Runtime version string (e.g. "1.6.0").
inline std::string version_string() {
#if YOLO_ORT_LEGACY
  return OrtGetApiBase()->GetVersionString();
#else
  return Ort::GetVersionString();
#endif
}

/// @brief Append the CUDA execution provider with the options supported by the
/// linked ONNX Runtime generation.
inline void append_cuda_provider(Ort::SessionOptions &options,
                                 const ProviderConfig &config) {
#if YOLO_ORT_LEGACY
  OrtCUDAProviderOptions legacy{};
  legacy.device_id = config.device_id;
  legacy.cuda_mem_limit = std::numeric_limits<std::size_t>::max();
  legacy.arena_extend_strategy = 1; // kSameAsRequested
  legacy.cudnn_conv_algo_search = EXHAUSTIVE;
  legacy.do_copy_in_default_stream = 1;
  options.AppendExecutionProvider_CUDA(legacy);
#else
  OrtCUDAProviderOptionsV2 *raw_cuda_options = nullptr;
  Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&raw_cuda_options));
  const auto cuda_deleter = [](OrtCUDAProviderOptionsV2 *ptr) {
    if (ptr != nullptr) {
      Ort::GetApi().ReleaseCUDAProviderOptions(ptr);
    }
  };
  std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(cuda_deleter)>
      cuda_options(raw_cuda_options, cuda_deleter);

  const std::string device_id = std::to_string(config.device_id);
  std::vector<const char *> keys = {"device_id", "arena_extend_strategy",
                                    "cudnn_conv_algo_search",
                                    "cudnn_conv_use_max_workspace"};
  std::vector<const char *> values = {device_id.c_str(), "kSameAsRequested",
                                      "EXHAUSTIVE", "1"};

  if (config.cuda_graph_enable) {
    keys.push_back("enable_cuda_graph");
    values.push_back("1");
  }

  Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(
      cuda_options.get(), keys.data(), values.data(), keys.size()));
  options.AppendExecutionProvider_CUDA_V2(*cuda_options);
#endif
}

/// @brief Append the TensorRT execution provider. Legacy ONNX Runtime builds
/// (< 1.12) cannot carry the fp16/cache options this project needs, so the
/// refusal throws and the provider chain falls through to CUDA.
inline void append_tensorrt_provider(Ort::SessionOptions &options,
                                     const ProviderConfig &config) {
#if YOLO_ORT_LEGACY
  (void)options;
  (void)config;
  throw std::runtime_error(
      "TensorRT EP is not supported with ONNX Runtime < 1.12; use provider: "
      "cuda");
#else
  OrtTensorRTProviderOptionsV2 *raw_trt_options = nullptr;
  Ort::ThrowOnError(
      Ort::GetApi().CreateTensorRTProviderOptions(&raw_trt_options));
  const auto trt_deleter = [](OrtTensorRTProviderOptionsV2 *ptr) {
    if (ptr != nullptr) {
      Ort::GetApi().ReleaseTensorRTProviderOptions(ptr);
    }
  };
  std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(trt_deleter)>
      trt_options(raw_trt_options, trt_deleter);

  const std::string device_id = std::to_string(config.device_id);
  const std::string fp16 = config.trt_fp16_enable ? "1" : "0";
  const bool cache_enabled =
      config.trt_engine_cache_enable && !config.trt_engine_cache_path.empty();
  const std::string cache_enable = cache_enabled ? "1" : "0";
  std::vector<const char *> keys = {"device_id", "trt_fp16_enable",
                                    "trt_engine_cache_enable"};
  std::vector<const char *> values = {device_id.c_str(), fp16.c_str(),
                                      cache_enable.c_str()};

  if (cache_enabled) {
    keys.push_back("trt_engine_cache_path");
    values.push_back(config.trt_engine_cache_path.c_str());
  }

  Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptions(
      trt_options.get(), keys.data(), values.data(), keys.size()));
  options.AppendExecutionProvider_TensorRT_V2(*trt_options);
#endif
}

} // namespace yolo_ros::engine::compat

#endif // YOLO_ROS__ENGINE__ORT_COMPAT_HPP_
