// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include "yolo_ros/engine/provider.hpp"

#include "yolo_ros/utils/logs.hpp"
#include "yolo_ros/utils/string_utils.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace yolo_ros::engine {
namespace {

bool contains(const std::vector<Provider> &providers, Provider provider) {
  return std::find(providers.begin(), providers.end(), provider) !=
         providers.end();
}

std::string model_stem(const std::string &model_path) {
  return std::filesystem::path(model_path).stem().string();
}

/// @brief Short fingerprint of a model file: a hash of its absolute path plus
/// size and mtime when available, so updated or same-named models never reuse
/// a stale context cache.
std::string model_fingerprint(const std::string &model_path) {
  std::error_code path_error;
  const std::filesystem::path absolute =
      std::filesystem::absolute(model_path, path_error);
  std::string key =
      path_error ? model_path : absolute.lexically_normal().string();

  std::error_code size_error;
  const auto size = std::filesystem::file_size(model_path, size_error);

  if (!size_error) {
    key += ":" + std::to_string(size);
  }

  std::error_code time_error;
  const auto mtime = std::filesystem::last_write_time(model_path, time_error);

  if (!time_error) {
    key += ":" + std::to_string(
                     static_cast<long long>(mtime.time_since_epoch().count()));
  }

  // FNV-1a; a 12-hex-character suffix is plenty for cache namespacing.
  std::uint64_t hash = 14695981039346656037ULL;

  for (const char character : key) {
    hash ^= static_cast<unsigned char>(character);
    hash *= 1099511628211ULL;
  }

  char hex[17];
  std::snprintf(hex, sizeof(hex), "%016llx",
                static_cast<unsigned long long>(hash));

  return std::string(hex, 12);
}

struct TensorRtOptionsDeleter {
  void operator()(OrtTensorRTProviderOptionsV2 *options) const {
    if (options != nullptr) {
      Ort::GetApi().ReleaseTensorRTProviderOptions(options);
    }
  }
};

struct CudaOptionsDeleter {
  void operator()(OrtCUDAProviderOptionsV2 *options) const {
    if (options != nullptr) {
      Ort::GetApi().ReleaseCUDAProviderOptions(options);
    }
  }
};

} // namespace

std::vector<Provider> available_providers() {
  std::vector<Provider> providers;

  for (const std::string &name : Ort::GetAvailableProviders()) {
    const std::string lower = yolo_ros::utils::to_lower(name);

    if (lower == "cudaexecutionprovider") {
      providers.push_back(Provider::Cuda);
    } else if (lower == "tensorrtexecutionprovider") {
      providers.push_back(Provider::TensorRt);
    } else if (lower == "qnnexecutionprovider") {
      providers.push_back(Provider::Qnn);
    } else if (lower == "cpuexecutionprovider") {
      providers.push_back(Provider::Cpu);
    }
  }

  return providers;
}

std::vector<Provider> provider_chain(const std::string &requested,
                                     const std::vector<Provider> &available) {
  const std::string key =
      yolo_ros::utils::to_lower(requested.empty() ? "auto" : requested);
  std::vector<Provider> base;

  if (key == "cpu") {
    base = {Provider::Cpu};
  } else if (key == "qnn") {
    base = {Provider::Qnn, Provider::Cpu};
  } else if (key == "tensorrt" || key == "trt") {
    base = {Provider::TensorRt, Provider::Cuda, Provider::Cpu};
  } else {
    // "auto", "cuda" and unknown values prefer CUDA.
    base = {Provider::Cuda, Provider::Cpu};
  }

  std::vector<Provider> chain;

  for (const Provider provider : base) {
    if (provider == Provider::Cpu || contains(available, provider)) {
      chain.push_back(provider);
    }
  }

  return chain;
}

const char *provider_name(Provider provider) {
  switch (provider) {
  case Provider::Cpu:
    return "cpu";
  case Provider::Cuda:
    return "cuda";
  case Provider::TensorRt:
    return "tensorrt";
  case Provider::Qnn:
    return "qnn";
  }

  return "unknown";
}

int parse_device_id(const std::string &device) {
  const std::string value = yolo_ros::utils::to_lower(device);
  const std::size_t colon = value.rfind(':');
  const std::string ordinal =
      colon == std::string::npos ? value : value.substr(colon + 1);
  int id = 0;
  const char *begin = ordinal.data();
  const char *end = begin + ordinal.size();
  const auto result = std::from_chars(begin, end, id);

  if (result.ec != std::errc() || result.ptr != end || id < 0) {
    return 0;
  }

  return id;
}

const char *device_provider_name(const std::string &device) {
  const std::string value = yolo_ros::utils::to_lower(device);
  const std::size_t colon = value.find(':');
  const std::string prefix =
      colon == std::string::npos ? value : value.substr(0, colon);

  if (prefix == "cuda") {
    return "cuda";
  } else if (prefix == "tensorrt" || prefix == "trt") {
    return "tensorrt";
  } else if (prefix == "qnn") {
    return "qnn";
  } else if (prefix == "cpu") {
    return "cpu";
  }

  return "";
}

std::unordered_map<std::string, std::string>
qnn_provider_options(const ProviderConfig &config) {
  std::unordered_map<std::string, std::string> options = {
      {"soc_model", config.qnn_soc_model},
      {"htp_arch", config.qnn_htp_arch},
      {"htp_performance_mode", config.qnn_htp_performance_mode},
      {"offload_graph_io_quantization",
       config.qnn_offload_graph_io_quantization ? "1" : "0"},
      {"enable_htp_fp16_precision",
       config.qnn_enable_htp_fp16_precision ? "1" : "0"},
  };

  if (!config.qnn_backend_path.empty()) {
    options["backend_path"] = config.qnn_backend_path;
  }

  return options;
}

Ort::SessionOptions build_session_options(Provider primary,
                                          const ProviderConfig &config) {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  options.SetIntraOpNumThreads(primary == Provider::Cpu ? config.n_threads : 1);

  if (primary == Provider::Qnn) {
    if (config.qnn_context_cache_enable &&
        !config.qnn_context_cache_path.empty()) {
      options.AddConfigEntry("ep.context_enable", "1");
      options.AddConfigEntry("ep.context_file_path",
                             config.qnn_context_cache_path.c_str());
      options.AddConfigEntry("ep.context_embed_mode", "1");
    }

    options.AppendExecutionProvider("QNN", qnn_provider_options(config));
  }

  if (primary == Provider::TensorRt) {
    OrtTensorRTProviderOptionsV2 *raw_trt_options = nullptr;
    Ort::ThrowOnError(
        Ort::GetApi().CreateTensorRTProviderOptions(&raw_trt_options));
    std::unique_ptr<OrtTensorRTProviderOptionsV2, TensorRtOptionsDeleter>
        trt_options(raw_trt_options);

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
  }

  if (primary == Provider::TensorRt || primary == Provider::Cuda) {
    OrtCUDAProviderOptionsV2 *raw_cuda_options = nullptr;
    Ort::ThrowOnError(
        Ort::GetApi().CreateCUDAProviderOptions(&raw_cuda_options));
    std::unique_ptr<OrtCUDAProviderOptionsV2, CudaOptionsDeleter> cuda_options(
        raw_cuda_options);

    const std::string device_id = std::to_string(config.device_id);
    std::vector<const char *> keys = {"device_id", "arena_extend_strategy",
                                      "cudnn_conv_algo_search",
                                      "cudnn_conv_use_max_workspace"};
    std::vector<const char *> values = {device_id.c_str(), "kSameAsRequested",
                                        "EXHAUSTIVE", "1"};

    if (primary == Provider::Cuda && config.cuda_graph_enable) {
      keys.push_back("enable_cuda_graph");
      values.push_back("1");
    }

    Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(
        cuda_options.get(), keys.data(), values.data(), keys.size()));
    options.AppendExecutionProvider_CUDA_V2(*cuda_options);
  }

  return options;
}

std::string engine_cache_dir(const std::string &base,
                             const std::string &model_path) {
  std::string root = base;

  if (root.empty()) {
    const char *home = std::getenv("HOME");

    if (home == nullptr) {
      YOLO_LOG_ERROR("Cannot resolve $HOME for the TensorRT engine cache.");
      return "";
    }

    root = std::string(home) + "/.cache/yolo_ros/trt_engines";
  }

  const std::filesystem::path dir =
      std::filesystem::path(root) / model_stem(model_path);
  std::error_code error;
  std::filesystem::create_directories(dir, error);

  if (error) {
    YOLO_LOG_ERROR("Cannot create the TensorRT engine cache directory %s: %s",
                   dir.string().c_str(), error.message().c_str());
    return "";
  }

  return dir.string();
}

std::string qnn_context_path(const std::string &base,
                             const std::string &model_path) {
  std::string root = base;

  if (root.empty()) {
    const char *home = std::getenv("HOME");

    if (home == nullptr) {
      YOLO_LOG_ERROR("Cannot resolve $HOME for the QNN context cache.");
      return "";
    }

    root = std::string(home) + "/.cache/yolo_ros/qnn_context";
  }

  std::error_code error;
  std::filesystem::create_directories(root, error);

  if (error) {
    YOLO_LOG_ERROR("Cannot create the QNN context cache directory %s: %s",
                   root.c_str(), error.message().c_str());
    return "";
  }

  return (std::filesystem::path(root) /
          (model_stem(model_path) + "-" + model_fingerprint(model_path) +
           ".ctx.onnx"))
      .string();
}

QnnModelPath resolve_qnn_model_path(const std::string &cache_base,
                                    const std::string &model_path,
                                    bool cache_enable) {
  QnnModelPath resolved;
  resolved.model_path = model_path;

  // Already an EPContext model: load it directly, never compile a cache.
  const std::string context_suffix = ".ctx.onnx";

  if (model_path.size() >= context_suffix.size() &&
      model_path.compare(model_path.size() - context_suffix.size(),
                         context_suffix.size(), context_suffix) == 0) {
    return resolved;
  }

  if (!cache_enable) {
    return resolved;
  }

  const std::string context_path = qnn_context_path(cache_base, model_path);

  if (context_path.empty()) {
    return resolved;
  }

  resolved.context_path = context_path;

  std::error_code error;
  const bool cache_exists = std::filesystem::exists(context_path, error);

  if (cache_exists && !error) {
    resolved.model_path = context_path;
  } else if (error) {
    // A stat failure means the cache is unusable: load the original model and
    // let the QNN EP compile in-memory for this session.
    resolved.context_path.clear();
    resolved.generate = false;
  } else {
    resolved.generate = true;
  }

  return resolved;
}

} // namespace yolo_ros::engine
