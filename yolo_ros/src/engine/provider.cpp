// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include "yolo_ros/engine/provider.hpp"

#include "yolo_ros/utils/logs.hpp"
#include "yolo_ros/utils/string_utils.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
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
  } else if (prefix == "cpu") {
    return "cpu";
  }

  return "";
}

Ort::SessionOptions build_session_options(Provider primary,
                                          const ProviderConfig &config) {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  options.SetIntraOpNumThreads(primary == Provider::Cpu ? config.n_threads : 1);

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
                                      "cudnn_conv_algo_search"};
    std::vector<const char *> values = {device_id.c_str(), "kSameAsRequested",
                                        "HEURISTIC"};
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

} // namespace yolo_ros::engine
