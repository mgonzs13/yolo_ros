// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include "yolo_ros/engine/provider.hpp"
#include "yolo_ros/engine/ort_compat.hpp"

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
    compat::append_tensorrt_provider(options, config);
  }

  if (primary == Provider::TensorRt || primary == Provider::Cuda) {
    compat::append_cuda_provider(options, config);
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
