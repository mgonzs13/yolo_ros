// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "yolo_ros/engine/ort_compat.hpp"
#include "yolo_ros/engine/provider.hpp"

namespace yolo_ros::engine {
namespace {

const std::vector<Provider> kAll = {Provider::TensorRt, Provider::Cuda,
                                    Provider::Cpu};

TEST(ProviderChain, CpuOnly) {
  EXPECT_EQ(provider_chain("cpu", kAll),
            (std::vector<Provider>{Provider::Cpu}));
}

TEST(ProviderChain, CudaFallsBackToCpu) {
  EXPECT_EQ(provider_chain("cuda", kAll),
            (std::vector<Provider>{Provider::Cuda, Provider::Cpu}));
}

TEST(ProviderChain, TensorRtFallsBackToCudaThenCpu) {
  EXPECT_EQ(provider_chain("tensorrt", kAll),
            (std::vector<Provider>{Provider::TensorRt, Provider::Cuda,
                                   Provider::Cpu}));
  EXPECT_EQ(provider_chain("trt", kAll),
            (std::vector<Provider>{Provider::TensorRt, Provider::Cuda,
                                   Provider::Cpu}));
}

TEST(ProviderChain, AutoPrefersCuda) {
  EXPECT_EQ(provider_chain("auto", kAll),
            (std::vector<Provider>{Provider::Cuda, Provider::Cpu}));
}

TEST(ProviderChain, UnknownAndEmptyFallBackToAuto) {
  EXPECT_EQ(provider_chain("bogus", kAll),
            (std::vector<Provider>{Provider::Cuda, Provider::Cpu}));
  EXPECT_EQ(provider_chain("", kAll),
            (std::vector<Provider>{Provider::Cuda, Provider::Cpu}));
}

TEST(ProviderChain, CaseInsensitive) {
  EXPECT_EQ(provider_chain("CUDA", kAll),
            (std::vector<Provider>{Provider::Cuda, Provider::Cpu}));
}

TEST(ProviderChain, FiltersUnavailableProviders) {
  EXPECT_EQ(provider_chain("auto", {Provider::Cuda, Provider::Cpu}),
            (std::vector<Provider>{Provider::Cuda, Provider::Cpu}));
  EXPECT_EQ(provider_chain("auto", {Provider::Cpu}),
            (std::vector<Provider>{Provider::Cpu}));
  EXPECT_EQ(provider_chain("tensorrt", {Provider::Cpu}),
            (std::vector<Provider>{Provider::Cpu}));
  EXPECT_EQ(provider_chain("cuda", {Provider::Cpu}),
            (std::vector<Provider>{Provider::Cpu}));
}

TEST(ProviderName, NamesAllProviders) {
  EXPECT_STREQ(provider_name(Provider::Cpu), "cpu");
  EXPECT_STREQ(provider_name(Provider::Cuda), "cuda");
  EXPECT_STREQ(provider_name(Provider::TensorRt), "tensorrt");
}

TEST(ParseDeviceId, ParsesOrdinals) {
  EXPECT_EQ(parse_device_id("cuda:0"), 0);
  EXPECT_EQ(parse_device_id("trt:1"), 1);
  EXPECT_EQ(parse_device_id("tensorrt:2"), 2);
  EXPECT_EQ(parse_device_id("0"), 0);
  EXPECT_EQ(parse_device_id("3"), 3);
}

TEST(ParseDeviceId, DefaultsToZero) {
  EXPECT_EQ(parse_device_id("cuda"), 0);
  EXPECT_EQ(parse_device_id("cpu"), 0);
  EXPECT_EQ(parse_device_id(""), 0);
  EXPECT_EQ(parse_device_id("cuda:x"), 0);
  EXPECT_EQ(parse_device_id("cuda:-1"), 0);
}

TEST(ParseDeviceId, RejectsTrailingGarbage) {
  EXPECT_EQ(parse_device_id("cuda:1.5"), 0);
  EXPECT_EQ(parse_device_id("cuda:2abc"), 0);
  EXPECT_EQ(parse_device_id("cuda: 1"), 0);
}

TEST(DeviceProviderName, MapsPrefixes) {
  EXPECT_STREQ(device_provider_name("cuda:0"), "cuda");
  EXPECT_STREQ(device_provider_name("cuda"), "cuda");
  EXPECT_STREQ(device_provider_name("CUDA:1"), "cuda");
  EXPECT_STREQ(device_provider_name("trt:1"), "tensorrt");
  EXPECT_STREQ(device_provider_name("tensorrt:2"), "tensorrt");
  EXPECT_STREQ(device_provider_name("cpu"), "cpu");
  EXPECT_STREQ(device_provider_name("0"), "");
  EXPECT_STREQ(device_provider_name("1"), "");
  EXPECT_STREQ(device_provider_name(""), "");
}

TEST(AvailableProviders, AlwaysContainsCpu) {
  const std::vector<Provider> available = available_providers();
  EXPECT_NE(std::find(available.begin(), available.end(), Provider::Cpu),
            available.end());
}

TEST(EngineCacheDir, CreatesModelSubdirectory) {
  const std::filesystem::path base =
      std::filesystem::temp_directory_path() /
      ("yolo_trt_cache_test_" + std::to_string(::getpid()));
  std::filesystem::remove_all(base);
  const std::string dir =
      engine_cache_dir(base.string(), "/models/yolo26m.onnx");
  EXPECT_EQ(dir, (base / "yolo26m").string());
  EXPECT_TRUE(std::filesystem::is_directory(dir));
  std::filesystem::remove_all(base);
}

TEST(EngineCacheDir, EmptyBaseUsesHome) {
  const char *original_home = std::getenv("HOME");
  const std::string saved_home = original_home == nullptr ? "" : original_home;

  const std::filesystem::path fake_home =
      std::filesystem::temp_directory_path() /
      ("yolo_trt_home_test_" + std::to_string(::getpid()));
  std::filesystem::remove_all(fake_home);
  std::filesystem::create_directories(fake_home);
  ASSERT_EQ(::setenv("HOME", fake_home.string().c_str(), 1), 0);

  const std::string dir = engine_cache_dir("", "/models/yolo26m.onnx");

  if (original_home == nullptr) {
    ::unsetenv("HOME");
  } else {
    ::setenv("HOME", saved_home.c_str(), 1);
  }

  ASSERT_FALSE(dir.empty());
  ASSERT_GE(dir.size(), 7u);
  EXPECT_NE(dir.find("trt_engines"), std::string::npos);
  EXPECT_EQ(dir.compare(dir.size() - 7, 7, "yolo26m"), 0);
  EXPECT_EQ(dir.find(fake_home.string()), 0u);
  EXPECT_TRUE(std::filesystem::is_directory(dir));

  std::filesystem::remove_all(fake_home);
}

TEST(BuildSessionOptions, CpuProviderDoesNotThrow) {
  ProviderConfig config;
  config.n_threads = 1;
  EXPECT_NO_THROW({
    auto options = build_session_options(Provider::Cpu, config);
    (void)options;
  });
}

TEST(BuildSessionOptions, AvailableGpuProvidersDoNotThrow) {
  // TensorRT appends log through ORT's default logger, which is only
  // registered once an Ort::Env exists (the engine keeps one alive before
  // building session options).
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test_provider");
  const std::vector<Provider> available = available_providers();
  ProviderConfig config;
  config.n_threads = 1;
  config.trt_engine_cache_enable = false; // do not write anything to disk

  if (std::find(available.begin(), available.end(), Provider::Cuda) !=
      available.end()) {
    EXPECT_NO_THROW({
      auto options = build_session_options(Provider::Cuda, config);
      (void)options;
    });
  }

  if (std::find(available.begin(), available.end(), Provider::TensorRt) !=
      available.end()) {
    EXPECT_NO_THROW({
      auto options = build_session_options(Provider::TensorRt, config);
      (void)options;
    });
  }
}

#if YOLO_ORT_LEGACY
TEST(BuildSessionOptions, LegacyTensorRtRefusal) {
  ProviderConfig config;
  config.n_threads = 1;
  EXPECT_THROW(build_session_options(Provider::TensorRt, config),
               std::runtime_error);
}
#endif

} // namespace
} // namespace yolo_ros::engine
