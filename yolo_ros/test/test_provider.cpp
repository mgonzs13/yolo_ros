// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "yolo_ros/engine/provider.hpp"

namespace yolo_ros::engine {
namespace {

const std::vector<Provider> kAll = {Provider::TensorRt, Provider::Cuda,
                                    Provider::Qnn, Provider::Cpu};

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

TEST(ProviderChain, QnnFallsBackToCpu) {
  EXPECT_EQ(provider_chain("qnn", kAll),
            (std::vector<Provider>{Provider::Qnn, Provider::Cpu}));
}

TEST(ProviderChain, QnnFallsBackToCpuWhenUnavailable) {
  EXPECT_EQ(provider_chain("qnn", {Provider::Cuda, Provider::Cpu}),
            (std::vector<Provider>{Provider::Cpu}));
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
  EXPECT_STREQ(provider_name(Provider::Qnn), "qnn");
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
  EXPECT_STREQ(device_provider_name("qnn:0"), "qnn");
  EXPECT_STREQ(device_provider_name("QNN"), "qnn");
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

TEST(QnnContextPath, CreatesModelContextFile) {
  const std::filesystem::path base =
      std::filesystem::temp_directory_path() /
      ("yolo_qnn_cache_test_" + std::to_string(::getpid()));
  std::filesystem::remove_all(base);
  const std::string path =
      qnn_context_path(base.string(), "/models/yolo26m.onnx");
  const std::string filename = std::filesystem::path(path).filename().string();
  EXPECT_EQ(std::filesystem::path(path).parent_path(), base);
  EXPECT_EQ(filename.rfind("yolo26m-", 0), 0u);
  EXPECT_EQ(filename.find(".ctx.onnx"), filename.size() - 9);
  EXPECT_TRUE(std::filesystem::is_directory(base));
  EXPECT_FALSE(std::filesystem::exists(path));
  std::filesystem::remove_all(base);
}

TEST(QnnContextPath, EmptyBaseUsesHome) {
  const char *original_home = std::getenv("HOME");
  const std::string saved_home = original_home == nullptr ? "" : original_home;

  const std::filesystem::path fake_home =
      std::filesystem::temp_directory_path() /
      ("yolo_qnn_home_test_" + std::to_string(::getpid()));
  std::filesystem::remove_all(fake_home);
  std::filesystem::create_directories(fake_home);
  ASSERT_EQ(::setenv("HOME", fake_home.string().c_str(), 1), 0);

  const std::string path = qnn_context_path("", "/models/yolo26m.onnx");

  if (original_home == nullptr) {
    ::unsetenv("HOME");
  } else {
    ::setenv("HOME", saved_home.c_str(), 1);
  }

  ASSERT_FALSE(path.empty());
  EXPECT_NE(path.find("qnn_context"), std::string::npos);
  const std::string filename = std::filesystem::path(path).filename().string();
  EXPECT_EQ(filename.rfind("yolo26m-", 0), 0u);
  EXPECT_EQ(filename.find(".ctx.onnx"), filename.size() - 9);
  EXPECT_EQ(path.find(fake_home.string()), 0u);
  std::filesystem::remove_all(fake_home);
}

TEST(ResolveQnnModelPath, DisabledKeepsModel) {
  const QnnModelPath resolved =
      resolve_qnn_model_path("/tmp/never", "/models/yolo26m.onnx", false);
  EXPECT_EQ(resolved.model_path, "/models/yolo26m.onnx");
  EXPECT_TRUE(resolved.context_path.empty());
  EXPECT_FALSE(resolved.generate);
}

TEST(ResolveQnnModelPath, MissingCacheGenerates) {
  const std::filesystem::path base =
      std::filesystem::temp_directory_path() /
      ("yolo_qnn_resolve_test_" + std::to_string(::getpid()));
  std::filesystem::remove_all(base);
  const QnnModelPath resolved =
      resolve_qnn_model_path(base.string(), "/models/yolo26m.onnx", true);
  EXPECT_EQ(resolved.model_path, "/models/yolo26m.onnx");
  EXPECT_EQ(std::filesystem::path(resolved.context_path).parent_path(), base);
  const std::string filename =
      std::filesystem::path(resolved.context_path).filename().string();
  EXPECT_EQ(filename.rfind("yolo26m-", 0), 0u);
  EXPECT_EQ(filename.find(".ctx.onnx"), filename.size() - 9);
  EXPECT_TRUE(resolved.generate);
  std::filesystem::remove_all(base);
}

TEST(ResolveQnnModelPath, ExistingCacheLoadsContext) {
  const std::filesystem::path base =
      std::filesystem::temp_directory_path() /
      ("yolo_qnn_resolve_hit_test_" + std::to_string(::getpid()));
  std::filesystem::remove_all(base);
  const std::string cache_path =
      qnn_context_path(base.string(), "/models/yolo26m.onnx");
  ASSERT_FALSE(cache_path.empty());
  {
    std::ofstream touch(cache_path);
    touch << "ctx";
  }
  const QnnModelPath resolved =
      resolve_qnn_model_path(base.string(), "/models/yolo26m.onnx", true);
  EXPECT_EQ(resolved.model_path, cache_path);
  EXPECT_EQ(resolved.context_path, cache_path);
  EXPECT_FALSE(resolved.generate);
  std::filesystem::remove_all(base);
}

TEST(QnnContextPath, SameStemDifferentModelsGetDifferentCaches) {
  const std::filesystem::path base =
      std::filesystem::temp_directory_path() /
      ("yolo_qnn_fingerprint_test_" + std::to_string(::getpid()));
  const std::filesystem::path model_a = base / "a";
  const std::filesystem::path model_b = base / "b";
  std::filesystem::remove_all(base);
  std::filesystem::create_directories(model_a);
  std::filesystem::create_directories(model_b);
  {
    std::ofstream a(model_a / "yolo26m.onnx");
    a << "aaa";
  }
  {
    std::ofstream b(model_b / "yolo26m.onnx");
    b << "bbbbbb";
  }
  const std::string path_a =
      qnn_context_path(base.string(), (model_a / "yolo26m.onnx").string());
  const std::string path_b =
      qnn_context_path(base.string(), (model_b / "yolo26m.onnx").string());
  EXPECT_NE(path_a, path_b);
  EXPECT_EQ(path_a, qnn_context_path(base.string(),
                                     (model_a / "yolo26m.onnx").string()));
  std::filesystem::remove_all(base);
}

TEST(ResolveQnnModelPath, PrecompiledContextKeepsModel) {
  const std::filesystem::path base =
      std::filesystem::temp_directory_path() /
      ("yolo_qnn_resolve_ctx_test_" + std::to_string(::getpid()));
  std::filesystem::remove_all(base);
  const std::string model = "/models/yolo26m.ctx.onnx";
  const QnnModelPath resolved =
      resolve_qnn_model_path(base.string(), model, true);
  EXPECT_EQ(resolved.model_path, model);
  EXPECT_TRUE(resolved.context_path.empty());
  EXPECT_FALSE(resolved.generate);
  EXPECT_FALSE(std::filesystem::exists(base)); // no cache dir was touched
  std::filesystem::remove_all(base);
}

TEST(QnnProviderOptions, DefaultsMatchDragonwing) {
  ProviderConfig config;
  const auto options = qnn_provider_options(config);
  EXPECT_EQ(options.at("backend_path"), "libQnnHtp.so");
  EXPECT_EQ(options.at("soc_model"), "82");
  EXPECT_EQ(options.at("htp_arch"), "75");
  EXPECT_EQ(options.at("htp_performance_mode"), "burst");
  EXPECT_EQ(options.at("offload_graph_io_quantization"), "1");
  EXPECT_EQ(options.at("enable_htp_fp16_precision"), "0");
}

TEST(QnnProviderOptions, HonorsOverrides) {
  ProviderConfig config;
  config.qnn_backend_path = "/opt/qairt/lib/libQnnHtp.so";
  config.qnn_htp_performance_mode = "sustained_high_performance";
  config.qnn_offload_graph_io_quantization = false;
  config.qnn_enable_htp_fp16_precision = true;
  const auto options = qnn_provider_options(config);
  EXPECT_EQ(options.at("backend_path"), "/opt/qairt/lib/libQnnHtp.so");
  EXPECT_EQ(options.at("htp_performance_mode"), "sustained_high_performance");
  EXPECT_EQ(options.at("offload_graph_io_quantization"), "0");
  EXPECT_EQ(options.at("enable_htp_fp16_precision"), "1");
}

TEST(QnnProviderOptions, EmptyBackendPathOmitsKey) {
  ProviderConfig config;
  config.qnn_backend_path.clear();
  const auto options = qnn_provider_options(config);
  EXPECT_EQ(options.find("backend_path"), options.end());
}

TEST(BuildSessionOptions, QnnProviderWhenAvailableDoesNotThrow) {
  // On a QNN-enabled ONNX Runtime build the options must append cleanly; on
  // the stock CPU/GPU builds the provider list has no QNN and the test skips.
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test_provider_qnn");
  const std::vector<Provider> available = available_providers();
  if (std::find(available.begin(), available.end(), Provider::Qnn) ==
      available.end()) {
    GTEST_SKIP() << "ONNX Runtime build has no QNN execution provider";
  }
  ProviderConfig config;
  config.n_threads = 1;
  EXPECT_NO_THROW({
    auto options = build_session_options(Provider::Qnn, config);
    (void)options;
  });
}

} // namespace
} // namespace yolo_ros::engine
