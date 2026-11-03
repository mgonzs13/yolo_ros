// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "yolo_ros/engine/ort_compat.hpp"

using yolo_ros::engine::ProviderConfig;
namespace compat = yolo_ros::engine::compat;

#if YOLO_ORT_LEGACY && YOLO_ORT_LEGACY_HAS_TENSORRT

static int g_trt_append_calls = 0;
static int g_trt_append_device_id = -1;

extern "C" OrtStatus *OrtSessionOptionsAppendExecutionProvider_Tensorrt(
    OrtSessionOptions *, int device_id) noexcept {
  ++g_trt_append_calls;
  g_trt_append_device_id = device_id;
  return nullptr; // success
}

TEST(OrtCompatLegacyTrt, AppendsTensorRtAndSetsEnv) {
  g_trt_append_calls = 0;
  g_trt_append_device_id = -1;

  ProviderConfig config;
  config.device_id = 1;
  config.trt_fp16_enable = true;
  config.trt_engine_cache_enable = true;
  config.trt_engine_cache_path = "/tmp/yolo_ort_compat_trt_cache";

  Ort::SessionOptions options;
  EXPECT_NO_THROW(compat::append_tensorrt_provider(options, config));
  EXPECT_EQ(g_trt_append_calls, 1);
  EXPECT_EQ(g_trt_append_device_id, config.device_id);
  EXPECT_STREQ(std::getenv("ORT_TENSORRT_FP16_ENABLE"), "1");
  EXPECT_STREQ(std::getenv("ORT_TENSORRT_ENGINE_CACHE_ENABLE"), "1");
  EXPECT_STREQ(std::getenv("ORT_TENSORRT_CACHE_PATH"),
               "/tmp/yolo_ort_compat_trt_cache");
}

TEST(OrtCompatLegacyTrt, CacheDisabledSetsEnableToZeroAndUnsetsPath) {
  setenv("ORT_TENSORRT_CACHE_PATH", "/stale/yolo/cache", 1);

  ProviderConfig config;
  config.trt_fp16_enable = false;
  config.trt_engine_cache_enable = false;
  config.trt_engine_cache_path.clear();

  Ort::SessionOptions options;
  EXPECT_NO_THROW(compat::append_tensorrt_provider(options, config));
  EXPECT_STREQ(std::getenv("ORT_TENSORRT_FP16_ENABLE"), "0");
  EXPECT_STREQ(std::getenv("ORT_TENSORRT_ENGINE_CACHE_ENABLE"), "0");
  EXPECT_EQ(std::getenv("ORT_TENSORRT_CACHE_PATH"), nullptr);
}

TEST(OrtCompatLegacyTrt, EmptyCachePathDisablesCachingAndUnsetsPath) {
  setenv("ORT_TENSORRT_CACHE_PATH", "/stale/yolo/cache", 1);

  ProviderConfig config;
  config.trt_fp16_enable = true;
  config.trt_engine_cache_enable = true;
  config.trt_engine_cache_path.clear();

  Ort::SessionOptions options;
  EXPECT_NO_THROW(compat::append_tensorrt_provider(options, config));
  EXPECT_STREQ(std::getenv("ORT_TENSORRT_FP16_ENABLE"), "1");
  EXPECT_STREQ(std::getenv("ORT_TENSORRT_ENGINE_CACHE_ENABLE"), "0");
  EXPECT_EQ(std::getenv("ORT_TENSORRT_CACHE_PATH"), nullptr);
}

#else

TEST(OrtCompatLegacyTrt, NotApplicableOnThisBuild) {
  GTEST_SKIP() << "legacy TRT branch only applies to ORT < 1.12 builds with a "
                  "TensorRT provider";
}

#endif
