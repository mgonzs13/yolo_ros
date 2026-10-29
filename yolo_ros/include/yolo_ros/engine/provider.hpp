// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

/// @file
/// @brief Execution-provider selection and ONNX Runtime session-option
/// building for the YOLO engine (CPU / CUDA / TensorRT / QNN with fallback).

#ifndef YOLO_ROS__ENGINE__PROVIDER_HPP_
#define YOLO_ROS__ENGINE__PROVIDER_HPP_

#include <onnxruntime_cxx_api.h>
#include <string>
#include <unordered_map>
#include <vector>

/// @addtogroup yolo_engine
/// @{
namespace yolo_ros::engine {
/// @brief Inference execution provider.
enum class Provider { Cpu, Cuda, TensorRt, Qnn };

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
  /// @brief QNN: full path (or bare soname) of the HTP backend library the
  /// QNN EP loads with dlopen on the source-built ONNX Runtime 1.28.0; the
  /// default "libQnnHtp.so" resolves from the loader path (e.g. /usr/lib on
  /// the VENTUNO Q). Point it at <sdk>/lib/<arch>/libQnnHtp.so for an SDK
  /// prefix.
  std::string qnn_backend_path{"libQnnHtp.so"};
  /// @brief QNN: SoC model number (QCS8275 = 82).
  std::string qnn_soc_model{"82"};
  /// @brief QNN: minimum HTP architecture (v75 on the Dragonwing IQ8).
  std::string qnn_htp_arch{"75"};
  /// @brief QNN: HTP performance mode ("burst", "sustained_high_performance",
  /// "balanced", ...).
  std::string qnn_htp_performance_mode{"burst"};
  /// @brief QNN: offload graph I/O quantization to the CPU EP, keeping the
  /// ONNX graph inputs/outputs float32.
  bool qnn_offload_graph_io_quantization{true};
  /// @brief QNN: allow fp16 inference for float32 graphs (quantized QDQ
  /// models keep this disabled).
  bool qnn_enable_htp_fp16_precision{false};
  /// @brief QNN: generate the context cache now, as returned by
  /// resolve_qnn_model_path() (not merely "caching enabled").
  bool qnn_context_cache_enable{true};
  /// @brief QNN: context cache file path; set by
  /// resolve_qnn_model_path() when empty.
  std::string qnn_context_cache_path;
};

/// @brief Execution providers available in the linked ONNX Runtime build.
/// @return The available providers, in ONNX Runtime's reported order.
std::vector<Provider> available_providers();

/// @brief Resolve @p requested into an ordered, availability-filtered fallback
/// chain. Unknown values are treated as "auto" (CUDA -> CPU).
/// @param[in] requested "auto" (CUDA -> CPU), "cuda" (CUDA -> CPU),
/// "tensorrt"/"trt" (TensorRT -> CUDA -> CPU), "qnn" (QNN -> CPU) or "cpu"
/// (case-insensitive).
/// @param[in] available Providers present in the ONNX Runtime build.
/// @return The ordered chain; CPU is always kept.
std::vector<Provider> provider_chain(const std::string &requested,
                                     const std::vector<Provider> &available);

/// @brief Lower-case name of @p provider.
/// @param[in] provider Provider to name.
/// @return "cpu", "cuda", "tensorrt" or "qnn".
const char *provider_name(Provider provider);

/// @brief Parse the device ordinal from an ultralytics-style `device` string.
/// @param[in] device e.g. "cuda:0", "trt:1", "1", "cpu".
/// @return The ordinal, or 0 when absent, negative or invalid.
int parse_device_id(const std::string &device);

/// @brief Provider named by the prefix of a device string ("cuda", "tensorrt",
/// "trt", "qnn" or "cpu"); empty when the device names no provider (e.g. "0",
/// "1").
/// @param[in] device Ultralytics-style device string.
/// @return "cpu", "cuda", "tensorrt", "qnn", or "" when absent/unrecognized.
const char *device_provider_name(const std::string &device);

/// @brief QNN provider option key/value map for @p config.
/// @param[in] config QNN configuration; a non-empty `qnn_backend_path` is
/// forwarded to ONNX Runtime 1.28.0 as the `backend_path` option (the HTP
/// backend library to dlopen), while an empty value omits the key so ONNX
/// Runtime resolves its own default HTP backend.
/// @return Options passed to AppendExecutionProvider("QNN", ...).
std::unordered_map<std::string, std::string>
qnn_provider_options(const ProviderConfig &config);

/// @brief Build session options for @p primary, appending the execution
/// provider(s). A TensorRT primary also appends CUDA as node-level fallback.
/// @param[in] primary Provider to append.
/// @param[in] config Provider configuration; @c n_threads must already be
/// resolved (no -1 sentinel).
/// @return The configured session options.
/// @throws Ort::Exception if the provider cannot be appended.
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

/// @brief QNN context cache file for @p model_path, creating @p base
/// recursively.
/// @param[in] base Configured cache base (empty ->
/// ~/.cache/yolo_ros/qnn_context).
/// @param[in] model_path Model path; its stem and a short file fingerprint
/// name the cache file.
/// @return "<base>/<stem>-<fingerprint>.ctx.onnx" (fingerprint = short hash of
/// the absolute path + size + mtime), or "" when the directory cannot be
/// created.
std::string qnn_context_path(const std::string &base,
                             const std::string &model_path);

/// @brief Resolve which model path a QNN session should load and whether it
/// must generate the context cache first.
struct QnnModelPath {
  /// @brief Path to pass to Ort::Session (the cache file when it exists).
  std::string model_path;
  /// @brief Cache file location when caching is enabled; empty when no cache
  /// path applies.
  std::string context_path;
  /// @brief True when the session must serialize the compiled context.
  bool generate{false};
};

/// @brief Apply the QNN context-cache policy.
/// @param[in] cache_base Cache base directory ("" -> default under $HOME).
/// @param[in] model_path Original ONNX model path.
/// @param[in] cache_enable False disables caching entirely.
/// @return Load path plus generation policy.
QnnModelPath resolve_qnn_model_path(const std::string &cache_base,
                                    const std::string &model_path,
                                    bool cache_enable);
} // namespace yolo_ros::engine
/// @}

#endif // YOLO_ROS__ENGINE__PROVIDER_HPP_
