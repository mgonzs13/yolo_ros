// Copyright (c) 2025 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include "yolo_ros/engine/model.hpp"
#include "onnxruntime_cxx_api.h"
#include "yolo_ros/engine/provider.hpp"
#include "yolo_ros/utils/string_utils.hpp"
#include "yolo_ros/yolo/utils.hpp"
#include <algorithm>
#include <ament_index_cpp/get_package_prefix.hpp>
#if __has_include(<ament_index_cpp/get_package_share_directory.hpp>)
#include <ament_index_cpp/get_package_share_directory.hpp>
#else
#include <ament_index_cpp/get_package_share_path.hpp>
#endif
#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <thread>

namespace {

// ament_index_cpp renamed get_package_share_directory() to
// get_package_share_path() (returning std::filesystem::path) in
// Rolling/Lyrical; normalise both to std::string here.
std::string get_package_share_directory(const std::string &package_name) {
#if __has_include(<ament_index_cpp/get_package_share_directory.hpp>)
  return ament_index_cpp::get_package_share_directory(package_name);
#else
  return ament_index_cpp::get_package_share_path(package_name).string();
#endif
}

} // namespace

namespace yolo_ros::engine {
Model::Model(yolo_ros::yolo::utils::YoloParams params)
    : env(ORT_LOGGING_LEVEL_WARNING, "yolo"), session_options(),
      input_image_shape(), num_input_nodes(0), num_output_nodes(0),
      memory_info(nullptr) {
  // Initialize session options
  int n_threads = params.n_threads;
  this->conf_threshold = params.threshold;
  this->iou_threshold = params.iou;
  std::string model_path = params.model_path;

  if (n_threads == -1) {
    n_threads = std::thread::hardware_concurrency();
  }

  // Resolve the availability-filtered provider fallback chain. "auto" prefers
  // CUDA, then CPU; an explicit provider forces its own chain.
  std::string requested = yolo_ros::utils::to_lower(params.provider);
  if (requested.empty()) {
    requested = "auto";
  }
  if (requested != "auto" && requested != "cpu" && requested != "cuda" &&
      requested != "tensorrt" && requested != "trt") {
    std::cerr << "Unknown provider \"" << params.provider << "\"; using auto."
              << std::endl;
    requested = "auto";
  }
  const std::vector<Provider> chain =
      provider_chain(requested, available_providers());
  std::cout << "Execution provider chain:";
  for (const Provider provider : chain) {
    std::cout << " " << provider_name(provider);
  }
  std::cout << std::endl;

  // Warn when an explicitly requested provider was dropped by the
  // availability filter (e.g. TensorRT requested on a CPU build).
  const std::string requested_norm =
      requested == "trt" ? "tensorrt" : requested;
  if (requested != "auto" && !chain.empty() &&
      requested_norm != provider_name(chain.front())) {
    std::cerr << "Requested provider \"" << requested
              << "\" is unavailable; using " << provider_name(chain.front())
              << "." << std::endl;
  }

  // An explicit provider wins over the device prefix; warn when they disagree
  // (only when the requested provider is actually the one being used).
  const std::string device_provider = device_provider_name(params.device);
  if (requested != "auto" && !chain.empty() &&
      requested_norm == provider_name(chain.front()) &&
      !device_provider.empty() && device_provider != requested_norm) {
    std::cerr << "provider \"" << requested << "\" contradicts device \""
              << params.device << "\"; using " << requested_norm << "."
              << std::endl;
  }

  // Try each provider in order. A provider can be available yet still fail to
  // build the session (unsupported op, TensorRT engine build error), so the
  // provider options and the Session construction both live in the retry loop.
  const int device_id = parse_device_id(params.device);
  std::string last_error;
  for (const Provider provider : chain) {
    ProviderConfig config;
    config.n_threads = n_threads;
    config.device_id = device_id;
    config.trt_fp16_enable = params.trt_fp16_enable;
    config.trt_engine_cache_enable = params.trt_engine_cache_enable;
    if (provider == Provider::TensorRt && config.trt_engine_cache_enable) {
      config.trt_engine_cache_path =
          engine_cache_dir(params.trt_engine_cache_path, model_path);
      if (config.trt_engine_cache_path.empty()) {
        config.trt_engine_cache_enable = false;
      }
    }
    try {
      this->session_options = build_session_options(provider, config);
      this->session =
          Ort::Session(this->env, model_path.c_str(), this->session_options);
      this->active_provider_ = provider_name(provider);
      break;
    } catch (const Ort::Exception &e) {
      last_error = e.what();
      std::cerr << "Execution provider " << provider_name(provider)
                << " failed to initialize: " << e.what() << std::endl;
    }
  }

  if (this->active_provider_.empty()) {
    throw std::runtime_error(
        "No execution provider could initialize the ONNX Runtime session" +
        (last_error.empty() ? std::string(".") : ": " + last_error));
  }
  // Names the primary EP that accepted the session; within a TensorRT session
  // ORT may still fall back node-by-node to CUDA.
  std::cout << "Using execution provider: " << this->active_provider_
            << std::endl;

  Ort::AllocatorWithDefaultOptions allocator;

  // Get input and output node information
  this->num_input_nodes = this->session.GetInputCount();
  this->num_output_nodes = this->session.GetOutputCount();

  // Allocate input and output node names
  for (size_t i = 0; i < this->num_input_nodes; i++) {
    this->input_node_name_alloc_strings.push_back(
        this->session.GetInputNameAllocated(i, allocator));
    this->inputNames.push_back(
        this->input_node_name_alloc_strings.back().get());
  }

  for (size_t i = 0; i < this->num_output_nodes; i++) {
    this->output_node_name_alloc_strings.push_back(
        this->session.GetOutputNameAllocated(i, allocator));
    this->outputNames.push_back(
        this->output_node_name_alloc_strings.back().get());
  }

  Ort::TypeInfo input_type_info = this->session.GetInputTypeInfo(0);
  std::vector<int64_t> input_tensor_shape_vec =
      input_type_info.GetTensorTypeAndShapeInfo().GetShape();

  if (input_tensor_shape_vec.size() >= 4) {
    this->fixed_batch_ =
        input_tensor_shape_vec[0] > 0 ? input_tensor_shape_vec[0] : 0;
    this->input_image_shape =
        cv::Size(static_cast<int>(input_tensor_shape_vec[3]),
                 static_cast<int>(input_tensor_shape_vec[2]));
    if (input_tensor_shape_vec[2] <= 0 || input_tensor_shape_vec[3] <= 0) {
      // `dynamic=True` exports carry dynamic H/W as well as batch, so the
      // graph cannot tell us the input size. All shipped models train at 640.
      this->input_image_shape = cv::Size(640, 640);
      std::cerr << "Warning: dynamic input size in the ONNX graph; pinning "
                   "640x640 (export with a static imgsz if this is wrong)."
                << std::endl;
    }
  } else {
    throw std::runtime_error("Invalid input tensor shape.");
  }

  // Pre-allocate the reusable input blob (1*3*H*W floats), written in place by
  // preprocess() and fed directly to ORT with no per-frame copy.
  this->input_buffer_.resize(
      static_cast<size_t>(this->input_image_shape.height) *
      this->input_image_shape.width * 3);

  // Load the class names (ONNX graph metadata first, coco.names as fallback).
  this->load_class_names();

  // Resolve the input channel order: parameter first, then the ONNX metadata
  // ("input_color") which our export tool stamps. Graphs cannot encode it.
  {
    const std::string color = yolo_ros::utils::to_lower(params.input_color);
    this->input_is_rgb_ = true; // ultralytics exports expect RGB
    if (color == "bgr") {
      this->input_is_rgb_ = false;
    } else if (!color.empty() && color != "rgb") {
      std::cerr << "Unknown input_color \"" << params.input_color
                << "\"; using rgb." << std::endl;
    }
    try {
      const Ort::ModelMetadata metadata = this->session.GetModelMetadata();
      Ort::AllocatorWithDefaultOptions allocator;
      auto value = metadata.LookupCustomMetadataMapAllocated(
          "input_color", static_cast<OrtAllocator *>(allocator));
      if (value) {
        const std::string meta = yolo_ros::utils::to_lower(value.get());
        if (meta == "rgb") {
          this->input_is_rgb_ = true;
        } else if (meta == "bgr") {
          this->input_is_rgb_ = false;
        }
      }
    } catch (const Ort::Exception &e) {
      std::cerr << "Warning: could not read \"input_color\" from the ONNX "
                   "metadata: "
                << e.what() << std::endl;
    }
    std::cout << "Input channel order: "
              << (this->input_is_rgb_ ? "rgb" : "bgr") << std::endl;
  }

  this->memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::cout << "Model " << model_path << " has been successfully loaded"
            << " (input " << this->input_image_shape.width << "x"
            << this->input_image_shape.height << ", FP32, batch "
            << (this->fixed_batch_ > 0 ? std::to_string(this->fixed_batch_)
                                       : std::string("dynamic"))
            << ")." << std::endl;
}

Model::~Model() {}

void yolo_ros::engine::Model::load_class_names() {
  // 1) Read the vocabulary embedded in the ONNX graph by ultralytics'
  //    exporter: key "names", value a Python-dict literal such as
  //    {0: 'person', 1: 'bicycle', ...} (some exporters write JSON-style
  //    {"0": "person", ...}; both are handled here).
  try {
    const Ort::ModelMetadata metadata = this->session.GetModelMetadata();
    Ort::AllocatorWithDefaultOptions allocator;
    auto names_value = metadata.LookupCustomMetadataMapAllocated(
        "names", static_cast<OrtAllocator *>(allocator));
    if (names_value) {
      const std::string names(names_value.get());
      std::map<int, std::string> indexed_names;
      int max_index = -1;
      const std::regex name_re(R"((\d+):\s*['"]([^'"]*)['"])");
      auto begin = names.cbegin();
      const auto end = names.cend();
      std::smatch match;
      while (std::regex_search(begin, end, match, name_re)) {
        const int idx = std::stoi(match[1].str());
        indexed_names[idx] = match[2].str();
        max_index = std::max(max_index, idx);
        begin = match.suffix().first;
      }
      if (max_index >= 0) {
        this->class_names.assign(static_cast<size_t>(max_index + 1), "");
        for (const auto &[idx, name] : indexed_names) {
          this->class_names[static_cast<size_t>(idx)] = name;
        }
        std::cout << "Loaded " << this->class_names.size()
                  << " class names from the ONNX metadata." << std::endl;
      }
    }
  } catch (const Ort::Exception &e) {
    std::cerr << "Warning: could not read \"names\" from the ONNX metadata: "
              << e.what() << std::endl;
  }

  // 2) Fall back to the coco.names file (previous behaviour) when the model
  //    does not carry a vocabulary. The file is installed into the package
  //    share directory, so resolve it through the ament index instead of a
  //    hardcoded relative path (which only worked from the workspace root).
  if (this->class_names.empty()) {
    std::string class_names_path;
    try {
      class_names_path =
          get_package_share_directory("yolo_ros") + "/conf/coco.names";
    } catch (const ament_index_cpp::PackageNotFoundError &e) {
      std::cerr << "Error: could not locate yolo_ros share directory: "
                << e.what() << std::endl;
      return;
    }
    std::ifstream class_names_file(class_names_path);
    if (!class_names_file.is_open()) {
      std::cerr << "Error: Could not open coco.names file at "
                << class_names_path << std::endl;
      return;
    }
    std::string line;
    while (std::getline(class_names_file, line)) {
      this->class_names.push_back(line);
    }
  }
}

std::vector<yolo_msgs::msg::Detection>
yolo_ros::engine::Model::detect(const cv::Mat &image) {
  auto batched = this->detect_batch({image});
  if (batched.empty()) {
    return {};
  }
  return std::move(batched.front());
}

std::vector<std::vector<yolo_msgs::msg::Detection>>
yolo_ros::engine::Model::detect_batch(const std::vector<cv::Mat> &images) {
  std::vector<std::vector<yolo_msgs::msg::Detection>> results;
  const std::size_t count = images.size();
  if (count == 0) {
    return results;
  }
  // A dynamic batch axis accepts any count; a fixed axis forces the chunk
  // size. A partial final chunk is padded by repeating its last image and the
  // padded results are dropped, so a stock batch-1 model keeps working.
  const std::size_t chunk = this->fixed_batch_ > 0
                                ? static_cast<std::size_t>(this->fixed_batch_)
                                : count;
  results.reserve(count);
  for (std::size_t start = 0; start < count; start += chunk) {
    const std::size_t valid = std::min(chunk, count - start);
    std::vector<cv::Mat> group(images.begin() + start,
                               images.begin() + start + valid);
    while (group.size() < chunk) {
      group.push_back(group.back());
    }
    auto group_results = this->run_batch(group);
    for (std::size_t i = 0; i < valid; ++i) {
      results.push_back(std::move(group_results[i]));
    }
  }
  return results;
}

std::vector<std::vector<yolo_msgs::msg::Detection>>
yolo_ros::engine::Model::run_batch(const std::vector<cv::Mat> &images) {
  const std::size_t count = images.size();
  std::vector<int64_t> input_tensor_shape = {static_cast<int64_t>(count), 3,
                                             this->input_image_shape.height,
                                             this->input_image_shape.width};
  const std::size_t per_image = static_cast<std::size_t>(
      this->input_image_shape.height * this->input_image_shape.width * 3);
  this->input_buffer_.resize(per_image * count);
  for (std::size_t i = 0; i < count; ++i) {
    this->preprocess_into(images[i], input_tensor_shape, i);
  }
  auto preds = this->inference(input_tensor_shape);
  std::vector<std::vector<yolo_msgs::msg::Detection>> results;
  results.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    auto sliced = slice_batch_outputs(preds, i, this->memory_info);
    results.push_back(
        this->postprocess(cv::Size(images[i].cols, images[i].rows),
                          this->input_image_shape, sliced));
  }
  return results;
}

void yolo_ros::engine::Model::preprocess_into(
    const cv::Mat &image, const std::vector<int64_t> &input_tensor_shape,
    std::size_t index) {
  const int H = static_cast<int>(input_tensor_shape[2]);
  const int W = static_cast<int>(input_tensor_shape[3]);
  cv::Mat resized_image = yolo_ros::yolo::utils::letterbox(
      image, cv::Size(W, H), cv::Scalar(114, 114, 114));
  if (this->input_is_rgb_) {
    resized_image = yolo_ros::yolo::utils::bgr_to_rgb(resized_image);
  }
  resized_image.convertTo(resized_image, CV_32FC3, 1.0 / 255.0);
  std::vector<cv::Mat> chw(resized_image.channels());
  for (int c = 0; c < resized_image.channels(); ++c) {
    chw[c] = cv::Mat(H, W, CV_32FC1,
                     this->input_buffer_.data() +
                         (static_cast<std::ptrdiff_t>(index) * 3 + c) * H * W);
  }
  cv::split(resized_image, chw);
}

std::vector<Ort::Value>
yolo_ros::engine::Model::inference(std::vector<int64_t> &input_tensor_shape) {
  size_t input_tensor_size =
      std::accumulate(input_tensor_shape.begin(), input_tensor_shape.end(), 1,
                      std::multiplies<int64_t>());

  // Reuse the persistent buffer as the (CPU-owned) input tensor — no copy.
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      this->memory_info, this->input_buffer_.data(), input_tensor_size,
      input_tensor_shape.data(), input_tensor_shape.size());

  std::vector<Ort::Value> predictions = this->session.Run(
      Ort::RunOptions{nullptr}, this->inputNames.data(), &input_tensor,
      this->num_input_nodes, this->outputNames.data(), this->num_output_nodes);

  return predictions;
}

std::vector<yolo_msgs::msg::Detection>
Model::postprocess(const cv::Size &original_image_size,
                   const cv::Size &resized_image_size,
                   const std::vector<Ort::Value> &preds) {
  (void)original_image_size;
  (void)resized_image_size;
  (void)preds;
  return std::vector<yolo_msgs::msg::Detection>();
}

} // namespace yolo_ros::engine

std::vector<Ort::Value>
yolo_ros::engine::slice_batch_outputs(const std::vector<Ort::Value> &preds,
                                      std::size_t batch_index,
                                      const Ort::MemoryInfo &memory_info) {
  std::vector<Ort::Value> sliced;
  sliced.reserve(preds.size());
  for (const auto &pred : preds) {
    const std::vector<int64_t> shape =
        pred.GetTensorTypeAndShapeInfo().GetShape();
    const std::size_t total = std::accumulate(
        shape.begin(), shape.end(), std::size_t{1}, std::multiplies<int64_t>());
    const std::size_t batch =
        shape.empty() ? 1 : static_cast<std::size_t>(shape[0]);
    const std::size_t per_batch = batch > 0 ? total / batch : total;
    float *data = const_cast<float *>(pred.GetTensorData<float>()) +
                  batch_index * per_batch;
    std::vector<int64_t> slice_shape;
    slice_shape.reserve(shape.size());
    slice_shape.push_back(1);
    for (std::size_t d = 1; d < shape.size(); ++d) {
      slice_shape.push_back(shape[d]);
    }
    sliced.push_back(Ort::Value::CreateTensor<float>(
        memory_info, data, per_batch, slice_shape.data(), slice_shape.size()));
  }
  return sliced;
}
