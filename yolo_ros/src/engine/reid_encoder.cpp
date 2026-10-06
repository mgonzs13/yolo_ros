// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2018 Kaiyang Zhou
// SPDX-License-Identifier: MIT

#include "yolo_ros/engine/reid_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "yolo_ros/engine/provider.hpp"
#include "yolo_ros/utils/logs.hpp"
#include "yolo_ros/utils/string_utils.hpp"

namespace yolo_ros::engine {

ReIDEncoder::ReIDEncoder(const std::string &model_path,
                         const std::string &provider, const std::string &device)
    : env_(ORT_LOGGING_LEVEL_WARNING, "yolo_reid") {
  std::string requested = yolo_ros::utils::to_lower(provider);

  if (requested.empty()) {
    requested = "auto";
  }

  if (requested != "auto" && requested != "cpu" && requested != "cuda" &&
      requested != "tensorrt" && requested != "trt") {
    YOLO_LOG_WARN("Unknown ReID provider \"%s\"; using auto.",
                  provider.c_str());
    requested = "auto";
  }

  const std::vector<Provider> chain =
      provider_chain(requested, available_providers());
  const int device_id = parse_device_id(device);

  // Provider retry loop (mirrors engine::Model): a provider can be available
  // yet fail to build the session, so construction lives inside the loop.
  std::string last_error;

  for (const Provider primary : chain) {
    ProviderConfig config;
    config.device_id = device_id;

    if (primary == Provider::TensorRt && config.trt_engine_cache_enable) {
      config.trt_engine_cache_path = engine_cache_dir("", model_path);

      if (config.trt_engine_cache_path.empty()) {
        config.trt_engine_cache_enable = false;
      }
    }

    try {
      this->session_options_ = build_session_options(primary, config);
      this->session_ =
          Ort::Session(this->env_, model_path.c_str(), this->session_options_);
      this->active_provider_ = provider_name(primary);
      break;
    } catch (const Ort::Exception &e) {
      last_error = e.what();
      YOLO_LOG_WARN("ReID execution provider %s failed to initialize: %s",
                    provider_name(primary), e.what());
    }
  }

  if (this->active_provider_.empty()) {
    throw std::runtime_error(
        "No execution provider could initialize the ReID session" +
        (last_error.empty() ? std::string(".") : ": " + last_error));
  }

  Ort::AllocatorWithDefaultOptions allocator;

  for (std::size_t i = 0; i < this->session_.GetInputCount(); ++i) {
    this->input_name_alloc_.push_back(
        this->session_.GetInputNameAllocated(i, allocator));
    this->input_names_.push_back(this->input_name_alloc_.back().get());
  }

  for (std::size_t i = 0; i < this->session_.GetOutputCount(); ++i) {
    this->output_name_alloc_.push_back(
        this->session_.GetOutputNameAllocated(i, allocator));
    this->output_names_.push_back(this->output_name_alloc_.back().get());
  }

  const std::vector<int64_t> input_shape =
      this->session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

  if (input_shape.size() != 4 || input_shape[2] <= 0 || input_shape[3] <= 0) {
    throw std::runtime_error("ReID model input must be a fixed NCHW tensor.");
  }

  this->input_height_ = static_cast<int>(input_shape[2]);
  this->input_width_ = static_cast<int>(input_shape[3]);

  const std::vector<int64_t> output_shape = this->session_.GetOutputTypeInfo(0)
                                                .GetTensorTypeAndShapeInfo()
                                                .GetShape();
  const int64_t static_batch = output_shape.empty() ? 1 : output_shape[0];

  if (static_batch > 1) {
    throw std::runtime_error("ReID model must use a dynamic or batch-1 output; "
                             "got a fixed batch of " +
                             std::to_string(static_batch) + ".");
  }

  // The graph's declared feature dim is not trusted: exporters can leave it
  // symbolic (a TorchScript F.normalize tail emits e.g. "Divoutput_dim_1"),
  // which reads back as -1. The dimension is resolved from the runtime output
  // tensor in inference() instead.

  this->blob_.resize(static_cast<std::size_t>(this->input_height_) *
                     this->input_width_ * 3);
  this->memory_info_ =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  YOLO_LOG_INFO("ReID encoder %s loaded (%dx%d, provider %s).",
                model_path.c_str(), this->input_width_, this->input_height_,
                this->active_provider_.c_str());
}

ReIDEncoder::~ReIDEncoder() {}

void ReIDEncoder::make_batch(const cv::Mat &frame,
                             const std::vector<std::array<float, 4>> &boxes,
                             int width, int height, std::vector<float> &out) {
  if (frame.empty() || width <= 0 || height <= 0) {
    out.clear();
    return;
  }

  const std::size_t per_box = static_cast<std::size_t>(width) * height * 3;
  out.assign(per_box * boxes.size(), 0.0f);

  for (std::size_t b = 0; b < boxes.size(); ++b) {
    const int x1 = std::max(0, static_cast<int>(std::lround(boxes[b][0])));
    const int y1 = std::max(0, static_cast<int>(std::lround(boxes[b][1])));
    const int x2 =
        std::min(frame.cols, static_cast<int>(std::lround(boxes[b][2])));
    const int y2 =
        std::min(frame.rows, static_cast<int>(std::lround(boxes[b][3])));

    if (x2 <= x1 || y2 <= y1) {
      continue; // degenerate or fully out of frame: leave the zero patch
    }

    cv::Mat patch = frame(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    cv::Mat resized;
    cv::resize(patch, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3); // keep the [0, 255] range
    std::vector<cv::Mat> channels(3);

    for (int c = 0; c < 3; ++c) {
      channels[c] = cv::Mat(height, width, CV_32FC1,
                            out.data() + b * per_box +
                                static_cast<std::size_t>(c) * height * width);
    }

    cv::split(rgb, channels);
  }
}

std::vector<std::vector<float>>
ReIDEncoder::inference(const cv::Mat &frame,
                       const std::vector<std::array<float, 4>> &boxes) {
  std::vector<std::vector<float>> features;

  if (boxes.empty() || frame.empty()) {
    return features;
  }

  make_batch(frame, boxes, this->input_width_, this->input_height_,
             this->blob_);

  const std::vector<int64_t> shape = {static_cast<int64_t>(boxes.size()), 3,
                                      this->input_height_, this->input_width_};
  const std::size_t count = static_cast<std::size_t>(boxes.size()) * 3 *
                            this->input_height_ * this->input_width_;
  Ort::Value input_tensor =
      Ort::Value::CreateTensor<float>(this->memory_info_, this->blob_.data(),
                                      count, shape.data(), shape.size());
  std::vector<Ort::Value> outputs = this->session_.Run(
      Ort::RunOptions{nullptr}, this->input_names_.data(), &input_tensor,
      this->input_names_.size(), this->output_names_.data(),
      this->output_names_.size());

  if (outputs.empty()) {
    return features;
  }

  const float *data = outputs[0].GetTensorData<float>();
  const std::size_t rows = boxes.size();
  // Resolve the embedding dim from the runtime output tensor, which is always
  // concrete (unlike the graph's possibly-symbolic declared shape).
  const std::size_t total = static_cast<std::size_t>(
      outputs[0].GetTensorTypeAndShapeInfo().GetElementCount());
  const std::size_t dim = rows > 0 ? total / rows : 0;

  if (dim == 0 || rows * dim != total) {
    return features; // unexpected output shape: report no features
  }

  features.resize(rows);

  for (std::size_t i = 0; i < rows; ++i) {
    std::vector<float> feat(dim, 0.0f);
    double norm = 0.0;

    for (std::size_t j = 0; j < dim; ++j) {
      const float value = data[i * dim + j];
      feat[j] = std::isfinite(value) ? value : 0.0f;
      norm += static_cast<double>(feat[j]) * feat[j];
    }

    norm = std::sqrt(norm);

    if (norm > 1e-12) {
      for (float &value : feat) {
        value = static_cast<float>(value / norm);
      }
    }

    features[i] = std::move(feat);
  }

  return features;
}

} // namespace yolo_ros::engine
