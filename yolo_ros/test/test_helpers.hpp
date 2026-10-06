// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#ifndef YOLO_ROS__TEST__TEST_HELPERS_HPP_
#define YOLO_ROS__TEST__TEST_HELPERS_HPP_

#include <cstdint>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace yolo_ros::test {

inline Ort::MemoryInfo &cpu_memory_info() {
  static Ort::MemoryInfo info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  return info;
}

// Owns the backing buffer and the value that views it. CreateTensor does not
// copy, so the buffer must outlive the value; member order (values_ declared
// before value_) guarantees that on construction, and the value is moved out
// only into a container that lives inside the same test scope.
class Tensor {
public:
  Tensor(std::vector<int64_t> shape, std::vector<float> values)
      : shape_(std::move(shape)), values_(std::move(values)),
        value_(Ort::Value::CreateTensor<float>(cpu_memory_info(),
                                               values_.data(), values_.size(),
                                               shape_.data(), shape_.size())) {}

  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;
  Tensor(Tensor &&) = delete;

  Ort::Value &value() { return value_; }

private:
  std::vector<int64_t> shape_;
  std::vector<float> values_;
  Ort::Value value_{nullptr};
};

} // namespace yolo_ros::test

#endif // YOLO_ROS__TEST__TEST_HELPERS_HPP_
