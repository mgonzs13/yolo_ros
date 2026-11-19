// Copyright (C) 2026 Alejandro González Cantón
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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

  Ort::Value &value() { return this->value_; }

private:
  std::vector<int64_t> shape_;
  std::vector<float> values_;
  Ort::Value value_{nullptr};
};

} // namespace yolo_ros::test

#endif // YOLO_ROS__TEST__TEST_HELPERS_HPP_
