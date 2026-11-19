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

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "test_model_helpers.hpp"
#include "yolo_ros/engine/ort_compat.hpp"

namespace yolo_ros::engine::compat {
namespace {

constexpr char kRepo[] = "zwh20081/yolo26-onnx";

class OrtCompatTest : public ::testing::Test {
protected:
  void SetUp() override {
    const std::string model =
        yolo_ros::test::download_model(kRepo, "yolo26n.onnx");

    if (model.empty()) {
      GTEST_SKIP() << "yolo26n.onnx is unavailable";
    }

    this->session_ = std::make_unique<Ort::Session>(this->env_, model.c_str(),
                                                    this->options_);
  }

  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "ort_compat_test"};
  Ort::SessionOptions options_;
  std::unique_ptr<Ort::Session> session_;
};

TEST_F(OrtCompatTest, ReadsInputAndOutputNames) {
  const std::vector<std::string> inputs = input_names(*this->session_);
  const std::vector<std::string> outputs = output_names(*this->session_);

  ASSERT_EQ(inputs.size(), 1u);
  EXPECT_FALSE(inputs[0].empty());
  EXPECT_FALSE(outputs.empty());

  for (const std::string &name : outputs) {
    EXPECT_FALSE(name.empty());
  }
}

TEST_F(OrtCompatTest, ReadsModelMetadata) {
  const Ort::ModelMetadata metadata = this->session_->GetModelMetadata();
  const std::vector<std::string> keys = metadata_keys(metadata);

  EXPECT_NE(std::find(keys.begin(), keys.end(), "names"), keys.end());

  const auto names = lookup_metadata(metadata, "names");
  ASSERT_TRUE(names.has_value());
  EXPECT_FALSE(names->empty());

  EXPECT_FALSE(lookup_metadata(metadata, "no_such_key").has_value());
}

} // namespace
} // namespace yolo_ros::engine::compat
