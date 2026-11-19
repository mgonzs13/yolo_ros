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

#include "yolo_ros/yolo/model_factory.hpp"

#include "yolo_ros/yolo/classify.hpp"
#include "yolo_ros/yolo/detect.hpp"
#include "yolo_ros/yolo/obb.hpp"
#include "yolo_ros/yolo/pose.hpp"
#include "yolo_ros/yolo/segment.hpp"

namespace yolo_ros::yolo {

std::unique_ptr<yolo_ros::engine::Model>
create_model(const yolo_ros::yolo::utils::YoloParams &params) {
  switch (select_model_task(params.model_type, params.model_path)) {
  case ModelTask::Obb:
    return std::make_unique<YoloOBB>(params);
  case ModelTask::Pose:
    return std::make_unique<YoloPose>(params);
  case ModelTask::Segment:
    return std::make_unique<YoloSegment>(params);
  case ModelTask::Classify:
    return std::make_unique<YoloClassify>(params);
  case ModelTask::Detect:
  default:
    return std::make_unique<YoloDetect>(params);
  }
}

} // namespace yolo_ros::yolo
