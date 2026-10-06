// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

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
