// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

/// @file
/// @brief Task selection and construction shared by the inference nodes.

#ifndef YOLO_ROS__YOLO__MODEL_FACTORY_HPP_
#define YOLO_ROS__YOLO__MODEL_FACTORY_HPP_

#include <memory>
#include <string>

#include "yolo_ros/engine/model.hpp"
#include "yolo_ros/yolo/utils.hpp"

/// @addtogroup yolo_tasks
/// @{
namespace yolo_ros::yolo {

/// @brief YOLO task selected by `model_type` and the model filename.
enum class ModelTask { Detect, Segment, Pose, Obb, Classify };

/// @brief Resolve the task from an explicit `model_type` or the file name.
/// @param[in] model_type Explicit type ("auto"/"" uses the file-name
/// heuristic).
/// @param[in] model_path Model path used by the heuristic.
/// @return The selected task.
ModelTask select_model_task(const std::string &model_type,
                            const std::string &model_path);

/// @brief Build the YOLO task object selected by @p params.
/// @param[in] params Model, task and preprocessing configuration.
/// @return A new model wrapper.
std::unique_ptr<yolo_ros::engine::Model>
create_model(const yolo_ros::yolo::utils::YoloParams &params);

} // namespace yolo_ros::yolo
/// @}

#endif // YOLO_ROS__YOLO__MODEL_FACTORY_HPP_
