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

/// @brief YOLO task selected by `model.type` and the model filename.
enum class ModelTask { Detect, Segment, Pose, Obb, Classify };

/// @brief Resolve the task from an explicit `model.type` or the file name.
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
