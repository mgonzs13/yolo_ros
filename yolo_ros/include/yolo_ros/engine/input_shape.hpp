// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief Resolution of the network input size from the ONNX graph shape.

#ifndef YOLO_ROS__ENGINE__INPUT_SHAPE_HPP_
#define YOLO_ROS__ENGINE__INPUT_SHAPE_HPP_

#include <cstdint>
#include <vector>

/// @addtogroup yolo_engine
/// @{
namespace yolo_ros::engine {

/// @brief Resolved network input size and which source won.
struct InputShape {
  /// @brief Width in pixels.
  int width = 0;
  /// @brief Height in pixels.
  int height = 0;
  /// @brief True when the graph did not provide a static size for at least one
  /// dimension (dynamic or missing), so the parameters were used for it.
  bool from_params = false;
  /// @brief True when a static graph dimension overrode a different parameter.
  bool params_ignored = false;
};

/// @brief Resolve the input size from the ONNX NCHW input shape.
///
/// Static dimensions come from the graph, dynamic ones (<= 0) from the
/// parameters. The batch and channel dimensions are not touched.
/// @param graph_shape NCHW input shape as reported by ONNX Runtime.
/// @param img_width Configured width.
/// @param img_height Configured height.
/// @return The resolved size plus which source won.
InputShape resolve_input_shape(const std::vector<int64_t> &graph_shape,
                               int img_width, int img_height);

} // namespace yolo_ros::engine
/// @}

#endif // YOLO_ROS__ENGINE__INPUT_SHAPE_HPP_
