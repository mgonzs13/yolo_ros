// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/engine/input_shape.hpp"

namespace yolo_ros::engine {

InputShape resolve_input_shape(const std::vector<int64_t> &graph_shape,
                               int img_width, int img_height) {
  InputShape shape;
  shape.width = img_width;
  shape.height = img_height;

  if (graph_shape.size() < 4) {
    shape.from_params = true;
    return shape;
  }

  // A channel axis of 3 in the last dimension (and not in the second) marks a
  // channel-last graph (NHWC), as produced by the Ultralytics QNN exports;
  // anything else keeps the ONNX-standard NCHW layout.
  shape.channel_last = graph_shape[3] == 3 && graph_shape[1] != 3;

  const int64_t graph_height =
      shape.channel_last ? graph_shape[1] : graph_shape[2];
  const int64_t graph_width =
      shape.channel_last ? graph_shape[2] : graph_shape[3];

  if (graph_width > 0) {
    shape.width = static_cast<int>(graph_width);
    shape.params_ignored = shape.width != img_width;
  } else {
    shape.from_params = true;
  }

  if (graph_height > 0) {
    shape.height = static_cast<int>(graph_height);
    shape.params_ignored = shape.params_ignored || shape.height != img_height;
  } else {
    shape.from_params = true;
  }

  return shape;
}

} // namespace yolo_ros::engine
