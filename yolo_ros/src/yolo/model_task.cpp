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

#include "yolo_ros/utils/string_utils.hpp"

#include "yolo_ros/yolo/model_factory.hpp"

namespace yolo_ros::yolo {

ModelTask select_model_task(const std::string &model_type,
                            const std::string &model_path) {
  const std::string type = yolo_ros::utils::to_lower(model_type);

  const bool explicit_pose =
      !type.empty() && type != "auto" &&
      (type.find("pose") != std::string::npos ||
       type.find("keypoint") != std::string::npos || type == "kpt");
  const bool explicit_obb = !type.empty() && type != "auto" &&
                            (type.find("obb") != std::string::npos ||
                             type.find("rotat") != std::string::npos);
  const bool explicit_segment = !type.empty() && type != "auto" &&
                                type.find("segment") != std::string::npos;
  const bool explicit_detect = type == "yolo" || type == "detect" ||
                               type == "det" || type == "detection";
  const bool explicit_classify = !type.empty() && type != "auto" &&
                                 (type.find("class") != std::string::npos ||
                                  type == "cls" || type == "clf");
  const bool by_filename_pose = model_path.find("pose") != std::string::npos;
  const bool by_filename_obb = model_path.find("obb") != std::string::npos;
  // Ultralytics exports segmentation models as `*-seg.onnx` (no "segment"
  // substring), so the heuristic accepts the `-seg`/`_seg` suffix too.
  const bool by_filename_segment =
      model_path.find("segment") != std::string::npos ||
      model_path.find("-seg") != std::string::npos ||
      model_path.find("_seg") != std::string::npos;
  const bool by_filename_classify =
      model_path.find("cls") != std::string::npos ||
      model_path.find("classify") != std::string::npos;

  if (explicit_obb ||
      (by_filename_obb && !explicit_detect && !explicit_segment &&
       !explicit_pose && !explicit_classify && !by_filename_pose &&
       !by_filename_segment && !by_filename_classify)) {
    return ModelTask::Obb;
  } else if (explicit_pose || (by_filename_pose && !explicit_detect &&
                               !explicit_segment && !explicit_classify &&
                               !by_filename_segment && !by_filename_classify)) {
    return ModelTask::Pose;
  } else if (explicit_segment ||
             (by_filename_segment && !explicit_detect && !explicit_classify)) {
    return ModelTask::Segment;
  } else if (explicit_classify || (by_filename_classify && !explicit_detect &&
                                   !explicit_segment && !by_filename_segment)) {
    return ModelTask::Classify;
  }

  return ModelTask::Detect;
}

} // namespace yolo_ros::yolo
