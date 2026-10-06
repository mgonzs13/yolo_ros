// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

/// @file
/// @brief Lowercase helper shared by the parameter/device string parsing.

#ifndef YOLO_ROS__UTILS__STRING_UTILS_HPP_
#define YOLO_ROS__UTILS__STRING_UTILS_HPP_

#include <algorithm>
#include <cctype>
#include <string>

/// @addtogroup yolo_utils
/// @{
namespace yolo_ros::utils {

/// @brief Return @p value lowercased (ASCII).
/// @param[in] value String to lowercase.
/// @return The lowercased copy.
inline std::string to_lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

} // namespace yolo_ros::utils
/// @}

#endif // YOLO_ROS__UTILS__STRING_UTILS_HPP_
