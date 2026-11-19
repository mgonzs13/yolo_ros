// Copyright (C) 2026 Miguel Ángel González Santamarta
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
/// @brief QoS compatibility helpers across distros.
///
/// Foxy's rclcpp::QoS::reliability() only accepts the raw
/// rmw_qos_reliability_policy_t and has no rclcpp::ReliabilityPolicy enum
/// (added in Galactic). reliability_policy_from_int() returns whichever type
/// the installed rclcpp expects so call sites stay identical.

#ifndef YOLO_ROS__UTILS__QOS_COMPAT_HPP_
#define YOLO_ROS__UTILS__QOS_COMPAT_HPP_

#include "rclcpp/qos.hpp"

namespace yolo_ros::utils {

/// @brief Map the camera reliability parameter to the local reliability type.
/// @param value 0 = system default, 1 = reliable, anything else = best effort.
#if defined(RCLCPP_LEGACY_QOS)
inline rmw_qos_reliability_policy_t reliability_policy_from_int(int value) {
  if (value == 0) {
    return RMW_QOS_POLICY_RELIABILITY_SYSTEM_DEFAULT;
  } else if (value == 1) {
    return RMW_QOS_POLICY_RELIABILITY_RELIABLE;
  }

  return RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;
}
#else
inline rclcpp::ReliabilityPolicy reliability_policy_from_int(int value) {
  if (value == 0) {
    return rclcpp::ReliabilityPolicy::SystemDefault;
  } else if (value == 1) {
    return rclcpp::ReliabilityPolicy::Reliable;
  }

  return rclcpp::ReliabilityPolicy::BestEffort;
}
#endif

} // namespace yolo_ros::utils

#endif // YOLO_ROS__UTILS__QOS_COMPAT_HPP_
