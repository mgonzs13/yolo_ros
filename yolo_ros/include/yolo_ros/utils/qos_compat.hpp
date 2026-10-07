// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

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

/// @brief Map the image_reliability parameter to the local reliability type.
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
