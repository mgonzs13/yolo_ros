// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief CPU topology helpers for automatic thread counts.

#ifndef YOLO_ROS__UTILS__CPU_UTILS_HPP_
#define YOLO_ROS__UTILS__CPU_UTILS_HPP_

namespace yolo_ros::utils {

/// @brief Logical CPUs available to this process (affinity/cgroup aware).
/// @return At least 1.
int available_cpus();

/// @brief Physical CPU cores visible to the system.
///
/// Counts unique (package, core) topology pairs, so simultaneous-multithreading
/// siblings count once. Falls back to the available logical CPUs when the
/// topology cannot be read.
/// @return At least 1.
int num_physical_cores();

/// @brief Thread count for CPU math work.
///
/// Uses the performance cores on hybrid CPUs (Linux `cpu_core` interface),
/// otherwise the physical cores, clamped to the CPUs available to the process.
/// @return At least 1.
int num_math_threads();

} // namespace yolo_ros::utils

#endif // YOLO_ROS__UTILS__CPU_UTILS_HPP_
