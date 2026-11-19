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
