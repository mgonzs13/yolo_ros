// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/utils/cpu_utils.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <thread>

TEST(CpuUtils, AvailableCpusAtLeastOne) {
  EXPECT_GE(yolo_ros::utils::available_cpus(), 1);
}

TEST(CpuUtils, PhysicalCoresAreBounded) {
  const int physical = yolo_ros::utils::num_physical_cores();
  const int logical = static_cast<int>(
      std::max(1u, std::thread::hardware_concurrency()));
  EXPECT_GE(physical, 1);
  EXPECT_LE(physical, logical);
}

TEST(CpuUtils, MathThreadsAreBounded) {
  const int math = yolo_ros::utils::num_math_threads();
  const int logical = static_cast<int>(
      std::max(1u, std::thread::hardware_concurrency()));
  EXPECT_GE(math, 1);
  EXPECT_LE(math, logical);
}

TEST(CpuUtils, MathThreadsDoNotExceedPhysicalCores) {
  EXPECT_LE(yolo_ros::utils::num_math_threads(),
            yolo_ros::utils::num_physical_cores());
}
