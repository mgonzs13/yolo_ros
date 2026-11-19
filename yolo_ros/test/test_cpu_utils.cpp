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

#include "yolo_ros/utils/cpu_utils.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <thread>

TEST(CpuUtils, AvailableCpusAtLeastOne) {
  EXPECT_GE(yolo_ros::utils::available_cpus(), 1);
}

TEST(CpuUtils, PhysicalCoresAreBounded) {
  const int physical = yolo_ros::utils::num_physical_cores();
  const int logical =
      static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  EXPECT_GE(physical, 1);
  EXPECT_LE(physical, logical);
}

TEST(CpuUtils, MathThreadsAreBounded) {
  const int math = yolo_ros::utils::num_math_threads();
  const int logical =
      static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  EXPECT_GE(math, 1);
  EXPECT_LE(math, logical);
}

TEST(CpuUtils, MathThreadsDoNotExceedPhysicalCores) {
  EXPECT_LE(yolo_ros::utils::num_math_threads(),
            yolo_ros::utils::num_physical_cores());
}
