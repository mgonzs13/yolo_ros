// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2021 Yifu Zhang
// Portions Copyright (c) 2012-2025 Tomas Kazmar
// SPDX-License-Identifier: MIT AND BSD-2-Clause

/// @file
/// @brief Dense Jonker-Volgenant linear-assignment solver.

#ifndef YOLO_ROS__TRACKING__UTILS__LAPJV_HPP_
#define YOLO_ROS__TRACKING__UTILS__LAPJV_HPP_

#include <cstddef>
#include <vector>

/// @addtogroup yolo_tracking
/// @{
namespace yolo_ros::tracking::utils {

/// @brief Dense Jonker-Volgenant solver for the linear assignment problem.
///
/// Ported from ByteTrack's C++ deployment (MIT license):
/// `deploy/{ncnn,TensorRT}/cpp/src/lapjv.cpp` + `include/lapjv.h`, which
/// incorporates the BSD-2-Clause `lap` library's LAPJV implementation. See
/// THIRD_PARTY_NOTICES.md.
/// @param[in] n Matrix order (cost must be n x n).
/// @param[in] cost Row-major cost matrix (vector of row vectors).
/// @param[out] rowsol Column assigned to each row (rowsol[i] for row i).
/// @param[out] colsol Row assigned to each column (colsol[j] for column j).
/// @return 0 on success, nonzero if allocation failed (very unlikely).
int lapjv_internal(std::size_t n, const std::vector<std::vector<double>> &cost,
                   std::vector<int> &rowsol, std::vector<int> &colsol);

} // namespace yolo_ros::tracking::utils
/// @}

#endif // YOLO_ROS__TRACKING__UTILS__LAPJV_HPP_
