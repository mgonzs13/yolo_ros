// Copyright (C) 2026 Alejandro González Cantón
// Portions Copyright (C) 2021 Yifu Zhang
// Portions Copyright (C) 2012-2025 Tomas Kazmar
//
// This file is distributed under the MIT License; the LAPJV portions are
// also distributed under the BSD 2-Clause License.
//
// MIT License
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
//
// BSD 2-Clause License
//
// Copyright (C) 2012-2025, Tomas Kazmar
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

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
