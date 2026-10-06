// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2021 Yifu Zhang
// Portions Copyright (c) 2012-2025 Tomas Kazmar
// SPDX-License-Identifier: MIT AND BSD-2-Clause

// Dense Jonker-Volgenant linear assignment solver.
// Ported from ByteTrack's C++ deployment (MIT license):
//   deploy/{ncnn,TensorRT}/cpp/src/lapjv.cpp
// The deployment implementation incorporates `gatagat/lap` (BSD-2-Clause).
// See THIRD_PARTY_NOTICES.md.
// The algorithm and constants are preserved verbatim; only the memory handling
// is replaced with std::vector (no malloc/free, no NULL checks).

#include "yolo_ros/tracking/utils/lapjv.hpp"

#include <cmath>
#include <cstddef>

namespace yolo_ros::tracking::utils {

namespace {

constexpr double kLarge = 1e6; // LARGE in the original lapjv.h

using cost_t = double; // matches original typedef

// Column-reduction and reduction transfer for a dense cost matrix.
int ccrrt_dense(std::size_t n, const std::vector<std::vector<double>> &cost,
                std::vector<int> &free_rows, std::vector<int> &x,
                std::vector<int> &y, std::vector<double> &v) {
  for (std::size_t i = 0; i < n; i++) {
    x[i] = -1;
    v[i] = kLarge;
    y[i] = 0;
  }
  for (std::size_t i = 0; i < n; i++) {
    for (std::size_t j = 0; j < n; j++) {
      const cost_t c = cost[i][j];
      if (c < v[j]) {
        v[j] = c;
        y[j] = static_cast<int>(i);
      }
    }
  }

  std::vector<bool> unique(n, true);
  {
    std::size_t j = n;
    do {
      j--;
      const int i = y[j];
      if (x[i] < 0) {
        x[i] = static_cast<int>(j);
      } else {
        unique[i] = false;
        y[j] = -1;
      }
    } while (j > 0);
  }

  int n_free_rows = 0;
  for (std::size_t i = 0; i < n; i++) {
    if (x[i] < 0) {
      free_rows[n_free_rows++] = static_cast<int>(i);
    } else if (unique[i]) {
      const int j = x[i];
      cost_t min = kLarge;
      for (std::size_t j2 = 0; j2 < n; j2++) {
        if (j2 == static_cast<std::size_t>(j)) {
          continue;
        }
        const cost_t c = cost[i][j2] - v[j2];
        if (c < min) {
          min = c;
        }
      }
      v[j] -= min;
    }
  }
  return n_free_rows;
}

// Augmenting row reduction for a dense cost matrix.
int carr_dense(std::size_t n, const std::vector<std::vector<double>> &cost,
               int n_free_rows, std::vector<int> &free_rows,
               std::vector<int> &x, std::vector<int> &y,
               std::vector<double> &v) {
  std::size_t current = 0;
  int new_free_rows = 0;
  std::size_t rr_cnt = 0;
  while (current < static_cast<std::size_t>(n_free_rows)) {
    int i0;
    int j1, j2;
    cost_t v1, v2, v1_new;
    bool v1_lowers;

    rr_cnt++;
    const int free_i = free_rows[current++];
    j1 = 0;
    v1 = cost[free_i][0] - v[0];
    j2 = -1;
    v2 = kLarge;
    for (std::size_t j = 1; j < n; j++) {
      const cost_t c = cost[free_i][j] - v[j];
      if (c < v2) {
        if (c >= v1) {
          v2 = c;
          j2 = static_cast<int>(j);
        } else {
          v2 = v1;
          v1 = c;
          j2 = j1;
          j1 = static_cast<int>(j);
        }
      }
    }
    i0 = y[j1];
    v1_new = v[j1] - (v2 - v1);
    v1_lowers = v1_new < v[j1];
    if (rr_cnt < current * n) {
      if (v1_lowers) {
        v[j1] = v1_new;
      } else if (i0 >= 0 && j2 >= 0) {
        j1 = j2;
        i0 = y[j2];
      }
      if (i0 >= 0) {
        if (v1_lowers) {
          free_rows[--current] = i0;
        } else {
          free_rows[new_free_rows++] = i0;
        }
      }
    } else {
      if (i0 >= 0) {
        free_rows[new_free_rows++] = i0;
      }
    }
    x[free_i] = j1;
    y[j1] = free_i;
  }
  return new_free_rows;
}

// Find columns with minimum d[j] and put them on the SCAN list.
// (y is unused, kept for signature parity with the original port.)
std::size_t find_dense(std::size_t n, std::size_t lo, std::vector<double> &d,
                       std::vector<int> &cols, const std::vector<int> &y) {
  (void)y;
  std::size_t hi = lo + 1;
  cost_t mind = d[cols[lo]];
  for (std::size_t k = hi; k < n; k++) {
    int j = cols[k];
    if (d[j] <= mind) {
      if (d[j] < mind) {
        hi = lo;
        mind = d[j];
      }
      cols[k] = cols[hi];
      cols[hi++] = j;
    }
  }
  return hi;
}

// Scan all columns in TODO starting from an arbitrary column in SCAN and try to
// decrease the d of the TODO columns using the SCAN column.
int scan_dense(std::size_t n, const std::vector<std::vector<double>> &cost,
               std::size_t *plo, std::size_t *phi, std::vector<double> &d,
               std::vector<int> &cols, std::vector<int> &pred,
               const std::vector<int> &y, const std::vector<double> &v) {
  std::size_t lo = *plo;
  std::size_t hi = *phi;
  cost_t h, cred_ij;

  while (lo != hi) {
    int j = cols[lo++];
    const int i = y[j];
    const cost_t mind = d[j];
    h = cost[i][j] - v[j] - mind;
    // For all columns in TODO
    for (std::size_t k = hi; k < n; k++) {
      j = cols[k];
      cred_ij = cost[i][j] - v[j] - h;
      if (cred_ij < d[j]) {
        d[j] = cred_ij;
        pred[j] = i;
        if (cred_ij == mind) {
          if (y[j] < 0) {
            return j;
          }
          cols[k] = cols[hi];
          cols[hi++] = j;
        }
      }
    }
  }
  *plo = lo;
  *phi = hi;
  return -1;
}

// Single iteration of the modified Dijkstra shortest path algorithm as
// explained in the JV paper. Returns the closest free column index.
int find_path_dense(std::size_t n, const std::vector<std::vector<double>> &cost,
                    const int start_i, const std::vector<int> &y,
                    std::vector<double> &v, std::vector<int> &pred) {
  std::size_t lo = 0;
  std::size_t hi = 0;
  int final_j = -1;
  std::size_t n_ready = 0;
  std::vector<int> cols(n);
  std::vector<double> d(n);

  for (std::size_t i = 0; i < n; i++) {
    cols[i] = static_cast<int>(i);
    pred[i] = start_i;
    d[i] = cost[start_i][i] - v[i];
  }

  while (final_j == -1) {
    // No columns left on the SCAN list.
    if (lo == hi) {
      n_ready = lo;
      hi = find_dense(n, lo, d, cols, y);
      for (std::size_t k = lo; k < hi; k++) {
        const int j = cols[k];
        if (y[j] < 0) {
          final_j = j;
        }
      }
    }
    if (final_j == -1) {
      final_j = scan_dense(n, cost, &lo, &hi, d, cols, pred, y, v);
    }
  }

  {
    const cost_t mind = d[cols[lo]];
    for (std::size_t k = 0; k < n_ready; k++) {
      const int j = cols[k];
      v[j] += d[j] - mind;
    }
  }

  return final_j;
}

// Augment for a dense cost matrix.
int ca_dense(std::size_t n, const std::vector<std::vector<double>> &cost,
             int n_free_rows, std::vector<int> &free_rows, std::vector<int> &x,
             std::vector<int> &y, std::vector<double> &v) {
  std::vector<int> pred(n);

  for (int *pfree_i = free_rows.data();
       pfree_i < free_rows.data() + n_free_rows; pfree_i++) {
    int i = -1, j;
    std::size_t k = 0;

    j = find_path_dense(n, cost, *pfree_i, y, v, pred);
    while (i != *pfree_i) {
      i = pred[j];
      y[j] = i;
      const int old_j = x[i];
      x[i] = j;
      j = old_j; // SWAP_INDICES(j, x[i])
      k++;
      if (k >= n) {
        // unreachable in practice for valid cost matrices
        return -1;
      }
    }
  }
  return 0;
}

} // namespace

int lapjv_internal(std::size_t n, const std::vector<std::vector<double>> &cost,
                   std::vector<int> &rowsol, std::vector<int> &colsol) {
  rowsol.assign(n, -1);
  colsol.assign(n, -1);
  std::vector<int> free_rows(n);
  std::vector<double> v(n);

  int ret = ccrrt_dense(n, cost, free_rows, rowsol, colsol, v);
  int i = 0;
  while (ret > 0 && i < 2) {
    ret = carr_dense(n, cost, ret, free_rows, rowsol, colsol, v);
    i++;
  }
  if (ret > 0) {
    ret = ca_dense(n, cost, ret, free_rows, rowsol, colsol, v);
  }
  return ret;
}

} // namespace yolo_ros::tracking::utils
