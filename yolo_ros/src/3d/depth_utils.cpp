// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2023-2025 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/3d/depth_utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include <opencv2/imgproc.hpp>

namespace yolo_ros::depth {

namespace {

// Left insertion index into a normalized, monotonically non-decreasing
// cumulative-weight array (np.searchsorted(cumsum, f, side='left')).
size_t weighted_searchsorted(const std::vector<double> &cum_weights, double f) {
  auto it = std::lower_bound(cum_weights.begin(), cum_weights.end(), f);
  return static_cast<size_t>(it - cum_weights.begin());
}

// Weighted order statistics over a sample: indices sorted by ascending value
// and the matching normalized cumulative weights.
struct WeightedOrder {
  std::vector<size_t> idx;
  std::vector<double> cum;
};

WeightedOrder weighted_order(const std::vector<double> &values,
                             const std::vector<double> &weights) {
  WeightedOrder order;
  const size_t n = values.size();
  order.idx.resize(n);
  std::iota(order.idx.begin(), order.idx.end(), 0);
  std::sort(order.idx.begin(), order.idx.end(),
            [&](size_t a, size_t b) { return values[a] < values[b]; });
  order.cum.assign(n, 0.0);
  double acc = 0.0;
  for (size_t i = 0; i < n; ++i) {
    acc += weights[order.idx[i]];
    order.cum[i] = acc;
  }
  if (order.cum.back() > 0.0) {
    for (double &c : order.cum) c /= order.cum.back();
  }
  return order;
}

// Weighted trimmed-mean center plus weighted-percentile [min, max]. Shared by
// compute_depth_bounds_weighted (trim 2/98, percentiles 1/99, weighted-mean
// fallback) and compute_axis_bounds (trim 5/95, percentiles 3/97, median
// fallback), matching the upstream Python statistics.
struct TrimmedBounds {
  double center;
  double min;
  double max;
};

TrimmedBounds weighted_trimmed_bounds(const std::vector<double> &values,
                                      const std::vector<double> &weights,
                                      double trim_lo, double trim_hi,
                                      double pct_lo, double pct_hi,
                                      bool weighted_mean_center) {
  const WeightedOrder order = weighted_order(values, weights);
  const std::vector<double> &cum = order.cum;

  const auto fallback = [&]() {
    return weighted_mean_center ? weighted_mean(values, weights)
                                : median(values);
  };

  double center = median(values);
  if (!cum.empty() && cum.back() > 0.0) {
    const size_t lo = weighted_searchsorted(cum, trim_lo);
    const size_t hi = weighted_searchsorted(cum, trim_hi);
    if (hi > lo) {
      double sw = 0.0;
      double zc = 0.0;
      for (size_t i = lo; i < hi; ++i) {
        zc += values[order.idx[i]] * weights[order.idx[i]];
        sw += weights[order.idx[i]];
      }
      center = (sw > 0.0) ? zc / sw : fallback();
    } else {
      center = fallback();
    }
  }

  double vmin, vmax;
  if (!cum.empty() && cum.back() > 0.0) {
    vmin = values[order.idx[weighted_searchsorted(cum, pct_lo)]];
    vmax = values[order.idx[weighted_searchsorted(cum, pct_hi)]];
  } else {
    const auto mm = std::minmax_element(values.begin(), values.end());
    vmin = *mm.first;
    vmax = *mm.second;
  }
  return {center, vmin, vmax};
}

// Valid-depth pixels of a detection. When the detection carries a segmentation
// mask the mask polygon is rasterized (cropped to its bounding rect) and
// sampled; otherwise the 2D bounding box is sampled. Only pixels with finite
// depth > 0 are returned, with their absolute image coordinates.
struct RegionPixels {
  std::vector<int> xs;
  std::vector<int> ys;
  std::vector<double> depths;
};

RegionPixels collect_region(const cv::Mat &depth_image,
                            const yolo_msgs::msg::Detection &detection,
                            int depth_units_divisor, int stride) {
  RegionPixels pixels;
  const auto push = [&](int v, int u) {
    const double d = depth_at_pixel(depth_image, v, u, depth_units_divisor);
    if (std::isfinite(d) && d > 0.0) {
      pixels.depths.push_back(d);
      pixels.xs.push_back(u);
      pixels.ys.push_back(v);
    }
  };

  if (!detection.mask.data.empty()) {
    std::vector<cv::Point> poly;
    poly.reserve(detection.mask.data.size());
    for (const auto &p : detection.mask.data) {
      poly.emplace_back(cvRound(p.x), cvRound(p.y));
    }
    cv::Rect roi = cv::boundingRect(poly);
    roi &= cv::Rect(0, 0, depth_image.cols, depth_image.rows);
    if (roi.width <= 0 || roi.height <= 0) {
      return pixels;
    }
    std::vector<std::vector<cv::Point>> local(1);
    local[0].reserve(poly.size());
    for (const auto &p : poly) {
      local[0].push_back(p - roi.tl());
    }
    cv::Mat mask = cv::Mat::zeros(roi.size(), CV_8UC1);
    cv::fillPoly(mask, local, cv::Scalar(255));
    for (int v = 0; v < mask.rows; v += stride) {
      const uchar *row = mask.ptr<uchar>(v);
      for (int u = 0; u < mask.cols; u += stride) {
        if (row[u]) {
          push(v + roi.y, u + roi.x);
        }
      }
    }
    return pixels;
  }

  const int center_x = static_cast<int>(detection.bbox.center.position.x);
  const int center_y = static_cast<int>(detection.bbox.center.position.y);
  const int size_x = static_cast<int>(detection.bbox.size.x);
  const int size_y = static_cast<int>(detection.bbox.size.y);
  const int u_min = std::max(center_x - size_x / 2, 0);
  const int u_max = std::min(center_x + size_x / 2, depth_image.cols - 1);
  const int v_min = std::max(center_y - size_y / 2, 0);
  const int v_max = std::min(center_y + size_y / 2, depth_image.rows - 1);
  if (u_max <= u_min || v_max <= v_min) {
    return pixels;
  }
  for (int v = v_min; v < v_max; v += stride) {
    for (int u = u_min; u < u_max; u += stride) {
      push(v, u);
    }
  }
  return pixels;
}

double dot3(const Point3 &a, const Point3 &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Point3 cross3(const Point3 &a, const Point3 &b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}

double norm3(const Point3 &a) { return std::sqrt(dot3(a, a)); }

void normalize3(Point3 &a) {
  const double n = norm3(a);
  if (n > 0.0) {
    a[0] /= n;
    a[1] /= n;
    a[2] /= n;
  }
}

// ---- Quaternion helpers (q stored as [w, x, y, z], matching tf2).

std::array<double, 4> quat_normalize(std::array<double, 4> q) {
  const double n =
      std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (n < 1e-12) {
    return {1.0, 0.0, 0.0, 0.0};
  }
  for (double &c : q) c /= n;
  return q;
}

std::array<double, 4> quat_multiply(const std::array<double, 4> &q1,
                                    const std::array<double, 4> &q2) {
  const double w1 = q1[0], x1 = q1[1], y1 = q1[2], z1 = q1[3];
  const double w2 = q2[0], x2 = q2[1], y2 = q2[2], z2 = q2[3];
  return {w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
          w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
          w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
          w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2};
}

bool quat_is_identity(const std::array<double, 4> &q_wxyz, double tol = 1e-3) {
  const auto q = quat_normalize(q_wxyz);
  return std::abs(q[1]) < tol && std::abs(q[2]) < tol && std::abs(q[3]) < tol &&
         std::abs(std::abs(q[0]) - 1.0) < tol;
}

// Convert a 3x3 rotation matrix (row-major R[row][col], columns = box axes)
// into a quaternion in ROS order [x, y, z, w] (scipy as_quat ordering).
std::array<double, 4> matrix_to_quat(const double R[9]) {
  // Indices 0..8 in row-major: m00 m01 m02 m10 m11 m12 m20 m21 m22.
  const double trace = R[0] + R[4] + R[8];
  double w, x, y, z;
  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    w = 0.25 * s;
    x = (R[7] - R[5]) / s;
    y = (R[2] - R[6]) / s;
    z = (R[3] - R[1]) / s;
  } else if (R[0] > R[4] && R[0] > R[8]) {
    const double s = std::sqrt(1.0 + R[0] - R[4] - R[8]) * 2.0;
    w = (R[7] - R[5]) / s;
    x = 0.25 * s;
    y = (R[1] + R[3]) / s;
    z = (R[2] + R[6]) / s;
  } else if (R[4] > R[8]) {
    const double s = std::sqrt(1.0 + R[4] - R[0] - R[8]) * 2.0;
    w = (R[2] - R[6]) / s;
    x = (R[1] + R[3]) / s;
    y = 0.25 * s;
    z = (R[5] + R[7]) / s;
  } else {
    const double s = std::sqrt(1.0 + R[8] - R[0] - R[4]) * 2.0;
    w = (R[3] - R[1]) / s;
    x = (R[2] + R[6]) / s;
    y = (R[5] + R[7]) / s;
    z = 0.25 * s;
  }
  const auto q = quat_normalize({w, x, y, z});
  return {q[1], q[2], q[3], q[0]}; // [x, y, z, w]
}

// ---- Orientation-estimation helpers (Python detect_3d_node.py).

// Weighted percentiles (np.interp on the weighted cumulative distribution).
std::pair<double, double>
weighted_percentiles(const std::vector<double> &values,
                     const std::vector<double> &weights, double q_lo,
                     double q_hi) {
  if (values.empty()) return {0.0, 0.0};
  std::vector<size_t> idx(values.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&](size_t a, size_t b) { return values[a] < values[b]; });
  std::vector<double> cum(values.size());
  double acc = 0.0;
  for (size_t i = 0; i < values.size(); ++i) {
    acc += weights[idx[i]];
    cum[i] = acc;
  }
  const double total = cum.back();
  if (total > 0.0) {
    for (double &c : cum) c /= total;
  }
  auto interp = [&](double q) -> double {
    if (q <= cum.front()) return values[idx.front()];
    if (q >= cum.back()) return values[idx.back()];
    const auto it = std::upper_bound(cum.begin(), cum.end(), q);
    const size_t hi = static_cast<size_t>(it - cum.begin());
    const size_t lo = hi - 1;
    const double x0 = cum[lo], x1 = cum[hi];
    const double y0 = values[idx[lo]], y1 = values[idx[hi]];
    const double t = (x1 > x0) ? (q - x0) / (x1 - x0) : 0.0;
    return y0 + t * (y1 - y0);
  };
  return {interp(q_lo), interp(q_hi)};
}

// Strided, mask-guided sample of the object's depth points in the camera
// frame (Python _sample_points_3d).
std::optional<std::vector<Point3>> sample_points_3d(
    const cv::Mat &depth_image, const sensor_msgs::msg::CameraInfo &depth_info,
    const yolo_msgs::msg::Detection &detection, int depth_units_divisor,
    int stride, int max_points, int min_seg_points) {
  const auto &k = depth_info.k;
  const double cx = k[2], cy = k[5], fx = k[0], fy = k[4];
  if (fx == 0.0 || fy == 0.0) {
    return std::nullopt;
  }

  const RegionPixels region =
      collect_region(depth_image, detection, depth_units_divisor, stride);
  if (region.depths.empty() ||
      static_cast<int>(region.depths.size()) < min_seg_points) {
    return std::nullopt;
  }
  const std::vector<double> &z = region.depths;
  const std::vector<int> &xs_img = region.xs;
  const std::vector<int> &ys_img = region.ys;

  // Local orientation-only depth cleanup around the median (replaces the old
  // maximum_detection_threshold guard; no parameter needed).
  const double orientation_depth_threshold = 0.30;
  const double z_med = median(z);
  std::vector<double> z2, xs2, ys2;
  for (size_t i = 0; i < z.size(); ++i) {
    if (std::abs(z[i] - z_med) <= orientation_depth_threshold) {
      z2.push_back(z[i]);
      xs2.push_back(xs_img[i]);
      ys2.push_back(ys_img[i]);
    }
  }
  if (z2.empty()) {
    return std::nullopt;
  }
  if (static_cast<int>(z2.size()) < min_seg_points) {
    return std::nullopt;
  }

  std::vector<size_t> keep(z2.size());
  std::iota(keep.begin(), keep.end(), 0);
  if (static_cast<int>(keep.size()) > max_points) {
    static thread_local std::mt19937 rng(std::random_device{}());
    std::vector<size_t> sampled;
    std::sample(keep.begin(), keep.end(), std::back_inserter(sampled),
                max_points, rng);
    keep = sampled;
  }

  std::vector<Point3> pts;
  pts.reserve(keep.size());
  for (const size_t i : keep) {
    const double zz = z2[i];
    const double x = zz * (xs2[i] - cx) / fx;
    const double y = zz * (ys2[i] - cy) / fy;
    if (std::isfinite(x) && std::isfinite(y) && std::isfinite(zz)) {
      pts.push_back({x, y, zz});
    }
  }
  if (static_cast<int>(pts.size()) < min_seg_points) {
    return std::nullopt;
  }
  return pts;
}

// PCA plane frame (normal, x_axis, y_axis) for a point cloud, with degeneracy
// checks (Python _plane_frame_from_pts_pca).
std::optional<std::array<Point3, 3>>
plane_frame_from_pts_pca(const std::vector<Point3> &pts) {
  if (pts.empty()) {
    return std::nullopt;
  }
  double mx = 0.0, my = 0.0, mz = 0.0;
  for (const auto &p : pts) {
    mx += p[0];
    my += p[1];
    mz += p[2];
  }
  const double n = static_cast<double>(pts.size());
  mx /= n;
  my /= n;
  mz /= n;

  double cov[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  for (const auto &p : pts) {
    const double dx = p[0] - mx, dy = p[1] - my, dz = p[2] - mz;
    cov[0][0] += dx * dx;
    cov[0][1] += dx * dy;
    cov[0][2] += dx * dz;
    cov[1][1] += dy * dy;
    cov[1][2] += dy * dz;
    cov[2][2] += dz * dz;
  }
  cov[1][0] = cov[0][1];
  cov[2][0] = cov[0][2];
  cov[2][1] = cov[1][2];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      cov[i][j] /= n;
    }
  }

  cv::Mat m(3, 3, CV_64F);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      m.at<double>(i, j) = cov[i][j];
    }
  }
  cv::Mat eigval, eigvec;
  cv::eigen(m, eigval, eigvec); // descending eigenvalues, rows = eigenvectors
  const double lam_max = eigval.at<double>(0, 0);
  const double lam_mid = eigval.at<double>(1, 0);
  const double lam_min = eigval.at<double>(2, 0);

  // Degenerate point clouds: reject blobs (no clear plane) and objects with
  // ambiguous in-plane axes (e.g. square walls).
  if (lam_max < 1e-12) {
    return std::nullopt;
  }
  if (lam_min / lam_max > 0.3) {
    return std::nullopt;
  }
  if ((lam_max - lam_mid) / lam_max < 0.15) {
    return std::nullopt;
  }

  // Normal = smallest-variance eigenvector.
  Point3 normal{eigvec.at<double>(2, 0), eigvec.at<double>(2, 1),
                eigvec.at<double>(2, 2)};
  const Point3 c{mx, my, mz};
  // Make the normal point away from the camera (camera at the origin).
  if (dot3(normal, c) > 0.0) {
    normal[0] = -normal[0];
    normal[1] = -normal[1];
    normal[2] = -normal[2];
  }

  // Major in-plane axis = largest-variance eigenvector.
  Point3 x{eigvec.at<double>(0, 0), eigvec.at<double>(0, 1),
           eigvec.at<double>(0, 2)};
  Point3 y = cross3(normal, x);
  if (norm3(y) < 1e-12) {
    return std::nullopt;
  }
  normalize3(y);
  x = cross3(y, normal);
  normalize3(x);

  // Reduce 180-degree yaw flips relative to a fixed x reference.
  if (x[0] < 0.0) { // dot(x, [1, 0, 0]) < 0
    x[0] = -x[0];
    x[1] = -x[1];
    x[2] = -x[2];
    y[0] = -y[0];
    y[1] = -y[1];
    y[2] = -y[2];
  }

  return std::array<Point3, 3>{normal, x, y};
}

// Enforce temporal sign consistency of the PCA in-plane axes (Python
// _consistent_axes): PCA has a sign ambiguity, so consecutive frames can flip
// the orientation by 180 degrees. Keeps the axes aligned with the previous
// frame for the same tracked object.
std::array<Point3, 3> consistent_axes(OrientationState &state,
                                      const std::string &id,
                                      const std::array<Point3, 3> &frame) {
  Point3 n = frame[0], x = frame[1], y = frame[2];
  const std::string key = id.empty() ? "_default" : id;
  auto it = state.last_axes.find(key);
  if (it != state.last_axes.end() && dot3(x, it->second) < -0.1) {
    x[0] = -x[0];
    x[1] = -x[1];
    x[2] = -x[2];
    y[0] = -y[0];
    y[1] = -y[1];
    y[2] = -y[2];
  }
  state.last_axes[key] = x;
  return {n, x, y};
}

} // namespace

double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  const size_t n = v.size();
  const size_t mid = n / 2;
  if (n % 2 == 1) {
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    return v[mid];
  }
  // Even count: nth_element places one middle element at `mid`; the other
  // middle is the largest element of the lower half.
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  const double hi = v[mid];
  const double lo = *std::max_element(v.begin(), v.begin() + mid);
  return 0.5 * (lo + hi);
}

double percentile_sorted(const std::vector<double> &v, double q) {
  if (v.empty()) return 0.0;
  const double pos = q * (static_cast<double>(v.size()) - 1.0);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = static_cast<size_t>(std::ceil(pos));
  if (lo == hi) return v[lo];
  const double frac = pos - lo;
  return v[lo] * (1.0 - frac) + v[hi] * frac;
}

double weighted_mean(const std::vector<double> &v,
                     const std::vector<double> &w) {
  const double sw = std::accumulate(w.begin(), w.end(), 0.0);
  if (sw <= 0.0) return median(v);
  double acc = 0.0;
  for (size_t i = 0; i < v.size(); ++i) acc += v[i] * w[i];
  return acc / sw;
}

std::vector<double> compute_spatial_weights(const std::vector<int> &xs,
                                            const std::vector<int> &ys,
                                            int center_x, int center_y,
                                            int size_x, int size_y) {
  const double hx = size_x / 2.0 + 1e-6;
  const double hy = size_y / 2.0 + 1e-6;
  std::vector<double> w(xs.size());
  for (size_t i = 0; i < xs.size(); ++i) {
    const double dx = (xs[i] - center_x) / hx;
    const double dy = (ys[i] - center_y) / hy;
    const double d = std::sqrt(dx * dx + dy * dy);
    w[i] = std::max(std::exp(-0.5 * std::pow(d / 0.8, 2.0)), 0.3);
  }
  return w;
}

DepthBounds compute_depth_bounds_weighted(std::vector<double> depth,
                                          std::vector<double> weight) {
  std::vector<double> d, w;
  for (size_t i = 0; i < depth.size(); ++i) {
    if (std::isfinite(depth[i]) && std::isfinite(weight[i])) {
      d.push_back(depth[i]);
      w.push_back(weight[i]);
    }
  }
  if (d.empty()) return {0.0, 0.0, 0.0};
  if (d.size() < 4) {
    const auto mm = std::minmax_element(d.begin(), d.end());
    return {median(d), *mm.first, *mm.second};
  }

  // Weighted histogram + smoothing for robust mode detection.
  const double d_min = *std::min_element(d.begin(), d.end());
  const double d_max = *std::max_element(d.begin(), d.end());
  const double range = d_max - d_min;
  const int n_bins =
      (!std::isfinite(range) || range <= 0.0)
          ? 30
          : std::clamp(static_cast<int>(std::lround(range / 0.01)), 20, 60);
  const double bin_w = (d_max - d_min) / n_bins;
  if (!(bin_w > 0.0)) {
    // Degenerate: all depths identical (zero-width histogram). The median
    // equals min and max, so return them directly.
    const auto mm = std::minmax_element(d.begin(), d.end());
    return {median(d), *mm.first, *mm.second};
  }
  std::vector<double> hist(n_bins, 0.0);
  for (size_t i = 0; i < d.size(); ++i) {
    const int idx =
        std::clamp(static_cast<int>((d[i] - d_min) / bin_w), 0, n_bins - 1);
    hist[idx] += w[i];
  }
  std::vector<double> smooth = hist;
  const int ks = std::min(5, n_bins / 4);
  if (ks >= 1) {
    const int half = ks / 2;
    for (int i = 0; i < n_bins; ++i) {
      double acc = 0.0;
      int cnt = 0;
      for (int k = -half; k <= half; ++k) {
        const int j = i + k;
        if (j >= 0 && j < n_bins) {
          acc += hist[j];
          ++cnt;
        }
      }
      smooth[i] = (cnt > 0) ? acc / cnt : hist[i];
    }
  }
  int peak = 0;
  for (int i = 1; i < n_bins; ++i) {
    if (smooth[i] > smooth[peak]) peak = i;
  }
  const double mode_depth = d_min + bin_w * (peak + 0.5);

  std::vector<double> dev(d.size());
  for (size_t i = 0; i < d.size(); ++i) dev[i] = std::abs(d[i] - mode_depth);
  const double mad = median(dev);
  std::vector<double> ds(d);
  std::sort(ds.begin(), ds.end());
  const double q25 = percentile_sorted(ds, 0.25);
  const double q75 = percentile_sorted(ds, 0.75);
  const double iqr = q75 - q25;

  double thr;
  if (iqr < 0.03) {
    thr = std::clamp(3.5 * mad, 0.08, 0.30);
  } else if (iqr < 0.10) {
    thr = std::clamp(4.0 * mad, 0.12, 0.40);
  } else {
    thr = std::clamp(5.0 * mad, 0.15, 0.60);
  }

  std::vector<double> obj_d, obj_w;
  for (size_t i = 0; i < d.size(); ++i) {
    if (std::abs(d[i] - mode_depth) <= thr) {
      obj_d.push_back(d[i]);
      obj_w.push_back(w[i]);
    }
  }
  const size_t min_points =
      std::max<size_t>(6, static_cast<size_t>(d.size() * 0.15));
  if (obj_d.size() < min_points) {
    // Fallback: weighted 2nd..85th percentile range.
    const WeightedOrder order = weighted_order(d, w);
    if (order.cum.back() > 0.0) {
      const double p2v = d[order.idx[weighted_searchsorted(order.cum, 0.02)]];
      const double p85v = d[order.idx[weighted_searchsorted(order.cum, 0.85)]];
      obj_d.clear();
      obj_w.clear();
      for (size_t i = 0; i < d.size(); ++i) {
        if (d[i] >= p2v && d[i] <= p85v) {
          obj_d.push_back(d[i]);
          obj_w.push_back(w[i]);
        }
      }
    }
  }
  if (obj_d.empty()) {
    obj_d = d;
    obj_w = w;
  }

  const TrimmedBounds bounds =
      weighted_trimmed_bounds(obj_d, obj_w, 0.02, 0.98, 0.01, 0.99, true);
  double z_center = bounds.center;
  double z_min = bounds.min;
  double z_max = bounds.max;
  if (z_center < z_min || z_center > z_max) {
    const double ext = std::max(z_max - z_min, 0.02);
    z_min = z_center - ext / 2.0;
    z_max = z_center + ext / 2.0;
  }
  if (z_max - z_min < 0.02) {
    z_min = z_center - 0.01;
    z_max = z_center + 0.01;
  }
  return {z_center, z_min, z_max};
}

AxisBounds compute_axis_bounds(const std::vector<double> &val3,
                               const std::vector<double> &w, double mad_mult,
                               double clip_lo, double clip_hi) {
  const size_t n = val3.size();
  if (n < 4) {
    const auto mm = std::minmax_element(val3.begin(), val3.end());
    return {median(val3), *mm.first, *mm.second};
  }
  auto sorted_idx = [&]() {
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b) { return val3[a] < val3[b]; });
    return idx;
  };

  // Weighted median as reference.
  const std::vector<size_t> idx = sorted_idx();
  std::vector<double> cum_weights(n, 0.0);
  {
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
      acc += w[idx[i]];
      cum_weights[i] = acc;
    }
    if (cum_weights.back() > 0.0) {
      for (auto &c : cum_weights) c /= cum_weights.back();
    }
  }
  const double med = val3[idx[weighted_searchsorted(cum_weights, 0.5)]];

  std::vector<double> dev(n);
  for (size_t i = 0; i < n; ++i) dev[i] = std::abs(val3[i] - med);
  const double mad = median(dev);
  const double thr = std::clamp(mad_mult * mad, clip_lo, clip_hi);

  std::vector<double> fv, fw;
  for (size_t i = 0; i < n; ++i) {
    if (dev[i] <= thr) {
      fv.push_back(val3[i]);
      fw.push_back(w[i]);
    }
  }
  const double min_pts = std::max(4.0, static_cast<double>(n) * 0.12);
  if (fv.size() < min_pts) {
    fv = val3;
    fw = w;
  }

  const TrimmedBounds bounds =
      weighted_trimmed_bounds(fv, fw, 0.05, 0.95, 0.03, 0.97, false);
  double center_val = bounds.center;
  double vmin = bounds.min;
  double vmax = bounds.max;
  const double min_size = 0.02;
  if (vmax - vmin < min_size) {
    vmin = center_val - min_size / 2.0;
    vmax = center_val + min_size / 2.0;
  }
  return {center_val, vmin, vmax};
}

double depth_at_pixel(const cv::Mat &depth_image, int v, int u,
                      int depth_units_divisor) {
  if (depth_image.empty() || v < 0 || u < 0 || v >= depth_image.rows ||
      u >= depth_image.cols) {
    return 0.0;
  }
  if (depth_image.type() == CV_16UC1) {
    if (depth_units_divisor <= 0) {
      return 0.0;
    }
    return static_cast<double>(depth_image.at<uint16_t>(v, u)) /
           depth_units_divisor;
  }
  if (depth_image.type() == CV_32FC1) {
    return static_cast<double>(depth_image.at<float>(v, u));
  }
  return 0.0;
}

std::optional<yolo_msgs::msg::BoundingBox3D> convert_bb_to_3d(
    const cv::Mat &depth_image, const sensor_msgs::msg::CameraInfo &depth_info,
    const yolo_msgs::msg::Detection &detection, int depth_units_divisor,
    const OrientationParams &orient_params, OrientationState *orient_state) {
  const int center_x = static_cast<int>(detection.bbox.center.position.x);
  const int center_y = static_cast<int>(detection.bbox.center.position.y);
  const int size_x = static_cast<int>(detection.bbox.size.x);
  const int size_y = static_cast<int>(detection.bbox.size.y);

  // Sample the detection region with an adaptive stride: large patches
  // (>=200 px per side, e.g. a person close to the camera) use step 2 so the
  // robust depth statistics see a representative 1/4 subset of the pixels.
  // The histogram/MAD/percentile pipeline is designed for dense sampling of
  // the same distribution, so this keeps the output statistics materially
  // unchanged while cutting the sample count (and every sort below) by 4x.
  const int step = (size_x * size_y >= 40000) ? 2 : 1;
  const RegionPixels region =
      collect_region(depth_image, detection, depth_units_divisor, step);
  if (region.depths.empty()) {
    return std::nullopt;
  }
  const std::vector<double> &depths = region.depths;
  const std::vector<int> &xs = region.xs;
  const std::vector<int> &ys = region.ys;

  // Weight the samples by their distance from the bbox centre so that
  // background/occluding pixels at the edges weigh less.
  const std::vector<double> weights =
      compute_spatial_weights(xs, ys, center_x, center_y, size_x, size_y);

  // Robust, depth-statistics based bounding box (position + per-axis extent).
  const DepthBounds db = compute_depth_bounds_weighted(depths, weights);
  if (!std::isfinite(db.center) || db.center == 0.0) {
    return std::nullopt;
  }

  const auto &k = depth_info.k; // [fx, 0, cx, 0, fy, cy, 0, 0, 1]
  const double fx = k[0], fy = k[4], px = k[2], py = k[5];
  if (fx == 0.0 || fy == 0.0) {
    return std::nullopt;
  }

  // Restrict the per-axis extents to the object's own depth cluster
  // (upstream: depth_cluster = (depths >= z_min) & (depths <= z_max)) so that
  // background/occluding depths at a different range are excluded.
  std::vector<double> cd, cw;
  std::vector<int> cx, cy;
  for (size_t i = 0; i < depths.size(); ++i) {
    if (depths[i] >= db.min && depths[i] <= db.max) {
      cd.push_back(depths[i]);
      cw.push_back(weights[i]);
      cx.push_back(xs[i]);
      cy.push_back(ys[i]);
    }
  }
  if (cd.empty()) {
    return std::nullopt;
  }

  // Back-project each cluster pixel to 3D and derive the per-axis bounds from
  // the actual 3D points (not just by projecting the 2D bbox).
  std::vector<double> x3(cd.size()), y3(cd.size());
  for (size_t i = 0; i < cd.size(); ++i) {
    x3[i] = cd[i] * (cx[i] - px) / fx;
    y3[i] = cd[i] * (cy[i] - py) / fy;
  }

  // Height uses a fixed MAD multiplier; width adapts to the depth variance to
  // distinguish occluded/3D objects from flat ones.
  const AxisBounds hb = compute_axis_bounds(y3, cw, 4.5, 0.06, 0.50);
  const double d_mean = std::accumulate(cd.begin(), cd.end(), 0.0) / cd.size();
  double d_var = 0.0;
  for (const double d : cd) {
    d_var += (d - d_mean) * (d - d_mean);
  }
  const double depth_std = std::sqrt(d_var / cd.size());
  const double w_mult = (depth_std > 0.15) ? 4.0 : 4.5;
  const double w_lo = (depth_std > 0.15) ? 0.06 : 0.08;
  const double w_hi = (depth_std > 0.15) ? 0.40 : 0.50;
  const AxisBounds wb = compute_axis_bounds(x3, cw, w_mult, w_lo, w_hi);

  if (!std::isfinite(hb.center) || !std::isfinite(hb.min) ||
      !std::isfinite(hb.max) || !std::isfinite(wb.center) ||
      !std::isfinite(wb.min) || !std::isfinite(wb.max) || db.center <= 0.0) {
    return std::nullopt;
  }

  yolo_msgs::msg::BoundingBox3D bbox3d;
  bbox3d.center.position.x = wb.center;
  bbox3d.center.position.y = hb.center;
  bbox3d.center.position.z = db.center;
  bbox3d.size.x = wb.max - wb.min;
  bbox3d.size.y = hb.max - hb.min;
  bbox3d.size.z = db.max - db.min;

  // --- oriented bounding box (OBB) ---
  if (orient_params.enable && orient_state != nullptr) {
    auto pts = sample_points_3d(depth_image, depth_info, detection,
                                depth_units_divisor,
                                /*stride=*/4, /*max_points=*/4000,
                                orient_params.min_seg_points_for_orientation);
    if (pts && static_cast<int>(pts->size()) >=
                   orient_params.min_seg_points_for_orientation) {
      auto frame = plane_frame_from_pts_pca(*pts);
      if (frame && cd.size() >= 4) {
        frame = consistent_axes(*orient_state, detection.id, *frame);
        const Point3 &n = (*frame)[0];
        const Point3 &xa = (*frame)[1];
        const Point3 &ya = (*frame)[2];

        // Back-project the cluster points once and recompute the box extents
        // along the object axes so the size matches the orientation
        // (camera-aligned extents would overestimate for rotated objects).
        const Point3 center3{wb.center, hb.center, db.center};
        const Point3 axes[3] = {xa, ya, n}; // R columns
        std::array<double, 3> half_extents{};
        bool ok = true;
        for (int axis = 0; axis < 3; ++axis) {
          std::vector<double> proj(cd.size());
          for (size_t i = 0; i < cd.size(); ++i) {
            const double dx = x3[i] - center3[0];
            const double dy = y3[i] - center3[1];
            const double dz = cd[i] - center3[2];
            proj[i] =
                dx * axes[axis][0] + dy * axes[axis][1] + dz * axes[axis][2];
          }
          const auto lo_hi = weighted_percentiles(proj, cw, 0.02, 0.98);
          half_extents[axis] =
              std::max((lo_hi.second - lo_hi.first) / 2.0, 0.01);
          if (!std::isfinite(half_extents[axis]) || half_extents[axis] <= 0.0) {
            ok = false;
            break;
          }
        }
        if (ok) {
          bbox3d.size.x = 2.0 * half_extents[0];
          bbox3d.size.y = 2.0 * half_extents[1];
          bbox3d.size.z = 2.0 * half_extents[2];

          // R (row-major) with columns = [xa, ya, n]; a proper rotation.
          const double R[9] = {xa[0], ya[0], n[0],  xa[1], ya[1],
                               n[1],  xa[2], ya[2], n[2]};
          const auto q = matrix_to_quat(R); // [x, y, z, w]
          bbox3d.center.orientation.x = q[0];
          bbox3d.center.orientation.y = q[1];
          bbox3d.center.orientation.z = q[2];
          bbox3d.center.orientation.w = q[3];
        }
      }
    }
  }

  return bbox3d;
}

yolo_msgs::msg::KeyPoint3DArray convert_keypoints_to_3d(
    const cv::Mat &depth_image, const sensor_msgs::msg::CameraInfo &depth_info,
    const yolo_msgs::msg::Detection &detection, int depth_units_divisor) {
  const auto &k = depth_info.k;
  const double fx = k[0], fy = k[4], px = k[2], py = k[5];

  yolo_msgs::msg::KeyPoint3DArray keypoints3d;
  for (const auto &kp : detection.keypoints.data) {
    // Clamp the 2D keypoint into the image (the row index is the y pixel).
    const int u = std::clamp(static_cast<int>(kp.point.y), 0,
                             static_cast<int>(depth_info.height) - 1);
    const int v = std::clamp(static_cast<int>(kp.point.x), 0,
                             static_cast<int>(depth_info.width) - 1);

    const double depth = depth_at_pixel(depth_image, u, v, depth_units_divisor);
    if (!std::isfinite(depth) || depth <= 0.0) {
      continue;
    }

    const double x = depth * (v - px) / fx;
    const double y = depth * (u - py) / fy;

    yolo_msgs::msg::KeyPoint3D kp3d;
    kp3d.id = kp.id;
    kp3d.score = kp.score;
    kp3d.point.x = x;
    kp3d.point.y = y;
    kp3d.point.z = depth;
    keypoints3d.data.push_back(kp3d);
  }

  return keypoints3d;
}

std::array<double, 3> qv_mult(const std::array<double, 4> &q,
                              const std::array<double, 3> &v) {
  const double qx = q[1], qy = q[2], qz = q[3], qw = q[0];

  // qvec = (qx, qy, qz); uv = qvec x v; uuv = qvec x uv
  const std::array<double, 3> uv{qy * v[2] - qz * v[1], qz * v[0] - qx * v[2],
                                 qx * v[1] - qy * v[0]};
  const std::array<double, 3> uuv{qy * uv[2] - qz * uv[1],
                                  qz * uv[0] - qx * uv[2],
                                  qx * uv[1] - qy * uv[0]};

  return {v[0] + 2.0 * (uv[0] * qw + uuv[0]),
          v[1] + 2.0 * (uv[1] * qw + uuv[1]),
          v[2] + 2.0 * (uv[2] * qw + uuv[2])};
}

yolo_msgs::msg::BoundingBox3D
transform_3d_box(const yolo_msgs::msg::BoundingBox3D &bbox,
                 const std::array<double, 3> &translation,
                 const std::array<double, 4> &rotation) {
  yolo_msgs::msg::BoundingBox3D out = bbox;

  // Position: rotate + translate.
  const auto position =
      qv_mult(rotation, {bbox.center.position.x, bbox.center.position.y,
                         bbox.center.position.z});
  out.center.position.x = position[0] + translation[0];
  out.center.position.y = position[1] + translation[1];
  out.center.position.z = position[2] + translation[2];

  // Orientation: compose the box orientation with the frame rotation. An
  // axis-aligned box (identity orientation in the source frame) keeps its
  // extents rotated as before; an oriented box carries its size in its own
  // local frame, so the size is left untouched and the composed quaternion
  // describes the rotation.
  const std::array<double, 4> q_bbox{
      out.center.orientation.w, out.center.orientation.x,
      out.center.orientation.y, out.center.orientation.z};
  if (quat_is_identity(q_bbox)) {
    const auto size =
        qv_mult(rotation, {bbox.size.x, bbox.size.y, bbox.size.z});
    out.size.x = std::abs(size[0]);
    out.size.y = std::abs(size[1]);
    out.size.z = std::abs(size[2]);
  } else {
    const auto q_new = quat_normalize(quat_multiply(rotation, q_bbox));
    out.center.orientation.x = q_new[1];
    out.center.orientation.y = q_new[2];
    out.center.orientation.z = q_new[3];
    out.center.orientation.w = q_new[0];
    out.size = bbox.size;
  }

  return out;
}

yolo_msgs::msg::KeyPoint3DArray
transform_3d_keypoints(const yolo_msgs::msg::KeyPoint3DArray &keypoints,
                       const std::array<double, 3> &translation,
                       const std::array<double, 4> &rotation) {
  yolo_msgs::msg::KeyPoint3DArray out = keypoints;

  for (auto &point : out.data) {
    const auto position =
        qv_mult(rotation, {point.point.x, point.point.y, point.point.z});
    point.point.x = position[0] + translation[0];
    point.point.y = position[1] + translation[1];
    point.point.z = position[2] + translation[2];
  }

  return out;
}

} // namespace yolo_ros::depth