// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2023-2025 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief Robust depth statistics and 2D to 3D lifting for detection boxes and
/// pose keypoints.

#ifndef YOLO_ROS__3D__DEPTH_UTILS_HPP_
#define YOLO_ROS__3D__DEPTH_UTILS_HPP_

#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "sensor_msgs/msg/camera_info.hpp"
#include "yolo_msgs/msg/bounding_box3_d.hpp"
#include "yolo_msgs/msg/detection.hpp"
#include "yolo_msgs/msg/key_point3_d_array.hpp"

/// @addtogroup yolo_3d
/// @{
namespace yolo_ros::depth {

/// @brief A 3D point in the depth (camera) frame.
using Point3 = std::array<double, 3>;

// ---------------------------------------------------------------------------
// Orientation-estimation options, ported from the upstream detect_3d_node.py
// (_sample_points_3d, _plane_frame_from_pts_pca, _consistent_axes,
// _weighted_percentiles). When enabled, each 3D box becomes an oriented
// bounding box (OBB): a PCA plane frame is fit to a strided sample of the
// object's depth points, the box extents are recomputed along those axes and
// the box orientation is published as a quaternion.
// ---------------------------------------------------------------------------
/// @brief Options controlling oriented-bounding-box estimation.
struct OrientationParams {
  /// @brief Enable PCA-based orientation estimation.
  bool enable = false;
  /// @brief Minimum valid depth points required to attempt orientation.
  int min_seg_points_for_orientation = 20;
};

/// @brief Temporal sign-consistency cache for the PCA in-plane axes, keyed by
/// the detection track id (_consistent_axes).
///
/// PCA has a 180-degree sign ambiguity that would otherwise flip the
/// orientation frame from frame to frame.
struct OrientationState {
  /// @brief Last accepted in-plane axes per track id.
  std::unordered_map<std::string, Point3> last_axes;
};

// ---------------------------------------------------------------------------
// Robust depth-statistics helpers, ported from the Python detect_3d_node.py
// (_compute_spatial_weights, _compute_depth_bounds_weighted,
// _compute_height_bounds, _compute_width_bounds). They turn the raw depth
// samples of a detection into a 3D box: a spatially weighted, trimmed-mean
// center and MAD/percentile-based extents per axis.
// ---------------------------------------------------------------------------

/// @brief Median of a copy of @p v (nth_element-based, O(N)).
/// @param[in] v Values (taken by value; reordered internally).
/// @return The median.
double median(std::vector<double> v);

/// @brief Linear-interpolation percentile on an already-sorted vector
/// (q in [0,1]), matching numpy.percentile(..., method='linear').
/// @param[in] v Sorted values.
/// @param[in] q Quantile in [0, 1].
/// @return The interpolated percentile.
double percentile_sorted(const std::vector<double> &v, double q);

/// @brief Weighted mean of @p v with weights @p w.
/// @param[in] v Values.
/// @param[in] w Per-value weights.
/// @return The weighted mean.
double weighted_mean(const std::vector<double> &v,
                     const std::vector<double> &w);

/// @brief Gaussian falloff from the bbox center, floored at 0.3
/// (Python _compute_spatial_weights).
/// @param[in] xs Pixel x coordinates.
/// @param[in] ys Pixel y coordinates.
/// @param[in] center_x Bounding-box center x.
/// @param[in] center_y Bounding-box center y.
/// @param[in] size_x Bounding-box width.
/// @param[in] size_y Bounding-box height.
/// @return One spatial weight per input point.
std::vector<double> compute_spatial_weights(const std::vector<int> &xs,
                                            const std::vector<int> &ys,
                                            int center_x, int center_y,
                                            int size_x, int size_y);

/// @brief Depth extent along the camera z axis.
struct DepthBounds {
  /// @brief Robust center depth (metres).
  double center;
  /// @brief Lower depth bound (metres).
  double min;
  /// @brief Upper depth bound (metres).
  double max;
};

/// @brief Weighted histogram peak + MAD/IQR adaptive filtering + trimmed
/// weighted center and 1st/99th weighted percentiles (Python
/// _compute_depth_bounds_weighted).
/// @param[in] depth Depth samples.
/// @param[in] weight Per-sample weights.
/// @return The robust depth bounds.
DepthBounds compute_depth_bounds_weighted(std::vector<double> depth,
                                          std::vector<double> weight);

/// @brief Extent along one lateral 3D axis.
struct AxisBounds {
  /// @brief Robust center coordinate.
  double center;
  /// @brief Lower bound.
  double min;
  /// @brief Upper bound.
  double max;
};

/// @brief Outlier-filtered (MAD), trimmed weighted center + 3rd/97th weighted
/// percentiles for one 3D axis (Python _compute_height_bounds /
/// _compute_width_bounds).
/// @param[in] val3 Coordinate values for the axis.
/// @param[in] w Per-value weights.
/// @param[in] mad_mult MAD multiplier for outlier rejection.
/// @param[in] clip_lo Lower clip quantile.
/// @param[in] clip_hi Upper clip quantile.
/// @return The robust axis bounds.
AxisBounds compute_axis_bounds(const std::vector<double> &val3,
                               const std::vector<double> &w, double mad_mult,
                               double clip_lo, double clip_hi);

/// @brief Raw depth at a pixel (row @p v, col @p u).
///
/// 16UC1 raw units are divided by @p depth_units_divisor to get metres;
/// 32FC1 is already in metres.
/// @param[in] depth_image Depth image (CV_16UC1 or CV_32FC1).
/// @param[in] v Row index.
/// @param[in] u Column index.
/// @param[in] depth_units_divisor Divisor applied to 16UC1 values.
/// @return Depth in metres (0 for invalid/out-of-range pixels).
double depth_at_pixel(const cv::Mat &depth_image, int v, int u,
                      int depth_units_divisor);

/// @brief Lift a 2D detection into a BoundingBox3D in the depth camera frame.
///
/// Uses the bbox (and optionally the segmentation mask) to sample depth, the
/// camera intrinsics and the robust depth statistics above. When
/// `orient_params.enable` is set, the box becomes an oriented bounding box
/// (OBB): a PCA plane frame is fit to a strided depth sample of the object, the
/// extents are recomputed along those axes and the orientation is published as
/// a quaternion. @p orient_state carries the per-track sign-consistency cache
/// across frames.
/// @param[in] depth_image Depth image.
/// @param[in] depth_info Camera intrinsics.
/// @param[in] detection 2D detection to lift.
/// @param[in] depth_units_divisor Divisor applied to 16UC1 depth values.
/// @param[in] orient_params Orientation-estimation options.
/// @param[in,out] orient_state Per-track sign-consistency cache (may be null
/// when orientation is disabled).
/// @return The 3D box, or std::nullopt when no valid depth is available.
std::optional<yolo_msgs::msg::BoundingBox3D> convert_bb_to_3d(
    const cv::Mat &depth_image, const sensor_msgs::msg::CameraInfo &depth_info,
    const yolo_msgs::msg::Detection &detection, int depth_units_divisor,
    const OrientationParams &orient_params = OrientationParams(),
    OrientationState *orient_state = nullptr);

/// @brief Back-project the 2D pose keypoints of a detection into 3D (depth
/// camera frame), keeping id/score; keypoints with no valid depth are skipped.
/// @param[in] depth_image Depth image.
/// @param[in] depth_info Camera intrinsics.
/// @param[in] detection Detection carrying the 2D keypoints.
/// @param[in] depth_units_divisor Divisor applied to 16UC1 depth values.
/// @return The 3D keypoints.
yolo_msgs::msg::KeyPoint3DArray convert_keypoints_to_3d(
    const cv::Mat &depth_image, const sensor_msgs::msg::CameraInfo &depth_info,
    const yolo_msgs::msg::Detection &detection, int depth_units_divisor);

/// @brief Quaternion-vector rotation: `v' = q v q^-1` (q = [w, x, y, z]).
/// @param[in] q Unit quaternion (w, x, y, z).
/// @param[in] v Vector to rotate.
/// @return The rotated vector.
std::array<double, 3> qv_mult(const std::array<double, 4> &q,
                              const std::array<double, 3> &v);

/// @brief Apply a rigid transform (translation + rotation) to a 3D box.
///
/// The position is rotated and translated, the orientation quaternion is
/// composed with the frame rotation. An axis-aligned box (identity orientation
/// in the source frame) keeps identity and its extents are rotated; an oriented
/// box keeps its local-frame size (the composed orientation carries the
/// rotation).
/// @param[in] bbox Box to transform.
/// @param[in] translation Frame translation.
/// @param[in] rotation Frame rotation as (w, x, y, z).
/// @return The transformed box.
yolo_msgs::msg::BoundingBox3D
transform_3d_box(const yolo_msgs::msg::BoundingBox3D &bbox,
                 const std::array<double, 3> &translation,
                 const std::array<double, 4> &rotation);

/// @brief Apply a rigid transform (translation + rotation) to 3D keypoints.
/// @param[in] keypoints Keypoints to transform.
/// @param[in] translation Frame translation.
/// @param[in] rotation Frame rotation as (w, x, y, z).
/// @return The transformed keypoints.
yolo_msgs::msg::KeyPoint3DArray
transform_3d_keypoints(const yolo_msgs::msg::KeyPoint3DArray &keypoints,
                       const std::array<double, 3> &translation,
                       const std::array<double, 4> &rotation);

} // namespace yolo_ros::depth
/// @}

#endif // YOLO_ROS__3D__DEPTH_UTILS_HPP_
