// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2022 Nir Aharon
// SPDX-License-Identifier: MIT

/// @file
/// @brief Global camera-motion estimation for BoT-SORT (sparse optical flow,
/// ORB and ECC), ported from the BoT-SORT reference tracker/gmc.py.

#ifndef YOLO_ROS__TRACKING__UTILS__CAMERA_MOTION_HPP_
#define YOLO_ROS__TRACKING__UTILS__CAMERA_MOTION_HPP_

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "yolo_ros/tracking/utils/kalman_filter.hpp"

/// @addtogroup yolo_tracking
/// @{
namespace yolo_ros::tracking::utils {

/// @brief Estimates the inter-frame affine warp used to compensate camera
/// motion before association.
///
/// Stateful: keeps the previous frame (and, for feature methods, its keypoints
/// and descriptors). The caller feeds each frame once, in order.
class CameraMotionCompensator {
public:
  /// @brief Create the estimator for @p method.
  /// @param[in] method "none", "sparseOptFlow", "orb" or "ecc"
  /// (case-insensitive); an unknown value warns and behaves as "none".
  /// @param[in] downscale Integer downscale factor (clamped to >= 1).
  explicit CameraMotionCompensator(const std::string &method,
                                   int downscale = 2);

  /// @brief Estimate the warp mapping the previous frame onto @p frame.
  /// @param[in] frame Current BGR or gray frame; an empty frame returns the
  /// identity without updating the internal state.
  /// @return The 2x3 affine warp (identity on the first frame or on failure).
  KalmanAffine apply(const cv::Mat &frame);

  /// @brief Whether the estimator performs any work.
  /// @return True unless the method is "none".
  bool enabled() const { return method_ != "none"; }

  /// @brief Normalized method name.
  /// @return "none", "sparseoptflow", "orb" or "ecc".
  const std::string &method() const { return method_; }

  /// @brief Drop the previous-frame state.
  void reset();

private:
  /// @brief Sparse optical flow (reference applySparseOptFlow).
  KalmanAffine apply_sparse_opt_flow(const cv::Mat &frame);
  /// @brief Feature matching (reference applyFeaures).
  KalmanAffine apply_features(const cv::Mat &frame);
  /// @brief Dense ECC alignment (reference applyEcc).
  KalmanAffine apply_ecc(const cv::Mat &frame);

  /// @brief Normalized (lowercase) method key.
  std::string method_;
  /// @brief Downscale factor (>= 1).
  int downscale_;
  /// @brief Previous frame, gray and downscaled.
  cv::Mat prev_frame_;
  /// @brief Previous sparse optical-flow points (sparseOptFlow).
  std::vector<cv::Point2f> prev_points_;
  /// @brief Previous keypoints (orb).
  std::vector<cv::KeyPoint> prev_keypoints_;
  /// @brief Previous descriptors (orb).
  cv::Mat prev_descriptors_;
  /// @brief Whether a previous frame has been stored.
  bool initialized_ = false;

  /// @brief ORB feature detector.
  cv::Ptr<cv::FeatureDetector> detector_;
  /// @brief ORB descriptor extractor.
  cv::Ptr<cv::Feature2D> extractor_;
  /// @brief ORB descriptor matcher (Hamming).
  cv::Ptr<cv::DescriptorMatcher> matcher_;
  /// @brief ECC termination criteria.
  cv::TermCriteria ecc_criteria_;
};

} // namespace yolo_ros::tracking::utils
/// @}

#endif // YOLO_ROS__TRACKING__UTILS__CAMERA_MOTION_HPP_
