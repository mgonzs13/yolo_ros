// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2021 Yifu Zhang
// Portions Copyright (c) 2022 Nir Aharon
// SPDX-License-Identifier: MIT

/// @file
/// @brief Kalman filters for the trackers: a common 8-dimensional
/// constant-velocity filter interface plus the XYAH (ByteTrack) and XYWH
/// (BoT-SORT) box states.

#ifndef YOLO_ROS__TRACKING__UTILS__KALMAN_FILTER_HPP_
#define YOLO_ROS__TRACKING__UTILS__KALMAN_FILTER_HPP_

#include <array>
#include <utility>

/// @addtogroup yolo_tracking
/// @{
namespace yolo_ros::tracking::utils {

/// @brief Kalman state mean: x, y, (aspect | width), height + velocities.
using KalmanMean = std::array<double, 8>; // x, y, s1, s2, vx, vy, vs1, vs2
/// @brief Kalman state covariance (8x8).
using KalmanCovariance = std::array<std::array<double, 8>, 8>;
/// @brief Measurement vector in projected (box) space.
using KalmanMeasurement = std::array<double, 4>;
/// @brief Covariance projected into measurement space (4x4).
using KalmanProjectedCov = std::array<std::array<double, 4>, 4>;

/// @brief 2x3 affine warp (image space) used for camera-motion compensation.
///
/// `r` is the 2x2 linear part (rotation/scale) and `t` the translation. The
/// defaults make this the identity warp.
struct KalmanAffine {
  /// @brief 2x2 linear part.
  double r[2][2] = {{1.0, 0.0}, {0.0, 1.0}};
  /// @brief Translation.
  double t[2] = {0.0, 0.0};
};

/// @brief Abstract 8-dimensional constant-velocity Kalman filter.
///
/// The state is (x, y, s1, s2, vx, vy, vs1, vs2) with a linear observation
/// model. Concrete filters define the box <-> measurement mapping (`s1`/`s2`
/// are the aspect/height pair for XYAH and the width/height pair for XYWH) and
/// the noise weights. update() is shared and dispatches to the virtual
/// project().
class KalmanFilter {
public:
  /// @brief Virtual destructor.
  virtual ~KalmanFilter() = default;

  /// @brief Initialize a state from a first measurement.
  /// @param[in] measurement Box measured in projected space.
  /// @return The initial (mean, covariance) pair.
  virtual std::pair<KalmanMean, KalmanCovariance>
  initiate(const KalmanMeasurement &measurement) const = 0;

  /// @brief Predict the state one time step forward.
  /// @param[in] mean Current state mean.
  /// @param[in] covariance Current state covariance.
  /// @return The predicted (mean, covariance) pair.
  virtual std::pair<KalmanMean, KalmanCovariance>
  predict(const KalmanMean &mean, const KalmanCovariance &covariance) const = 0;

  /// @brief Project the state distribution into measurement (box) space.
  /// @param[in] mean Current state mean.
  /// @param[in] covariance Current state covariance.
  /// @param[out] projected_mean Mean in projected space.
  /// @param[out] projected_covariance 4x4 covariance in projection space.
  virtual void project(const KalmanMean &mean,
                       const KalmanCovariance &covariance,
                       KalmanMeasurement &projected_mean,
                       KalmanProjectedCov &projected_covariance) const = 0;

  /// @brief Correct the state with a new measurement (shared implementation).
  /// @param[in] mean Predicted state mean.
  /// @param[in] covariance Predicted state covariance.
  /// @param[in] measurement Observed box in projected space.
  /// @return The updated (mean, covariance) pair.
  std::pair<KalmanMean, KalmanCovariance>
  update(const KalmanMean &mean, const KalmanCovariance &covariance,
         const KalmanMeasurement &measurement) const;

  /// @brief Convert a tlwh box into the measurement vector.
  /// @param[in] tlwh Box as top-left x, top-left y, width, height.
  /// @return The measurement vector.
  virtual KalmanMeasurement
  box_to_measurement(const std::array<float, 4> &tlwh) const = 0;

  /// @brief Convert a state mean into a tlwh box.
  /// @param[in] mean State mean.
  /// @return The tlwh box.
  virtual std::array<float, 4>
  measurement_to_tlwh(const KalmanMean &mean) const = 0;
};

/// @brief Kalman filter with the (center-x, center-y, aspect, height) state.
///
/// The ByteTrack reference model and uncertainty weights.
class KalmanFilterXYAH : public KalmanFilter {
public:
  std::pair<KalmanMean, KalmanCovariance>
  initiate(const KalmanMeasurement &measurement) const override;
  std::pair<KalmanMean, KalmanCovariance>
  predict(const KalmanMean &mean,
          const KalmanCovariance &covariance) const override;
  void project(const KalmanMean &mean, const KalmanCovariance &covariance,
               KalmanMeasurement &projected_mean,
               KalmanProjectedCov &projected_covariance) const override;
  KalmanMeasurement
  box_to_measurement(const std::array<float, 4> &tlwh) const override;
  std::array<float, 4>
  measurement_to_tlwh(const KalmanMean &mean) const override;

private:
  /// @brief Position uncertainty weight (1/20).
  static constexpr double kStdWeightPosition = 1.0 / 20;
  /// @brief Velocity uncertainty weight (1/160).
  static constexpr double kStdWeightVelocity = 1.0 / 160;
};

/// @brief Kalman filter with the (center-x, center-y, width, height) state.
///
/// Used by BoT-SORT; ported from the reference tracker/kalman_filter.py
/// (width/height noise, no NSA scaling).
class KalmanFilterXYWH : public KalmanFilter {
public:
  std::pair<KalmanMean, KalmanCovariance>
  initiate(const KalmanMeasurement &measurement) const override;
  std::pair<KalmanMean, KalmanCovariance>
  predict(const KalmanMean &mean,
          const KalmanCovariance &covariance) const override;
  void project(const KalmanMean &mean, const KalmanCovariance &covariance,
               KalmanMeasurement &projected_mean,
               KalmanProjectedCov &projected_covariance) const override;
  KalmanMeasurement
  box_to_measurement(const std::array<float, 4> &tlwh) const override;
  std::array<float, 4>
  measurement_to_tlwh(const KalmanMean &mean) const override;

private:
  /// @brief Position uncertainty weight (1/20).
  static constexpr double kStdWeightPosition = 1.0 / 20;
  /// @brief Velocity uncertainty weight (1/160).
  static constexpr double kStdWeightVelocity = 1.0 / 160;
};

} // namespace yolo_ros::tracking::utils
/// @}

#endif // YOLO_ROS__TRACKING__UTILS__KALMAN_FILTER_HPP_
