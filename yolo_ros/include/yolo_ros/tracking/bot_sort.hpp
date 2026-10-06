// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2021 Yifu Zhang
// Portions Copyright (c) 2022 Nir Aharon
// SPDX-License-Identifier: MIT

/// @file
/// @brief BoT-SORT tracker (XYWH Kalman filter + camera-motion compensation,
/// plus an optional ReID appearance branch): parameters, ROS parameter bridge
/// and the tracker implementation.

#ifndef YOLO_ROS__TRACKING__BOT_SORT_HPP_
#define YOLO_ROS__TRACKING__BOT_SORT_HPP_

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "yolo_ros/tracking/strack.hpp"
#include "yolo_ros/tracking/tracker.hpp"
#include "yolo_ros/tracking/utils/camera_motion.hpp"
#include "yolo_ros/tracking/utils/kalman_filter.hpp"

/// @brief Forward declaration: BotSort owns an ONNX encoder but this header
/// stays free of ONNX Runtime includes.
namespace yolo_ros::engine {
class ReIDEncoder;
} // namespace yolo_ros::engine

/// @addtogroup yolo_tracking
/// @{
namespace yolo_ros::tracking {

/// @brief BoT-SORT configuration (optional ReID appearance branch selected by
/// `with_reid`).
///
/// `type` is set to "botsort", the key used by the tracking node's
/// `tracker_type` parameter and by create_tracker(). The association knobs
/// share ByteTrack's names/defaults; `gmc_method`/`gmc_downscale` drive the
/// camera-motion compensator.
struct BotSortParams : public TrackerParams {
  /// @brief Construct with `type = "botsort"`.
  BotSortParams() { type = "botsort"; }
  /// @brief First-stage (high-score) association threshold.
  double track_high_thresh = 0.25;
  /// @brief Second-stage low-score detection threshold.
  double track_low_thresh = 0.1;
  /// @brief Minimum score required to start a new track.
  double new_track_thresh = 0.25;
  /// @brief Number of frames a lost track is kept alive.
  int track_buffer = 30;
  /// @brief Maximum association cost accepted as a match.
  double match_thresh = 0.8;
  /// @brief Fuse the IoU cost with the detection score during matching.
  bool fuse_score = true;
  /// @brief Camera-motion method: "none", "sparseOptFlow", "orb" or "ecc".
  std::string gmc_method = "none";
  /// @brief Camera-motion downscale factor (>= 1).
  int gmc_downscale = 2;
  /// @brief Enable the ReID appearance association branch.
  bool with_reid = false;
  /// @brief IoU distance above which appearance is ignored during matching.
  double proximity_thresh = 0.5;
  /// @brief Cap on the embedding distance accepted as a match.
  double appearance_thresh = 0.25;
  /// @brief ONNX ReID encoder path (empty disables appearance even when
  /// with_reid is set).
  std::string reid_model = "";
  /// @brief Execution provider for the ReID encoder.
  std::string provider = "auto";
  /// @brief Device ordinal for the ReID encoder.
  std::string device = "cuda:0";
};

// --- ROS parameter bridge (tracker-specific) ------------------------------
// See byte_tracker.hpp for the rationale: each tracker owns the declaration and
// loading of its own parameters, templated on the node type so the tracking
// layer stays free of rclcpp includes.

/// @brief Declare the BoT-SORT parameters on @p node with their defaults.
/// @tparam NodeT Node type providing declare_parameter().
/// @param[in,out] node Node on which the parameters are declared.
template <typename NodeT> void declare_bot_sort_params(NodeT &node) {
  node.template declare_parameter<double>("track_high_thresh", 0.25);
  node.template declare_parameter<double>("track_low_thresh", 0.1);
  node.template declare_parameter<double>("new_track_thresh", 0.25);
  node.template declare_parameter<int>("track_buffer", 30);
  node.template declare_parameter<double>("match_thresh", 0.8);
  node.template declare_parameter<bool>("fuse_score", true);
  node.template declare_parameter<std::string>("gmc_method", "none");
  node.template declare_parameter<int>("gmc_downscale", 2);
  node.template declare_parameter<bool>("with_reid", false);
  node.template declare_parameter<std::string>("reid_model", "");
  node.template declare_parameter<double>("proximity_thresh", 0.5);
  node.template declare_parameter<double>("appearance_thresh", 0.25);
  node.template declare_parameter<std::string>("provider", "auto");
  node.template declare_parameter<std::string>("device", "cuda:0");
}

/// @brief Read the already-declared BoT-SORT parameters from @p node.
/// @tparam NodeT Node type providing get_parameter().
/// @param[in] node Node whose parameters are read.
/// @return The params struct ready for create_tracker().
template <typename NodeT>
BotSortParams load_bot_sort_params(const NodeT &node) {
  BotSortParams params;
  node.get_parameter("track_high_thresh", params.track_high_thresh);
  node.get_parameter("track_low_thresh", params.track_low_thresh);
  node.get_parameter("new_track_thresh", params.new_track_thresh);
  node.get_parameter("track_buffer", params.track_buffer);
  node.get_parameter("match_thresh", params.match_thresh);
  node.get_parameter("fuse_score", params.fuse_score);
  node.get_parameter("gmc_method", params.gmc_method);
  node.get_parameter("gmc_downscale", params.gmc_downscale);
  node.get_parameter("with_reid", params.with_reid);
  node.get_parameter("reid_model", params.reid_model);
  node.get_parameter("proximity_thresh", params.proximity_thresh);
  node.get_parameter("appearance_thresh", params.appearance_thresh);
  node.get_parameter("provider", params.provider);
  node.get_parameter("device", params.device);
  return params;
}

/// @brief BoT-SORT implementation based on the reference BoT-SORT tracker
/// (MIT). See THIRD_PARTY_NOTICES.md. Optional ReID appearance branch
/// selected by `with_reid`; otherwise motion association only.
class BotSort : public Tracker {
public:
  /// @brief Bring the one-argument Tracker::update overload into scope.
  using Tracker::update;

  /// @brief Create the tracker from @p params.
  /// @param params BoT-SORT tuning parameters.
  explicit BotSort(const BotSortParams &params);
  /// @brief Destroy the tracker (encoder is an incomplete type here, so the
  /// destructor is defined in the .cpp).
  ~BotSort() override;

  /// @brief Advance the tracker one frame.
  /// @param[in] detections NMS-filtered detections for the current frame.
  /// @param[in] frame Current camera frame; used only when the camera-motion
  /// method is enabled.
  /// @return The currently tracked objects.
  std::vector<Track> update(const std::vector<TrackDetection> &detections,
                            const cv::Mat &frame) override;

  /// @brief Whether update() consumes the frame argument.
  /// @return True when a camera-motion method other than "none" is enabled.
  bool needs_frame() const override;

  /// @brief Clear all track state, the track-id counter and the CMC state.
  void reset() override;

  /// @brief Current internal frame counter.
  /// @return Number of frames processed since construction/reset().
  int frame_id() const { return frame_id_; }

private:
  /// @brief Configuration copied at construction.
  BotSortParams params_;
  /// @brief Optional ONNX ReID encoder (null when appearance is disabled).
  std::unique_ptr<engine::ReIDEncoder> reid_;
  /// @brief Whether the ReID inference-failure warning was already emitted.
  bool reid_warned_ = false;
  /// @brief Kalman filter with the XYWH state.
  utils::KalmanFilterXYWH kalman_filter_;
  /// @brief Camera-motion estimator.
  utils::CameraMotionCompensator cmc_;
  /// @brief Currently tracked (activated) tracks.
  std::vector<std::shared_ptr<STrack>> tracked_stracks_;
  /// @brief Tracks temporarily lost but kept alive for re-association.
  std::vector<std::shared_ptr<STrack>> lost_stracks_;
  /// @brief Recently removed tracks, kept to avoid id reuse.
  std::vector<std::shared_ptr<STrack>> removed_stracks_;
  /// @brief Internal frame counter.
  int frame_id_ = 0;
  /// @brief Cap on removed_stracks_, trimming the oldest entries.
  static constexpr std::size_t kRemovedBuffer = 1000;
};

} // namespace yolo_ros::tracking
/// @}

#endif // YOLO_ROS__TRACKING__BOT_SORT_HPP_
