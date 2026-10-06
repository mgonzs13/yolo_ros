// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2021 Yifu Zhang
// SPDX-License-Identifier: MIT

/// @file
/// @brief ByteTrack tracker: parameters, ROS parameter bridge and the tracker
/// implementation.

#ifndef YOLO_ROS__TRACKING__BYTE_TRACKER_HPP_
#define YOLO_ROS__TRACKING__BYTE_TRACKER_HPP_

#include <memory>
#include <vector>

#include "yolo_ros/tracking/strack.hpp"
#include "yolo_ros/tracking/tracker.hpp"
#include "yolo_ros/tracking/utils/kalman_filter.hpp"

/// @brief Forward declaration; ByteTrack ignores the frame argument.
namespace cv {
class Mat;
} // namespace cv

/// @addtogroup yolo_tracking
/// @{
namespace yolo_ros::tracking {

/// @brief ByteTrack configuration.
///
/// `type` is set to "bytetrack", the key used by the tracking node's
/// `tracker_type` parameter and by create_tracker().
struct ByteTrackParams : public TrackerParams {
  /// @brief Construct with `type = "bytetrack"` and the reference defaults.
  ByteTrackParams() { this->type = "bytetrack"; }
  /// @brief First-stage (high-score) association threshold.
  double track_high_thresh = 0.25; // first-stage match threshold
  /// @brief Second-stage low-score detection threshold.
  double track_low_thresh = 0.1; // second-stage low-score threshold
  /// @brief Minimum score required to start a new track.
  double new_track_thresh = 0.25; // min score to start a new track
  /// @brief Number of frames a lost track is kept alive.
  int track_buffer = 30; // frames a lost track is kept alive
  /// @brief Maximum association cost accepted as a match.
  double match_thresh = 0.8; // association cost threshold
  /// @brief Fuse the IoU cost with the detection score during matching.
  bool fuse_score = true; // fuse IoU cost with detection score
};

// --- ROS parameter bridge (tracker-specific) ------------------------------
// Each tracker owns the declaration/loading of its own ROS parameters
// through its specific functions, so the tracking node only dispatches by the
// `tracker_type` key and never accumulates every algorithm's knobs. The
// functions are templates on the node type for two reasons: rclcpp_lifecycle
// nodes do NOT derive from rclcpp::Node on Humble (so a concrete rclcpp::Node
// reference would not accept the tracking node), and the tracker layer stays
// free of rclcpp includes — the templates only call declare_parameter /
// get_parameter, which any node type provides.

/// @brief Declare the ByteTrack parameters on @p node with their defaults.
///
/// Only called for the tracker selected by the `tracker_type` parameter, so an
/// unselected tracker's knobs do not appear in `ros2 param list`.
/// @tparam NodeT Node type providing declare_parameter().
/// @param[in,out] node Node on which the parameters are declared.
template <typename NodeT> void declare_byte_track_params(NodeT &node) {
  node.template declare_parameter<double>("track_high_thresh", 0.25);
  node.template declare_parameter<double>("track_low_thresh", 0.1);
  node.template declare_parameter<double>("new_track_thresh", 0.25);
  node.template declare_parameter<int>("track_buffer", 30);
  node.template declare_parameter<double>("match_thresh", 0.8);
  node.template declare_parameter<bool>("fuse_score", true);
}

/// @brief Read the already-declared ByteTrack parameters from @p node.
/// @tparam NodeT Node type providing get_parameter().
/// @param[in] node Node whose parameters are read.
/// @return The params struct ready for create_tracker().
template <typename NodeT>
ByteTrackParams load_byte_track_params(const NodeT &node) {
  ByteTrackParams params;
  node.get_parameter("track_high_thresh", params.track_high_thresh);
  node.get_parameter("track_low_thresh", params.track_low_thresh);
  node.get_parameter("new_track_thresh", params.new_track_thresh);
  node.get_parameter("track_buffer", params.track_buffer);
  node.get_parameter("match_thresh", params.match_thresh);
  node.get_parameter("fuse_score", params.fuse_score);
  return params;
}

/// @brief ByteTrack implementation based on the original paper and
/// MIT-licensed reference implementation. See THIRD_PARTY_NOTICES.md.
class ByteTrack : public Tracker {
public:
  /// @brief Bring the one-argument Tracker::update overload into scope.
  using Tracker::update;

  /// @brief Create the tracker from @p params.
  /// @param params ByteTrack tuning parameters.
  explicit ByteTrack(const ByteTrackParams &params);
  /// @brief Destroy the tracker.
  ~ByteTrack() override = default;

  /// @brief Advance the tracker one frame.
  ///
  /// @p detections should already be NMS-filtered (the tracker re-splits them
  /// by confidence into high / low stages).
  /// @param[in] detections NMS-filtered detections for the current frame.
  /// @param[in] frame Unused; ByteTrack has no camera-motion compensation.
  /// @return The currently active (activated) tracked objects.
  std::vector<Track> update(const std::vector<TrackDetection> &detections,
                            const cv::Mat &frame) override;

  /// @brief Clear all track state and the track-id counter.
  void reset() override;

  /// @brief Current internal frame counter.
  /// @return Number of frames processed since construction/reset().
  int frame_id() const { return this->frame_id_; }

private:
  /// @brief Configuration copied at construction.
  ByteTrackParams params_;
  /// @brief Kalman filter used to predict/refine track boxes.
  utils::KalmanFilterXYAH kalman_filter_;
  /// @brief Currently tracked (activated) tracks.
  std::vector<std::shared_ptr<STrack>> tracked_stracks_;
  /// @brief Tracks temporarily lost but kept alive for re-association.
  std::vector<std::shared_ptr<STrack>> lost_stracks_;
  /// @brief Recently removed tracks, kept to avoid id reuse.
  std::vector<std::shared_ptr<STrack>> removed_stracks_;
  /// @brief Internal frame counter.
  int frame_id_ = 0;
  /// @brief Cap on removed_stracks_, trimming the oldest entries.
  static constexpr std::size_t kRemovedBuffer = 1000; // cap on removed_stracks_
};

} // namespace yolo_ros::tracking
/// @}

#endif // YOLO_ROS__TRACKING__BYTE_TRACKER_HPP_
