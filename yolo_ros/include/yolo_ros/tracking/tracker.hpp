// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

/// @file
/// @brief Tracker-agnostic interface: detection input, tracked-object output,
/// base configuration and the create_tracker() factory.

#ifndef YOLO_ROS__TRACKING__TRACKER_HPP_
#define YOLO_ROS__TRACKING__TRACKER_HPP_

#include <memory>
#include <string>
#include <vector>

/// @brief cv::Mat is only used as a reference in this header; the definition is
/// pulled in by the implementations that actually touch the frame.
namespace cv {
class Mat;
} // namespace cv

/// @addtogroup yolo_tracking
/// @{
namespace yolo_ros::tracking {

/// @brief One raw detection fed to a tracker.
///
/// Tracker-agnostic: every tracker implementation consumes the same input
/// shape, so the tracking node can switch algorithms without converting to a
/// per-tracker format.
struct TrackDetection {
  float cx = 0;     ///< Bounding box center x.
  float cy = 0;     ///< Bounding box center y.
  float w = 0;      ///< Width.
  float h = 0;      ///< Height.
  float score = 0;  ///< Detection confidence in [0, 1].
  int class_id = 0; ///< Zero-based class id.
  int index = 0;    ///< Position in the original detection array (metadata).
};

/// @brief One tracked object returned by Tracker::update().
struct Track {
  int id = 0;       ///< Stable track id.
  float x1 = 0;     ///< Tracker-refined left x.
  float y1 = 0;     ///< Tracker-refined top y.
  float x2 = 0;     ///< Tracker-refined right x.
  float y2 = 0;     ///< Tracker-refined bottom y.
  float score = 0;  ///< Detection confidence in [0, 1].
  int class_id = 0; ///< Zero-based class id.
  int index = 0;    ///< Detection index (from TrackDetection::index).
};

/// @brief Base tracker configuration.
///
/// `type` is the canonical tracker key — the same string the tracking node's
/// `tracker_type` parameter accepts (e.g. "bytetrack") — and create_tracker()
/// dispatches on it. New trackers derive their own params struct from this one
/// and set `type` in their constructor; the derived struct holds the
/// algorithm's tuning knobs.
struct TrackerParams {
  /// @brief Virtual destructor (enables derived parameter structs).
  virtual ~TrackerParams() = default;
  /// @brief Canonical tracker key used by create_tracker().
  std::string type;
};

/// @brief Abstract multi-object tracker interface.
///
/// The tracking node (and nothing else in the package) interacts with trackers
/// through this interface, so a new tracker implementation never touches the
/// node, the message flow or the downstream topics.
class Tracker {
public:
  /// @brief Virtual destructor.
  virtual ~Tracker() = default;

  /// @brief Advance the tracker one frame.
  ///
  /// @p detections should already be NMS-filtered (implementations re-split
  /// them by confidence as their algorithm requires).
  /// @param[in] detections NMS-filtered detections for the current frame.
  /// @param[in] frame Current camera frame, empty when the tracker does not
  /// consume it (see needs_frame()). Valid only for this call.
  /// @return The currently active (activated) tracked objects.
  virtual std::vector<Track>
  update(const std::vector<TrackDetection> &detections,
         const cv::Mat &frame) = 0;

  /// @brief Advance the tracker one frame without an image.
  ///
  /// Convenience overload for trackers that do not need the frame; forwards an
  /// empty cv::Mat to update().
  /// @param[in] detections NMS-filtered detections for the current frame.
  /// @return The currently active (activated) tracked objects.
  std::vector<Track> update(const std::vector<TrackDetection> &detections);

  /// @brief Whether update() consumes the frame argument.
  /// @return True for trackers that use camera-motion compensation.
  virtual bool needs_frame() const { return false; }

  /// @brief Clear all track state and the track-id counter.
  virtual void reset() = 0;
};

/// @brief Build the tracker named by `params.type` (case-insensitive).
/// @param[in] params Tracker parameters; its dynamic type selects the concrete
/// tracker.
/// @return The tracker, or nullptr for unknown types or when @p params is not
/// the params struct of the named tracker, so callers can fall back gracefully
/// instead of crashing.
std::unique_ptr<Tracker> create_tracker(const TrackerParams &params);

} // namespace yolo_ros::tracking
/// @}

#endif // YOLO_ROS__TRACKING__TRACKER_HPP_
