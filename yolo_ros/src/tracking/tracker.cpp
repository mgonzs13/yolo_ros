// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include "yolo_ros/tracking/tracker.hpp"

#include <algorithm>
#include <cctype>
#include <string>

#include "yolo_ros/tracking/bot_sort.hpp"
#include "yolo_ros/tracking/byte_tracker.hpp"
#include "yolo_ros/utils/string_utils.hpp"

#include <opencv2/core.hpp>

namespace yolo_ros::tracking {

std::vector<Track>
Tracker::update(const std::vector<TrackDetection> &detections) {
  return update(detections, cv::Mat{});
}

std::unique_ptr<Tracker> create_tracker(const TrackerParams &params) {
  const std::string type = yolo_ros::utils::to_lower(params.type);

  // --- add new trackers here ---------------------------------------------
  // Register the key accepted by the tracking node's `tracker_type` parameter
  // and construct your tracker from its own params struct (derived from
  // TrackerParams). The node additionally dispatches to the tracker's own
  // parameter functions — declare_<tracker>_params / load_<tracker>_params
  // (see byte_tracker.hpp) — so each algorithm owns its config surface;
  // nothing else in the pipeline changes.
  if (type == "bytetrack") {
    // Guard the dynamic_cast: nullptr means the caller passed the wrong params
    // struct for the requested tracker — report it via the nullptr return
    // instead of throwing std::bad_cast.
    const auto *byte_params = dynamic_cast<const ByteTrackParams *>(&params);
    if (byte_params == nullptr) {
      return nullptr;
    }
    return std::make_unique<ByteTrack>(*byte_params);
  }

  if (type == "botsort") {
    const auto *bot_params = dynamic_cast<const BotSortParams *>(&params);
    if (bot_params == nullptr) {
      return nullptr;
    }
    return std::make_unique<BotSort>(*bot_params);
  }

  return nullptr;
}

} // namespace yolo_ros::tracking