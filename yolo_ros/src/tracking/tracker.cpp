// Copyright (C) 2026 Alejandro González Cantón
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

#include "yolo_ros/tracking/tracker.hpp"

#include <algorithm>
#include <cctype>
#include <string>

#include <opencv2/core.hpp>

#include "yolo_ros/tracking/bot_sort.hpp"
#include "yolo_ros/tracking/byte_tracker.hpp"
#include "yolo_ros/utils/string_utils.hpp"

namespace yolo_ros::tracking {

std::vector<Track>
Tracker::update(const std::vector<TrackDetection> &detections) {
  return this->update(detections, cv::Mat{});
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
  } else if (type == "botsort") {
    const auto *bot_params = dynamic_cast<const BotSortParams *>(&params);

    if (bot_params == nullptr) {
      return nullptr;
    }

    return std::make_unique<BotSort>(*bot_params);
  }

  return nullptr;
}

} // namespace yolo_ros::tracking
