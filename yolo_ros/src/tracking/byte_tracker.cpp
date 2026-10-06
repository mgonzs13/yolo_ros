// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2021 Yifu Zhang
// SPDX-License-Identifier: MIT

#include "yolo_ros/tracking/byte_tracker.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <utility>

#include "yolo_ros/tracking/utils/matching.hpp"

namespace yolo_ros::tracking {

using namespace utils;

ByteTrack::ByteTrack(const ByteTrackParams &params) : params_(params) {}

std::vector<Track> ByteTrack::update(const std::vector<TrackDetection> &dets,
                                     const cv::Mat &frame) {
  (void)frame; // ByteTrack does not use camera-motion compensation
  ++frame_id_;
  std::vector<std::shared_ptr<STrack>> activated_stracks;
  std::vector<std::shared_ptr<STrack>> refind_stracks;
  std::vector<std::shared_ptr<STrack>> lost_stracks;
  std::vector<std::shared_ptr<STrack>> removed_stracks;

  // --- Step 1: split detections into high / low score pools.
  // ------------------
  std::vector<std::shared_ptr<STrack>> detections;        // high-score
  std::vector<std::shared_ptr<STrack>> detections_second; // low-score
  for (const auto &d : dets) {
    if (d.w <= 0 || d.h <= 0) {
      continue; // guard: the XYAH measurement divides by height
    }
    const std::array<float, 4> xywh = {d.cx, d.cy, d.w, d.h};
    auto detection =
        std::make_shared<STrack>(xywh, d.score, d.class_id, d.index);
    if (d.score >= params_.track_high_thresh) {
      detections.push_back(detection);
    } else if (d.score > params_.track_low_thresh &&
               d.score < params_.track_high_thresh) {
      detections_second.push_back(detection);
    }
  }

  // --- Step 2: split the tracked pool into unconfirmed / confirmed.
  // -----------
  std::vector<std::shared_ptr<STrack>> unconfirmed;
  std::vector<std::shared_ptr<STrack>> tracked;
  for (auto &t : tracked_stracks_) {
    (t->is_activated() ? tracked : unconfirmed).push_back(t);
  }

  // --- Step 3: joint pool (confirmed tracked + lost) and Kalman predict.
  // ------
  std::vector<std::shared_ptr<STrack>> strack_pool =
      joint_stracks(tracked, lost_stracks_);
  for (auto &t : strack_pool) {
    t->predict();
  }

  // --- Step 4: first association, high-score detections.
  // ----------------------
  std::vector<std::pair<int, int>> matches;
  std::vector<int> u_track;
  std::vector<int> u_detection;
  {
    auto dists = iou_distance(strack_pool, detections);
    if (params_.fuse_score) {
      fuse_score(dists, detections);
    }
    linear_assignment(strack_pool.size(), detections.size(), dists,
                      params_.match_thresh, matches, u_track, u_detection);
    for (const auto &m : matches) {
      auto track = strack_pool[m.first];
      auto detection = detections[m.second];
      if (track->state() == TrackState::Tracked) {
        track->update(*detection, frame_id_);
        activated_stracks.push_back(track);
      } else {
        track->re_activate(*detection, frame_id_, false);
        refind_stracks.push_back(track);
      }
    }
  }

  // --- Step 5: second association with low-score detections.
  // -------------------
  std::vector<std::shared_ptr<STrack>> r_tracked_stracks;
  for (int i : u_track) {
    if (strack_pool[i]->state() == TrackState::Tracked) {
      r_tracked_stracks.push_back(strack_pool[i]);
    }
  }
  std::vector<int> u_track_second;
  if (!r_tracked_stracks.empty() && !detections_second.empty()) {
    matches.clear();
    std::vector<int> tmp_detection; // unmatched low detections are discarded
    auto dists = iou_distance(r_tracked_stracks, detections_second); // no fuse
    linear_assignment(r_tracked_stracks.size(), detections_second.size(), dists,
                      0.5, matches, u_track_second, tmp_detection);
    for (const auto &m : matches) {
      auto track = r_tracked_stracks[m.first];
      auto detection = detections_second[m.second];
      if (track->state() == TrackState::Tracked) {
        track->update(*detection, frame_id_);
        activated_stracks.push_back(track);
      } else {
        track->re_activate(*detection, frame_id_, false);
        refind_stracks.push_back(track);
      }
    }
  } else {
    u_track_second.resize(r_tracked_stracks.size());
    std::iota(u_track_second.begin(), u_track_second.end(), 0);
  }
  for (int i : u_track_second) {
    auto track = r_tracked_stracks[i];
    if (track->state() != TrackState::Lost) {
      track->mark_lost();
      lost_stracks.push_back(track);
    }
  }

  // --- Step 6: associate unconfirmed tracks with leftover high detections.
  // ----
  std::vector<std::shared_ptr<STrack>> detections_left;
  for (int i : u_detection) {
    detections_left.push_back(detections[i]);
  }
  std::vector<int> u_detection_left;
  if (!unconfirmed.empty()) {
    matches.clear();
    std::vector<int> u_unconfirmed;
    auto dists = iou_distance(unconfirmed, detections_left);
    if (params_.fuse_score) {
      fuse_score(dists, detections_left);
    }
    linear_assignment(unconfirmed.size(), detections_left.size(), dists, 0.7,
                      matches, u_unconfirmed, u_detection_left);
    for (const auto &m : matches) {
      unconfirmed[m.first]->update(*detections_left[m.second], frame_id_);
      activated_stracks.push_back(unconfirmed[m.first]);
    }
    for (int i : u_unconfirmed) {
      auto track = unconfirmed[i];
      track->mark_removed();
      removed_stracks.push_back(track);
    }
  } else {
    u_detection_left.resize(detections_left.size());
    std::iota(u_detection_left.begin(), u_detection_left.end(), 0);
  }

  // --- Step 7: activate brand-new tracks.
  // --------------------------------------
  for (int inew : u_detection_left) {
    auto track = detections_left[inew];
    if (track->score() < params_.new_track_thresh) {
      continue;
    }
    track->activate(&kalman_filter_, frame_id_);
    activated_stracks.push_back(track);
  }

  // --- Step 8: remove lost tracks aged past the buffer.
  // ------------------------
  for (auto &track : lost_stracks_) {
    if (frame_id_ - track->end_frame() > params_.track_buffer) {
      track->mark_removed();
      removed_stracks.push_back(track);
    }
  }

  // --- Step 9: update the persistent pools in the original ByteTrack order.
  // ----
  tracked_stracks_.erase(
      std::remove_if(tracked_stracks_.begin(), tracked_stracks_.end(),
                     [](const std::shared_ptr<STrack> &track) {
                       return track->state() != TrackState::Tracked;
                     }),
      tracked_stracks_.end());
  tracked_stracks_ = joint_stracks(tracked_stracks_, activated_stracks);
  tracked_stracks_ = joint_stracks(tracked_stracks_, refind_stracks);

  lost_stracks_ = sub_stracks(lost_stracks_, tracked_stracks_);
  lost_stracks_.insert(lost_stracks_.end(), lost_stracks.begin(),
                       lost_stracks.end());
  lost_stracks_ = sub_stracks(lost_stracks_, removed_stracks_);
  removed_stracks_.insert(removed_stracks_.end(), removed_stracks.begin(),
                          removed_stracks.end());
  if (removed_stracks_.size() > kRemovedBuffer) {
    removed_stracks_.erase(removed_stracks_.begin(),
                           removed_stracks_.end() - kRemovedBuffer);
  }

  auto deduplicated = remove_duplicate_stracks(tracked_stracks_, lost_stracks_);
  tracked_stracks_ = std::move(deduplicated.first);
  lost_stracks_ = std::move(deduplicated.second);

  // --- Step 10: format output (only activated tracks).
  // --------------------------
  std::vector<Track> output;
  output.reserve(tracked_stracks_.size());
  for (auto &t : tracked_stracks_) {
    if (!t->is_activated()) {
      continue;
    }
    const auto xyxy = t->xyxy();
    Track tr;
    tr.id = t->track_id();
    tr.x1 = xyxy[0];
    tr.y1 = xyxy[1];
    tr.x2 = xyxy[2];
    tr.y2 = xyxy[3];
    tr.score = t->score();
    tr.class_id = t->class_id();
    tr.index = t->idx();
    output.push_back(tr);
  }
  return output;
}

void ByteTrack::reset() {
  tracked_stracks_.clear();
  lost_stracks_.clear();
  removed_stracks_.clear();
  frame_id_ = 0;
  kalman_filter_ = KalmanFilterXYAH();
  STrack::reset_id();
}

} // namespace yolo_ros::tracking
