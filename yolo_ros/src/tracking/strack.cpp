// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2021 Yifu Zhang
// SPDX-License-Identifier: MIT

#include "yolo_ros/tracking/strack.hpp"

#include <cmath>
#include <cstddef>

namespace yolo_ros::tracking {

using namespace utils;

int STrack::count_ = 0;

STrack::STrack(const std::array<float, 4> &xywh, float score, int class_id,
               int idx) {
  // xywh (center) -> tlwh
  _tlwh_[0] = xywh[0] - xywh[2] / 2;
  _tlwh_[1] = xywh[1] - xywh[3] / 2;
  _tlwh_[2] = xywh[2];
  _tlwh_[3] = xywh[3];
  kf_ = nullptr;
  mean_ = {};
  covariance_ = {};
  has_state_ = false;
  is_activated_ = false;
  track_id_ = 0;
  frame_id_ = 0;
  start_frame_ = 0;
  tracklet_len_ = 0;
  state_ = TrackState::New;
  score_ = score;
  class_id_ = class_id;
  idx_ = idx;
}

void STrack::update_features(const std::vector<float> &feat) {
  if (feat.empty()) {
    return;
  }
  double norm = 0.0;
  for (const float value : feat) {
    norm += static_cast<double>(value) * value;
  }
  norm = std::sqrt(norm);
  if (norm <= 1e-12) {
    return; // a zero-norm feature is not a usable appearance descriptor
  }

  curr_feat_.assign(feat.size(), 0.0f);
  for (std::size_t i = 0; i < feat.size(); ++i) {
    curr_feat_[i] = static_cast<float>(feat[i] / norm);
  }

  // First observation (or a dimension change): smooth starts equal to curr.
  if (smooth_feat_.size() != curr_feat_.size()) {
    smooth_feat_ = curr_feat_;
    return;
  }
  double smooth_norm = 0.0;
  for (std::size_t i = 0; i < curr_feat_.size(); ++i) {
    smooth_feat_[i] = static_cast<float>(kFeatureAlpha * smooth_feat_[i] +
                                         (1.0 - kFeatureAlpha) * curr_feat_[i]);
    smooth_norm += static_cast<double>(smooth_feat_[i]) * smooth_feat_[i];
  }
  smooth_norm = std::sqrt(smooth_norm);
  if (smooth_norm > 1e-12) {
    for (float &value : smooth_feat_) {
      value = static_cast<float>(value / smooth_norm);
    }
  }
}

void STrack::predict() {
  if (!has_state_) {
    return;
  }
  KalmanMean mean_state = mean_;
  if (state_ != TrackState::Tracked) {
    mean_state[7] = 0; // freeze height velocity for lost tracks
  }
  auto predicted = kf_->predict(mean_state, covariance_);
  mean_ = predicted.first;
  covariance_ = predicted.second;
}

void STrack::activate(const utils::KalmanFilter *kalman_filter, int frame_id) {
  kf_ = kalman_filter;
  track_id_ = next_id();
  auto initiated = kf_->initiate(kf_->box_to_measurement(_tlwh_));
  mean_ = initiated.first;
  covariance_ = initiated.second;
  has_state_ = true;

  tracklet_len_ = 0;
  state_ = TrackState::Tracked;
  is_activated_ = (frame_id == 1);
  frame_id_ = frame_id;
  start_frame_ = frame_id;
}

void STrack::re_activate(const STrack &new_track, int frame_id, bool new_id) {
  auto updated = kf_->update(mean_, covariance_,
                             kf_->box_to_measurement(new_track._tlwh_));
  mean_ = updated.first;
  covariance_ = updated.second;

  tracklet_len_ = 0;
  state_ = TrackState::Tracked;
  is_activated_ = true;
  frame_id_ = frame_id;
  if (new_id) {
    track_id_ = next_id();
  }
  score_ = new_track.score_;
  class_id_ = new_track.class_id_;
  idx_ = new_track.idx_;
  if (new_track.has_feature()) {
    update_features(new_track.curr_feat());
  }
}

void STrack::update(const STrack &new_track, int frame_id) {
  frame_id_ = frame_id;
  tracklet_len_ += 1;

  auto updated = kf_->update(mean_, covariance_,
                             kf_->box_to_measurement(new_track._tlwh_));
  mean_ = updated.first;
  covariance_ = updated.second;
  state_ = TrackState::Tracked;
  is_activated_ = true;

  score_ = new_track.score_;
  class_id_ = new_track.class_id_;
  idx_ = new_track.idx_;
  if (new_track.has_feature()) {
    update_features(new_track.curr_feat());
  }
}

int STrack::next_id() { return ++count_; }

std::array<float, 4> STrack::tlwh() const {
  if (!has_state_) {
    return _tlwh_;
  }
  return kf_->measurement_to_tlwh(mean_);
}

std::array<float, 4> STrack::xyxy() const {
  auto box = tlwh();
  box[2] += box[0];
  box[3] += box[1];
  return box;
}

std::array<float, 4> STrack::xywh() const {
  auto box = tlwh();
  box[0] += box[2] / 2;
  box[1] += box[3] / 2;
  return box;
}

void STrack::apply_affine(const utils::KalmanAffine &warp) {
  if (!has_state_) {
    return;
  }
  // R8 = kron(I4, R): the 2x2 linear part repeated as four 2x2 diagonal blocks.
  double r8[8][8] = {};
  for (std::size_t b = 0; b < 4; ++b) {
    r8[2 * b][2 * b] = warp.r[0][0];
    r8[2 * b][2 * b + 1] = warp.r[0][1];
    r8[2 * b + 1][2 * b] = warp.r[1][0];
    r8[2 * b + 1][2 * b + 1] = warp.r[1][1];
  }

  KalmanMean new_mean{};
  for (std::size_t i = 0; i < 8; ++i) {
    double acc = 0.0;
    for (std::size_t j = 0; j < 8; ++j) {
      acc += r8[i][j] * mean_[j];
    }
    new_mean[i] = acc;
  }
  new_mean[0] += warp.t[0];
  new_mean[1] += warp.t[1];

  // cov = R8 * cov * R8^T
  KalmanCovariance tmp{};
  for (std::size_t i = 0; i < 8; ++i) {
    for (std::size_t j = 0; j < 8; ++j) {
      double acc = 0.0;
      for (std::size_t k = 0; k < 8; ++k) {
        acc += r8[i][k] * covariance_[k][j];
      }
      tmp[i][j] = acc;
    }
  }
  KalmanCovariance new_cov{};
  for (std::size_t i = 0; i < 8; ++i) {
    for (std::size_t j = 0; j < 8; ++j) {
      double acc = 0.0;
      for (std::size_t k = 0; k < 8; ++k) {
        acc += tmp[i][k] * r8[j][k];
      }
      new_cov[i][j] = acc;
    }
  }

  mean_ = new_mean;
  covariance_ = new_cov;
}

} // namespace yolo_ros::tracking
