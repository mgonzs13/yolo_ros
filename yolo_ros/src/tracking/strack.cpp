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
  this->_tlwh_[0] = xywh[0] - xywh[2] / 2;
  this->_tlwh_[1] = xywh[1] - xywh[3] / 2;
  this->_tlwh_[2] = xywh[2];
  this->_tlwh_[3] = xywh[3];
  this->kf_ = nullptr;
  this->mean_ = {};
  this->covariance_ = {};
  this->has_state_ = false;
  this->is_activated_ = false;
  this->track_id_ = 0;
  this->frame_id_ = 0;
  this->start_frame_ = 0;
  this->tracklet_len_ = 0;
  this->state_ = TrackState::New;
  this->score_ = score;
  this->class_id_ = class_id;
  this->idx_ = idx;
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

  this->curr_feat_.assign(feat.size(), 0.0f);

  for (std::size_t i = 0; i < feat.size(); ++i) {
    this->curr_feat_[i] = static_cast<float>(feat[i] / norm);
  }

  // First observation (or a dimension change): smooth starts equal to curr.
  if (this->smooth_feat_.size() != this->curr_feat_.size()) {
    this->smooth_feat_ = this->curr_feat_;
    return;
  }

  double smooth_norm = 0.0;

  for (std::size_t i = 0; i < this->curr_feat_.size(); ++i) {
    this->smooth_feat_[i] =
        static_cast<float>(kFeatureAlpha * this->smooth_feat_[i] +
                           (1.0 - kFeatureAlpha) * this->curr_feat_[i]);
    smooth_norm +=
        static_cast<double>(this->smooth_feat_[i]) * this->smooth_feat_[i];
  }

  smooth_norm = std::sqrt(smooth_norm);

  if (smooth_norm > 1e-12) {
    for (float &value : this->smooth_feat_) {
      value = static_cast<float>(value / smooth_norm);
    }
  }
}

void STrack::predict() {
  if (!this->has_state_) {
    return;
  }

  KalmanMean mean_state = this->mean_;

  if (this->state_ != TrackState::Tracked) {
    mean_state[7] = 0; // freeze height velocity for lost tracks
  }

  auto predicted = this->kf_->predict(mean_state, this->covariance_);
  this->mean_ = predicted.first;
  this->covariance_ = predicted.second;
}

void STrack::activate(const utils::KalmanFilter *kalman_filter, int frame_id) {
  this->kf_ = kalman_filter;
  this->track_id_ = next_id();
  auto initiated =
      this->kf_->initiate(this->kf_->box_to_measurement(this->_tlwh_));
  this->mean_ = initiated.first;
  this->covariance_ = initiated.second;
  this->has_state_ = true;

  this->tracklet_len_ = 0;
  this->state_ = TrackState::Tracked;
  this->is_activated_ = (frame_id == 1);
  this->frame_id_ = frame_id;
  this->start_frame_ = frame_id;
}

void STrack::re_activate(const STrack &new_track, int frame_id, bool new_id) {
  auto updated =
      this->kf_->update(this->mean_, this->covariance_,
                        this->kf_->box_to_measurement(new_track._tlwh_));
  this->mean_ = updated.first;
  this->covariance_ = updated.second;

  this->tracklet_len_ = 0;
  this->state_ = TrackState::Tracked;
  this->is_activated_ = true;
  this->frame_id_ = frame_id;

  if (new_id) {
    this->track_id_ = next_id();
  }

  this->score_ = new_track.score_;
  this->class_id_ = new_track.class_id_;
  this->idx_ = new_track.idx_;

  if (new_track.has_feature()) {
    this->update_features(new_track.curr_feat());
  }
}

void STrack::update(const STrack &new_track, int frame_id) {
  this->frame_id_ = frame_id;
  this->tracklet_len_ += 1;

  auto updated =
      this->kf_->update(this->mean_, this->covariance_,
                        this->kf_->box_to_measurement(new_track._tlwh_));
  this->mean_ = updated.first;
  this->covariance_ = updated.second;
  this->state_ = TrackState::Tracked;
  this->is_activated_ = true;

  this->score_ = new_track.score_;
  this->class_id_ = new_track.class_id_;
  this->idx_ = new_track.idx_;

  if (new_track.has_feature()) {
    this->update_features(new_track.curr_feat());
  }
}

int STrack::next_id() { return ++count_; }

std::array<float, 4> STrack::tlwh() const {
  if (!this->has_state_) {
    return this->_tlwh_;
  }

  return this->kf_->measurement_to_tlwh(this->mean_);
}

std::array<float, 4> STrack::xyxy() const {
  auto box = this->tlwh();
  box[2] += box[0];
  box[3] += box[1];
  return box;
}

std::array<float, 4> STrack::xywh() const {
  auto box = this->tlwh();
  box[0] += box[2] / 2;
  box[1] += box[3] / 2;
  return box;
}

void STrack::apply_affine(const utils::KalmanAffine &warp) {
  if (!this->has_state_) {
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
      acc += r8[i][j] * this->mean_[j];
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
        acc += r8[i][k] * this->covariance_[k][j];
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

  this->mean_ = new_mean;
  this->covariance_ = new_cov;
}

} // namespace yolo_ros::tracking
