// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2021 Yifu Zhang
// SPDX-License-Identifier: MIT

#include "yolo_ros/tracking/utils/matching.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>

#include "yolo_ros/tracking/utils/lapjv.hpp"

namespace yolo_ros::tracking::utils {

namespace {

double box_iou(const std::array<float, 4> &box1,
               const std::array<float, 4> &box2) {
  const double x1 = std::max(box1[0], box2[0]);
  const double y1 = std::max(box1[1], box2[1]);
  const double x2 = std::min(box1[2], box2[2]);
  const double y2 = std::min(box1[3], box2[3]);
  const double inter = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
  const double area1 = std::max(0.0, static_cast<double>(box1[2]) - box1[0]) *
                       std::max(0.0, static_cast<double>(box1[3]) - box1[1]);
  const double area2 = std::max(0.0, static_cast<double>(box2[2]) - box2[0]) *
                       std::max(0.0, static_cast<double>(box2[3]) - box2[1]);
  const double uni = area1 + area2 - inter;
  return uni > 0 ? inter / uni : 0.0;
}

} // namespace

std::vector<std::vector<double>>
iou_distance(const std::vector<std::shared_ptr<STrack>> &atracks,
             const std::vector<std::shared_ptr<STrack>> &btracks) {
  const std::size_t rows = atracks.size();
  const std::size_t cols = btracks.size();
  std::vector<std::vector<double>> cost(rows, std::vector<double>(cols, 1.0));
  for (std::size_t i = 0; i < rows; ++i) {
    const auto a = atracks[i]->xyxy();
    for (std::size_t j = 0; j < cols; ++j) {
      cost[i][j] = 1.0 - box_iou(a, btracks[j]->xyxy());
    }
  }
  return cost;
}

std::vector<std::vector<double>> &
fuse_score(std::vector<std::vector<double>> &cost_matrix,
           const std::vector<std::shared_ptr<STrack>> &detections) {
  const std::size_t rows = cost_matrix.size();
  const std::size_t cols = rows ? cost_matrix[0].size() : 0;
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      const double iou_sim = 1.0 - cost_matrix[i][j];
      const double score = detections[j]->score();
      cost_matrix[i][j] = 1.0 - iou_sim * score;
    }
  }
  return cost_matrix;
}

std::vector<std::vector<double>>
embedding_distance(const std::vector<std::shared_ptr<STrack>> &atracks,
                   const std::vector<std::shared_ptr<STrack>> &btracks) {
  const std::size_t rows = atracks.size();
  const std::size_t cols = btracks.size();
  std::vector<std::vector<double>> cost(rows, std::vector<double>(cols, 2.0));
  for (std::size_t i = 0; i < rows; ++i) {
    const std::vector<float> &a = atracks[i]->smooth_feat();
    if (a.empty()) {
      continue;
    }
    for (std::size_t j = 0; j < cols; ++j) {
      const std::vector<float> &b = btracks[j]->curr_feat();
      if (b.empty() || b.size() != a.size()) {
        continue;
      }
      double dot = 0.0;
      for (std::size_t k = 0; k < a.size(); ++k) {
        dot += static_cast<double>(a[k]) * b[k];
      }
      cost[i][j] = std::max(0.0, 1.0 - dot);
    }
  }
  return cost;
}

std::vector<std::vector<double>>
fuse_appearance(const std::vector<std::vector<double>> &iou_dists,
                const std::vector<std::vector<double>> &emb_dists,
                const std::vector<std::vector<bool>> &iou_mask,
                double appearance_thresh) {
  std::vector<std::vector<double>> dists = iou_dists;
  for (std::size_t i = 0; i < dists.size(); ++i) {
    for (std::size_t j = 0; j < dists[i].size(); ++j) {
      double emb = emb_dists[i][j] / 2.0;
      if (emb > appearance_thresh) {
        emb = 1.0;
      }
      if (iou_mask[i][j]) {
        emb = 1.0;
      }
      dists[i][j] = std::min(dists[i][j], emb);
    }
  }
  return dists;
}

void linear_assignment(std::size_t n_rows, std::size_t n_cols,
                       const std::vector<std::vector<double>> &cost_matrix,
                       double thresh, std::vector<std::pair<int, int>> &matches,
                       std::vector<int> &unmatched_a,
                       std::vector<int> &unmatched_b) {
  matches.clear();
  unmatched_a.clear();
  unmatched_b.clear();

  if (n_rows == 0 || n_cols == 0) {
    for (std::size_t i = 0; i < n_rows; ++i) {
      unmatched_a.push_back(static_cast<int>(i));
    }
    for (std::size_t j = 0; j < n_cols; ++j) {
      unmatched_b.push_back(static_cast<int>(j));
    }
    return;
  }

  // Extend to a square matrix of order n = n_rows + n_cols. Dummy assignments
  // model unmatched rows and columns under a finite assignment threshold.
  const std::size_t n = n_rows + n_cols;
  std::vector<std::vector<double>> cost_ext(
      n, std::vector<double>(n, thresh / 2.0));
  for (std::size_t i = n_rows; i < n; ++i) {
    for (std::size_t j = n_cols; j < n; ++j) {
      cost_ext[i][j] = 0.0;
    }
  }
  for (std::size_t i = 0; i < n_rows; ++i) {
    for (std::size_t j = 0; j < n_cols; ++j) {
      cost_ext[i][j] = cost_matrix[i][j];
    }
  }

  std::vector<int> rowsol, colsol;
  lapjv_internal(n, cost_ext, rowsol, colsol);

  // Map back: assignments that land on the padding block mean "no match".
  for (std::size_t i = 0; i < n_rows; ++i) {
    if (rowsol[i] >= 0 && rowsol[i] < static_cast<int>(n_cols)) {
      matches.emplace_back(static_cast<int>(i), rowsol[i]);
    } else {
      unmatched_a.push_back(static_cast<int>(i));
    }
  }
  for (std::size_t j = 0; j < n_cols; ++j) {
    if (colsol[j] < 0 || colsol[j] >= static_cast<int>(n_rows)) {
      unmatched_b.push_back(static_cast<int>(j));
    }
  }
}

std::vector<std::shared_ptr<STrack>>
joint_stracks(const std::vector<std::shared_ptr<STrack>> &atracks,
              const std::vector<std::shared_ptr<STrack>> &btracks) {
  std::set<int> seen_ids;
  for (const auto &t : atracks) {
    seen_ids.insert(t->track_id());
  }
  std::vector<std::shared_ptr<STrack>> res = atracks;
  res.reserve(atracks.size() + btracks.size());
  for (const auto &t : btracks) {
    if (seen_ids.count(t->track_id()) == 0) {
      seen_ids.insert(t->track_id());
      res.push_back(t);
    }
  }
  return res;
}

std::vector<std::shared_ptr<STrack>>
sub_stracks(const std::vector<std::shared_ptr<STrack>> &atracks,
            const std::vector<std::shared_ptr<STrack>> &btracks) {
  std::set<int> btrack_ids;
  for (const auto &t : btracks) {
    btrack_ids.insert(t->track_id());
  }
  std::vector<std::shared_ptr<STrack>> res;
  res.reserve(atracks.size());
  for (const auto &t : atracks) {
    if (btrack_ids.count(t->track_id()) == 0) {
      res.push_back(t);
    }
  }
  return res;
}

std::pair<std::vector<std::shared_ptr<STrack>>,
          std::vector<std::shared_ptr<STrack>>>
remove_duplicate_stracks(const std::vector<std::shared_ptr<STrack>> &atracks,
                         const std::vector<std::shared_ptr<STrack>> &btracks,
                         double dup_thresh) {
  const auto pdist = iou_distance(atracks, btracks);
  std::vector<int> dupa, dupb;
  for (std::size_t p = 0; p < atracks.size(); ++p) {
    for (std::size_t q = 0; q < btracks.size(); ++q) {
      if (pdist[p][q] < dup_thresh) {
        const int timep = atracks[p]->frame_id() - atracks[p]->start_frame();
        const int timeq = btracks[q]->frame_id() - btracks[q]->start_frame();
        if (timep > timeq) {
          dupb.push_back(static_cast<int>(q));
        } else {
          dupa.push_back(static_cast<int>(p));
        }
      }
    }
  }

  const std::set<int> dupa_set(dupa.begin(), dupa.end());
  const std::set<int> dupb_set(dupb.begin(), dupb.end());
  std::vector<std::shared_ptr<STrack>> resa, resb;
  for (std::size_t i = 0; i < atracks.size(); ++i) {
    if (dupa_set.count(static_cast<int>(i)) == 0) {
      resa.push_back(atracks[i]);
    }
  }
  for (std::size_t i = 0; i < btracks.size(); ++i) {
    if (dupb_set.count(static_cast<int>(i)) == 0) {
      resb.push_back(btracks[i]);
    }
  }
  return {resa, resb};
}

} // namespace yolo_ros::tracking::utils
