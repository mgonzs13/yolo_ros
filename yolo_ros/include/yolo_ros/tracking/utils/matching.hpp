// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2021 Yifu Zhang
// SPDX-License-Identifier: MIT

/// @file
/// @brief IoU cost, score fusion, linear assignment and track-list utilities
/// for ByteTrack.

#ifndef YOLO_ROS__TRACKING__UTILS__MATCHING_HPP_
#define YOLO_ROS__TRACKING__UTILS__MATCHING_HPP_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "yolo_ros/tracking/strack.hpp"

/// @addtogroup yolo_tracking
/// @{
namespace yolo_ros::tracking::utils {

/// @brief Compute 1 - IoU between every (a, b) pair of tracks.
/// @param[in] atracks First track list.
/// @param[in] btracks Second track list.
/// @return Cost matrix of shape (atracks.size(), btracks.size()).
std::vector<std::vector<double>>
iou_distance(const std::vector<std::shared_ptr<STrack>> &atracks,
             const std::vector<std::shared_ptr<STrack>> &btracks);

/// @brief Fuse the IoU cost with detection scores as in the ByteTrack
/// reference code: `fuse_cost = 1 - (1 - cost) * det_score` (per column).
/// @param[in,out] cost_matrix Cost matrix, modified in place (no extra
/// allocation).
/// @param[in] detections Detections whose scores are fused in.
/// @return Reference to the modified @p cost_matrix.
std::vector<std::vector<double>> &
fuse_score(std::vector<std::vector<double>> &cost_matrix,
           const std::vector<std::shared_ptr<STrack>> &detections);

/// @brief Cosine distance between each (track.smooth_feat, detection.curr_feat)
/// pair, clamped to >= 0. A missing feature on either side yields 2.0, so the
/// caller's `/2.0` produces the maximum cost of 1.0 (no appearance help).
/// @param[in] atracks Tracks supplying smooth_feat().
/// @param[in] btracks Detections supplying curr_feat().
/// @return The (atracks.size() x btracks.size()) cosine cost matrix.
std::vector<std::vector<double>>
embedding_distance(const std::vector<std::shared_ptr<STrack>> &atracks,
                   const std::vector<std::shared_ptr<STrack>> &btracks);

/// @brief Fuse the gated appearance distance into an IoU cost, as in the
/// BoT-SORT reference: `emb/2`, cap at @p appearance_thresh (to 1.0), force to
/// 1.0 where @p iou_mask is true, then element-wise `min` with @p iou_dists.
/// @param[in] iou_dists IoU cost (already score-fused if fuse_score is on).
/// @param[in] emb_dists Raw embedding_distance() output (divided by 2 here).
/// @param[in] iou_mask True where the pair's raw IoU cost exceeded
/// proximity_thresh.
/// @param[in] appearance_thresh Maximum accepted embedding distance.
/// @return The fused cost matrix.
std::vector<std::vector<double>>
fuse_appearance(const std::vector<std::vector<double>> &iou_dists,
                const std::vector<std::vector<double>> &emb_dists,
                const std::vector<std::vector<bool>> &iou_mask,
                double appearance_thresh);

/// @brief Hungarian (lapjv, extend_cost + cost_limit) linear assignment on the
/// cost matrix.
///
/// Only pairs whose assignment cost is <= @p thresh are retained as matches;
/// the rest are reported unmatched. The matrix dims are passed explicitly
/// (@p n_rows x @p n_cols) because an empty cost matrix (no tracks and/or no
/// detections) still needs to report its shape. The matrix is extended with
/// dummy assignments so a finite cost threshold can represent unmatched rows.
/// @param[in] n_rows Number of rows (cost-matrix first dimension).
/// @param[in] n_cols Number of columns (cost-matrix second dimension).
/// @param[in] cost_matrix Cost matrix (n_rows x n_cols).
/// @param[in] thresh Maximum cost accepted as a match.
/// @param[out] matches Accepted (row, col) pairs.
/// @param[out] unmatched_a Row indices left unmatched.
/// @param[out] unmatched_b Column indices left unmatched.
void linear_assignment(std::size_t n_rows, std::size_t n_cols,
                       const std::vector<std::vector<double>> &cost_matrix,
                       double thresh, std::vector<std::pair<int, int>> &matches,
                       std::vector<int> &unmatched_a,
                       std::vector<int> &unmatched_b);

/// @brief Union of two track lists, de-duplicated by track_id.
/// @param[in] atracks First list (wins on id collision).
/// @param[in] btracks Second list.
/// @return The merged list.
std::vector<std::shared_ptr<STrack>>
joint_stracks(const std::vector<std::shared_ptr<STrack>> &atracks,
              const std::vector<std::shared_ptr<STrack>> &btracks);

/// @brief Set difference: @p atracks minus any track whose track_id appears in
/// @p btracks.
/// @param[in] atracks Source list.
/// @param[in] btracks Tracks to subtract.
/// @return The reduced list.
std::vector<std::shared_ptr<STrack>>
sub_stracks(const std::vector<std::shared_ptr<STrack>> &atracks,
            const std::vector<std::shared_ptr<STrack>> &btracks);

/// @brief Remove duplicate tracks across two lists based on IoU distance.
///
/// Uses `dup_thresh`; the shorter-lived track is dropped, ties drop from
/// @p atracks (the tie-break used by the ByteTrack reference implementation).
/// @param[in] atracks First list.
/// @param[in] btracks Second list.
/// @param[in] dup_thresh IoU distance above which two tracks are duplicates.
/// @return The two deduplicated lists, in input order.
std::pair<std::vector<std::shared_ptr<STrack>>,
          std::vector<std::shared_ptr<STrack>>>
remove_duplicate_stracks(const std::vector<std::shared_ptr<STrack>> &atracks,
                         const std::vector<std::shared_ptr<STrack>> &btracks,
                         double dup_thresh = 0.15);

} // namespace yolo_ros::tracking::utils
/// @}

#endif // YOLO_ROS__TRACKING__UTILS__MATCHING_HPP_
