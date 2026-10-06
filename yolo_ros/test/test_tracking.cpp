// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "yolo_ros/engine/reid_encoder.hpp"
#include "yolo_ros/tracking/bot_sort.hpp"
#include "yolo_ros/tracking/byte_tracker.hpp"
#include "yolo_ros/tracking/strack.hpp"
#include "yolo_ros/tracking/tracker.hpp"
#include "yolo_ros/tracking/utils/camera_motion.hpp"
#include "yolo_ros/tracking/utils/lapjv.hpp"
#include "yolo_ros/tracking/utils/matching.hpp"

namespace yolo_ros::tracking {
namespace {

TEST(KalmanFilter, InitiateSetsPositionAndZeroVelocity) {
  utils::KalmanFilterXYAH kf;
  const auto [mean, cov] = kf.initiate({10.0, 20.0, 2.0, 40.0});
  EXPECT_DOUBLE_EQ(mean[0], 10.0);
  EXPECT_DOUBLE_EQ(mean[1], 20.0);
  EXPECT_DOUBLE_EQ(mean[2], 2.0);
  EXPECT_DOUBLE_EQ(mean[3], 40.0);
  for (int i = 4; i < 8; ++i) {
    EXPECT_DOUBLE_EQ(mean[i], 0.0);
  }
  EXPECT_GT(cov[0][0], 0.0);
  EXPECT_GT(cov[3][3], 0.0);
}

TEST(KalmanFilter, PredictAdvancesByVelocity) {
  utils::KalmanFilterXYAH kf;
  auto [mean, cov] = kf.initiate({10.0, 20.0, 2.0, 40.0});
  mean[4] = 5.0;
  mean[5] = -3.0;
  const auto [pm, pc] = kf.predict(mean, cov);
  EXPECT_DOUBLE_EQ(pm[0], 15.0);
  EXPECT_DOUBLE_EQ(pm[1], 17.0);
  EXPECT_DOUBLE_EQ(pm[2], 2.0);
  EXPECT_DOUBLE_EQ(pm[3], 40.0);
  EXPECT_GT(pc[0][0], cov[0][0]);
}

TEST(KalmanFilter, UpdatePullsTowardMeasurement) {
  utils::KalmanFilterXYAH kf;
  const auto [mean, cov] = kf.initiate({0.0, 0.0, 1.0, 20.0});
  const auto [um, uc] = kf.update(mean, cov, {10.0, 0.0, 1.0, 20.0});
  EXPECT_GT(um[0], 0.0);
  EXPECT_LT(um[0], 10.0);
  EXPECT_LT(uc[0][0], cov[0][0]);
}

TEST(Matching, IouDistance) {
  auto a = std::make_shared<STrack>(std::array<float, 4>{50, 50, 20, 20}, 0.9f,
                                    0, 0);
  auto b = std::make_shared<STrack>(std::array<float, 4>{50, 50, 20, 20}, 0.9f,
                                    0, 1);
  auto c = std::make_shared<STrack>(std::array<float, 4>{500, 500, 20, 20},
                                    0.9f, 0, 2);
  const auto d = utils::iou_distance({a, b}, {b, c});
  ASSERT_EQ(d.size(), 2u);
  EXPECT_NEAR(d[0][0], 0.0, 1e-9);
  EXPECT_NEAR(d[0][1], 1.0, 1e-9);
  EXPECT_NEAR(d[1][1], 1.0, 1e-9);
}

TEST(Matching, FuseScore) {
  auto det = std::make_shared<STrack>(std::array<float, 4>{50, 50, 20, 20},
                                      0.5f, 0, 0);
  std::vector<std::vector<double>> cost = {{0.4}};
  utils::fuse_score(cost, {det});
  EXPECT_NEAR(cost[0][0], 1.0 - (1.0 - 0.4) * 0.5, 1e-9);
}

TEST(Matching, LinearAssignmentOptimal) {
  const std::vector<std::vector<double>> cost = {{0.1, 0.9}, {0.8, 0.2}};
  std::vector<std::pair<int, int>> matches;
  std::vector<int> ua, ub;
  utils::linear_assignment(2, 2, cost, 0.5, matches, ua, ub);
  ASSERT_EQ(matches.size(), 2u);
  EXPECT_TRUE(ua.empty());
  EXPECT_TRUE(ub.empty());
  const std::set<std::pair<int, int>> got(matches.begin(), matches.end());
  EXPECT_TRUE(got.count({0, 0}));
  EXPECT_TRUE(got.count({1, 1}));
}

TEST(Matching, LinearAssignmentThresholdLeavesUnmatched) {
  const std::vector<std::vector<double>> cost = {{0.9}};
  std::vector<std::pair<int, int>> matches;
  std::vector<int> ua, ub;
  utils::linear_assignment(1, 1, cost, 0.5, matches, ua, ub);
  EXPECT_TRUE(matches.empty());
  EXPECT_EQ(ua.size(), 1u);
  EXPECT_EQ(ub.size(), 1u);
}

TEST(Matching, LinearAssignmentEmpty) {
  const std::vector<std::vector<double>> cost;
  std::vector<std::pair<int, int>> matches;
  std::vector<int> ua, ub;
  utils::linear_assignment(0, 0, cost, 0.5, matches, ua, ub);
  EXPECT_TRUE(matches.empty());
  EXPECT_TRUE(ua.empty());
  EXPECT_TRUE(ub.empty());
}

TEST(Lapjv, IdentityCost) {
  const std::vector<std::vector<double>> cost = {
      {0.0, 1.0, 2.0}, {2.0, 0.0, 1.0}, {1.0, 2.0, 0.0}};
  std::vector<int> rowsol, colsol;
  ASSERT_EQ(utils::lapjv_internal(3, cost, rowsol, colsol), 0);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(rowsol[i], i);
  }
}

TEST(KalmanFilterXYAH, BoxToMeasurement) {
  utils::KalmanFilterXYAH kf;
  const auto xyah = kf.box_to_measurement({10, 20, 40, 20});
  EXPECT_DOUBLE_EQ(xyah[0], 30.0);
  EXPECT_DOUBLE_EQ(xyah[1], 30.0);
  EXPECT_DOUBLE_EQ(xyah[2], 2.0);
  EXPECT_DOUBLE_EQ(xyah[3], 20.0);
}

TEST(KalmanFilterXYWH, InitiateSetsObservationAndZeroVelocity) {
  utils::KalmanFilterXYWH kf;
  const auto [mean, cov] = kf.initiate({10.0, 20.0, 30.0, 40.0});
  EXPECT_DOUBLE_EQ(mean[0], 10.0);
  EXPECT_DOUBLE_EQ(mean[1], 20.0);
  EXPECT_DOUBLE_EQ(mean[2], 30.0);
  EXPECT_DOUBLE_EQ(mean[3], 40.0);
  for (int i = 4; i < 8; ++i) {
    EXPECT_DOUBLE_EQ(mean[i], 0.0);
  }
  EXPECT_GT(cov[0][0], 0.0);
  EXPECT_GT(cov[2][2], 0.0);
  EXPECT_GT(cov[3][3], 0.0);
}

TEST(KalmanFilterXYWH, PredictAdvancesPositionAndKeepsSize) {
  utils::KalmanFilterXYWH kf;
  auto [mean, cov] = kf.initiate({10.0, 20.0, 30.0, 40.0});
  mean[4] = 2.0;
  mean[5] = -1.0;
  const auto [pm, pc] = kf.predict(mean, cov);
  EXPECT_DOUBLE_EQ(pm[0], 12.0);
  EXPECT_DOUBLE_EQ(pm[1], 19.0);
  EXPECT_DOUBLE_EQ(pm[2], 30.0);
  EXPECT_DOUBLE_EQ(pm[3], 40.0);
  EXPECT_GT(pc[0][0], cov[0][0]);
}

TEST(KalmanFilterXYWH, UpdatePullsTowardMeasurement) {
  utils::KalmanFilterXYWH kf;
  const auto [mean, cov] = kf.initiate({0.0, 0.0, 20.0, 20.0});
  const auto [um, uc] = kf.update(mean, cov, {10.0, 0.0, 20.0, 20.0});
  EXPECT_GT(um[0], 0.0);
  EXPECT_LT(um[0], 10.0);
  EXPECT_LT(uc[0][0], cov[0][0]);
}

TEST(KalmanFilterXYWH, BoxMeasurementRoundTrip) {
  utils::KalmanFilterXYWH kf;
  const std::array<float, 4> tlwh{10, 20, 40, 20};
  const auto m = kf.box_to_measurement(tlwh);
  EXPECT_DOUBLE_EQ(m[0], 30.0);
  EXPECT_DOUBLE_EQ(m[1], 30.0);
  EXPECT_DOUBLE_EQ(m[2], 40.0);
  EXPECT_DOUBLE_EQ(m[3], 20.0);
  utils::KalmanMean mean{};
  mean[0] = m[0];
  mean[1] = m[1];
  mean[2] = m[2];
  mean[3] = m[3];
  const auto back = kf.measurement_to_tlwh(mean);
  EXPECT_NEAR(back[0], 10.0, 1e-4);
  EXPECT_NEAR(back[1], 20.0, 1e-4);
  EXPECT_NEAR(back[2], 40.0, 1e-4);
  EXPECT_NEAR(back[3], 20.0, 1e-4);
}

TEST(STrack, ApplyAffineWarpsPositionAndCovariance) {
  utils::KalmanFilterXYAH kf;
  STrack::reset_id();
  STrack s({50, 50, 20, 20}, 0.9f, 0, 0);
  s.activate(&kf, 1);
  utils::KalmanAffine warp;
  warp.t[0] = 3.0;
  warp.t[1] = -2.0;
  s.apply_affine(warp);
  const auto xyxy = s.xyxy();
  EXPECT_NEAR(xyxy[0], 43.0, 1e-3);
  EXPECT_NEAR(xyxy[1], 38.0, 1e-3);
}

TEST(STrack, ActivateAssignsIdStateAndBox) {
  utils::KalmanFilterXYAH kf;
  STrack::reset_id();
  STrack s({50, 50, 20, 10}, 0.8f, 2, 7);
  EXPECT_FALSE(s.is_activated());
  s.activate(&kf, 1);
  EXPECT_TRUE(s.is_activated());
  EXPECT_EQ(s.track_id(), 1);
  EXPECT_EQ(s.state(), TrackState::Tracked);
  const auto xyxy = s.xyxy();
  EXPECT_NEAR(xyxy[0], 40.0, 1e-4);
  EXPECT_NEAR(xyxy[1], 45.0, 1e-4);
  EXPECT_NEAR(xyxy[2], 60.0, 1e-4);
  EXPECT_NEAR(xyxy[3], 55.0, 1e-4);
}

TEST(Tracks, JointDedupesById) {
  utils::KalmanFilterXYAH kf;
  STrack::reset_id();
  auto a =
      std::make_shared<STrack>(std::array<float, 4>{0, 0, 10, 10}, 0.9f, 0, 0);
  auto b = std::make_shared<STrack>(std::array<float, 4>{20, 20, 10, 10}, 0.9f,
                                    0, 1);
  a->activate(&kf, 1);
  b->activate(&kf, 1);
  STrack::count_ = 0; // force the next activation to reuse id 1
  auto a_dup =
      std::make_shared<STrack>(std::array<float, 4>{0, 0, 10, 10}, 0.9f, 0, 2);
  a_dup->activate(&kf, 1);
  ASSERT_EQ(a_dup->track_id(), a->track_id());
  ASSERT_NE(a_dup.get(), a.get());

  const auto j = utils::joint_stracks({a}, {b, a_dup});
  ASSERT_EQ(j.size(), 2u);
  EXPECT_EQ(j[0].get(), a.get());
  EXPECT_EQ(j[1].get(), b.get());
}

TEST(Tracks, SubRemovesById) {
  utils::KalmanFilterXYAH kf;
  STrack::reset_id();
  auto a =
      std::make_shared<STrack>(std::array<float, 4>{0, 0, 10, 10}, 0.9f, 0, 0);
  auto b = std::make_shared<STrack>(std::array<float, 4>{20, 20, 10, 10}, 0.9f,
                                    0, 1);
  auto c = std::make_shared<STrack>(std::array<float, 4>{40, 40, 10, 10}, 0.9f,
                                    0, 2);
  a->activate(&kf, 1);
  b->activate(&kf, 1);
  c->activate(&kf, 1);
  STrack::count_ = 1; // force the next activation to reuse id 2
  auto b_dup = std::make_shared<STrack>(std::array<float, 4>{20, 20, 10, 10},
                                        0.9f, 0, 3);
  b_dup->activate(&kf, 1);
  ASSERT_EQ(b_dup->track_id(), b->track_id());
  ASSERT_NE(b_dup.get(), b.get());

  const auto s = utils::sub_stracks({a, b, c}, {b_dup});
  ASSERT_EQ(s.size(), 2u);
  EXPECT_EQ(s[0]->track_id(), a->track_id());
  EXPECT_EQ(s[1]->track_id(), c->track_id());
}

TEST(Tracks, RemoveDuplicateDropsTieFromFirstList) {
  utils::KalmanFilterXYAH kf;
  STrack::reset_id();
  auto a = std::make_shared<STrack>(std::array<float, 4>{50, 50, 20, 20}, 0.9f,
                                    0, 0);
  auto b = std::make_shared<STrack>(std::array<float, 4>{50, 50, 20, 20}, 0.9f,
                                    0, 1);
  a->activate(&kf, 1);
  b->activate(&kf, 1);
  const auto [ra, rb] = utils::remove_duplicate_stracks({a}, {b});
  EXPECT_TRUE(ra.empty());
  EXPECT_EQ(rb.size(), 1u);
}

TEST(ByteTrack, KeepsIdAcrossFrames) {
  ByteTrack tracker(ByteTrackParams{});
  std::vector<TrackDetection> dets(1);
  dets[0] = {50, 50, 20, 20, 0.9f, 0, 0};
  const auto out1 = tracker.update(dets);
  ASSERT_EQ(out1.size(), 1u);
  const int id = out1[0].id;
  dets[0].cx = 52.0f;
  const auto out2 = tracker.update(dets);
  ASSERT_EQ(out2.size(), 1u);
  EXPECT_EQ(out2[0].id, id);
}

TEST(ByteTrack, LowScoreSecondStageAssociates) {
  ByteTrack tracker(ByteTrackParams{});
  std::vector<TrackDetection> high(1);
  high[0] = {50, 50, 20, 20, 0.9f, 0, 0};
  const auto out_high = tracker.update(high);
  ASSERT_EQ(out_high.size(), 1u);
  const int id = out_high[0].id;
  std::vector<TrackDetection> low(1);
  low[0] = {51, 50, 20, 20, 0.15f, 0, 0};
  const auto out = tracker.update(low);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].id, id);
}

TEST(ByteTrack, ReacquiresLostTrackWithinBuffer) {
  ByteTrack tracker(ByteTrackParams{});
  std::vector<TrackDetection> high(1);
  high[0] = {50, 50, 20, 20, 0.9f, 0, 0};
  const auto out1 = tracker.update(high);
  ASSERT_EQ(out1.size(), 1u);
  const int id = out1[0].id;
  EXPECT_TRUE(tracker.update({}).empty());
  const auto out3 = tracker.update(high);
  ASSERT_EQ(out3.size(), 1u);
  EXPECT_EQ(out3[0].id, id);
}

TEST(ByteTrack, DropsTrackAfterBufferExpiry) {
  ByteTrackParams params;
  params.track_buffer = 1;
  ByteTrack tracker(params);
  std::vector<TrackDetection> high(1);
  high[0] = {50, 50, 20, 20, 0.9f, 0, 0};
  const auto out1 = tracker.update(high);
  ASSERT_EQ(out1.size(), 1u);
  const int id = out1[0].id;
  // The expired track is dropped from the lost pool after a short grace
  // (upstream ByteTrack's removal ordering); once gone, a new detection must
  // get a fresh id rather than reviving the old one. A brand-new track is only
  // emitted once confirmed on a subsequent frame, so feed the detection twice.
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(tracker.update({}).empty());
  }
  const auto first = tracker.update(high);
  for (const auto &t : first) {
    EXPECT_NE(t.id, id);
  }
  const auto out = tracker.update(high);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_NE(out[0].id, id);
}

TEST(ByteTrack, BelowNewTrackThresholdStartsNoTrack) {
  ByteTrackParams params;
  params.new_track_thresh = 0.5;
  ByteTrack tracker(params);
  std::vector<TrackDetection> dets(1);
  dets[0] = {50, 50, 20, 20, 0.3f, 0, 0};
  EXPECT_TRUE(tracker.update(dets).empty());
}

TEST(ByteTrack, ResetClearsStateAndRestartsIds) {
  ByteTrack tracker(ByteTrackParams{});
  std::vector<TrackDetection> dets(1);
  dets[0] = {50, 50, 20, 20, 0.9f, 0, 0};
  ASSERT_EQ(tracker.update(dets).size(), 1u);
  tracker.reset();
  EXPECT_EQ(tracker.frame_id(), 0);
  const auto out = tracker.update(dets);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].id, 1);
}

TEST(TrackerFactory, CreatesByteTrack) {
  ByteTrackParams params;
  EXPECT_NE(create_tracker(params), nullptr);
}

TEST(TrackerFactory, CaseInsensitive) {
  ByteTrackParams params;
  params.type = "ByteTrack";
  EXPECT_NE(create_tracker(params), nullptr);
}

TEST(TrackerFactory, UnknownTypeIsNull) {
  TrackerParams params;
  params.type = "nope";
  EXPECT_EQ(create_tracker(params), nullptr);
}

TEST(TrackerFactory, MismatchedParamsIsNull) {
  TrackerParams params;
  params.type = "bytetrack";
  EXPECT_EQ(create_tracker(params), nullptr);
}

TEST(CameraMotionCompensator, NoneIsIdentityAndDisabled) {
  utils::CameraMotionCompensator cmc("none");
  EXPECT_FALSE(cmc.enabled());
  const cv::Mat frame(16, 16, CV_8UC1, cv::Scalar(0));
  const auto warp = cmc.apply(frame);
  EXPECT_DOUBLE_EQ(warp.t[0], 0.0);
  EXPECT_DOUBLE_EQ(warp.t[1], 0.0);
  EXPECT_DOUBLE_EQ(warp.r[0][0], 1.0);
}

TEST(CameraMotionCompensator, EmptyFrameIsIdentity) {
  utils::CameraMotionCompensator cmc("sparseOptFlow", 2);
  EXPECT_TRUE(cmc.enabled());
  const auto warp = cmc.apply(cv::Mat{});
  EXPECT_DOUBLE_EQ(warp.t[0], 0.0);
  EXPECT_DOUBLE_EQ(warp.t[1], 0.0);
}

TEST(CameraMotionCompensator, FirstFrameIsIdentity) {
  utils::CameraMotionCompensator cmc("sparseOptFlow", 1);
  const cv::Mat frame(64, 64, CV_8UC1, cv::Scalar(0));
  const auto warp = cmc.apply(frame);
  EXPECT_DOUBLE_EQ(warp.t[0], 0.0);
  EXPECT_DOUBLE_EQ(warp.t[1], 0.0);
}

TEST(CameraMotionCompensator, SparseOptFlowRecoversTranslation) {
  utils::CameraMotionCompensator cmc("sparseOptFlow", 1);
  cv::Mat frame1(240, 320, CV_8UC1, cv::Scalar(0));
  for (int y = 20; y < 220; y += 40) {
    for (int x = 20; x < 300; x += 40) {
      cv::circle(frame1, cv::Point(x, y), 5, cv::Scalar(255), cv::FILLED);
    }
  }
  cv::Mat frame2;
  const cv::Mat translate = (cv::Mat_<double>(2, 3) << 1, 0, 4, 0, 1, -3);
  cv::warpAffine(frame1, frame2, translate, frame1.size());

  const auto first = cmc.apply(frame1);
  EXPECT_NEAR(first.t[0], 0.0, 1e-9);
  const auto warp = cmc.apply(frame2);
  EXPECT_NEAR(warp.t[0], 4.0, 1.0);
  EXPECT_NEAR(warp.t[1], -3.0, 1.0);
}

TEST(CameraMotionCompensator, OrbAndEccAreFiniteAndIdentityOnFirstFrame) {
  for (const std::string method : {"orb", "ecc"}) {
    utils::CameraMotionCompensator cmc(method, 2);
    EXPECT_TRUE(cmc.enabled()) << method;
    cv::Mat frame(64, 64, CV_8UC1, cv::Scalar(0));
    cv::rectangle(frame, cv::Rect(10, 10, 40, 40), cv::Scalar(255), cv::FILLED);
    const auto warp = cmc.apply(frame);
    EXPECT_TRUE(std::isfinite(warp.t[0])) << method;
    EXPECT_TRUE(std::isfinite(warp.t[1])) << method;
  }
}

TEST(BotSort, KeepsIdAcrossFrames) {
  BotSort tracker(BotSortParams{});
  std::vector<TrackDetection> dets(1);
  dets[0] = {50, 50, 20, 20, 0.9f, 0, 0};
  const auto out1 = tracker.update(dets);
  ASSERT_EQ(out1.size(), 1u);
  const int id = out1[0].id;
  dets[0].cx = 52.0f;
  const auto out2 = tracker.update(dets);
  ASSERT_EQ(out2.size(), 1u);
  EXPECT_EQ(out2[0].id, id);
}

TEST(BotSort, EmitsUnconfirmedNewTrackImmediately) {
  BotSort tracker(BotSortParams{});
  std::vector<TrackDetection> dets(1);
  dets[0] = {50, 50, 20, 20, 0.9f, 0, 0};
  ASSERT_EQ(tracker.update(dets).size(), 1u);
  dets.push_back({500, 500, 20, 20, 0.9f, 0, 1});
  const auto out = tracker.update(dets);
  EXPECT_EQ(out.size(), 2u);
}

TEST(BotSort, BelowNewTrackThresholdStartsNoTrack) {
  BotSortParams params;
  params.new_track_thresh = 0.5;
  BotSort tracker(params);
  std::vector<TrackDetection> dets(1);
  dets[0] = {50, 50, 20, 20, 0.3f, 0, 0};
  EXPECT_TRUE(tracker.update(dets).empty());
}

TEST(BotSort, ExpiresLostTrackAfterBuffer) {
  BotSortParams params;
  params.track_buffer = 1;
  BotSort tracker(params);
  std::vector<TrackDetection> dets(1);
  dets[0] = {50, 50, 20, 20, 0.9f, 0, 0};
  const auto out1 = tracker.update(dets);
  ASSERT_EQ(out1.size(), 1u);
  const int id = out1[0].id;
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(tracker.update({}).empty());
  }
  const auto out = tracker.update(dets);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_NE(out[0].id, id);
}

TEST(BotSort, ResetClearsStateAndRestartsIds) {
  BotSort tracker(BotSortParams{});
  std::vector<TrackDetection> dets(1);
  dets[0] = {50, 50, 20, 20, 0.9f, 0, 0};
  ASSERT_EQ(tracker.update(dets).size(), 1u);
  tracker.reset();
  EXPECT_EQ(tracker.frame_id(), 0);
  const auto out = tracker.update(dets);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].id, 1);
}

TEST(BotSort, NeedsFrameFollowsGmcMethod) {
  BotSortParams params;
  BotSort without_gmc(params);
  EXPECT_FALSE(without_gmc.needs_frame());
  params.gmc_method = "sparseOptFlow";
  BotSort with_gmc(params);
  EXPECT_TRUE(with_gmc.needs_frame());
}

TEST(TrackerFactory, CreatesBotSort) {
  BotSortParams params;
  EXPECT_NE(create_tracker(params), nullptr);
}

TEST(TrackerFactory, BotSortCaseInsensitive) {
  BotSortParams params;
  params.type = "BotSort";
  EXPECT_NE(create_tracker(params), nullptr);
}

TEST(TrackerFactory, BotSortMismatchedParamsIsNull) {
  TrackerParams params;
  params.type = "botsort";
  EXPECT_EQ(create_tracker(params), nullptr);
}

TEST(STrack, FeaturesAreNormalizedAndEmaSmoothed) {
  STrack s({50, 50, 20, 20}, 0.9f, 0, 0);
  EXPECT_FALSE(s.has_feature());
  s.update_features({3.0f, 4.0f});
  ASSERT_TRUE(s.has_feature());
  EXPECT_NEAR(s.curr_feat()[0], 0.6f, 1e-6);
  EXPECT_NEAR(s.curr_feat()[1], 0.8f, 1e-6);
  // The first feature sets smooth == curr.
  EXPECT_NEAR(s.smooth_feat()[0], 0.6f, 1e-6);
  EXPECT_NEAR(s.smooth_feat()[1], 0.8f, 1e-6);

  s.update_features({1.0f, 0.0f});
  // smooth = 0.9*old + 0.1*new, renormalized.
  const float e0 = 0.9f * 0.6f + 0.1f * 1.0f;
  const float e1 = 0.9f * 0.8f + 0.1f * 0.0f;
  const float n = std::sqrt(e0 * e0 + e1 * e1);
  EXPECT_NEAR(s.curr_feat()[0], 1.0f, 1e-6);
  EXPECT_NEAR(s.smooth_feat()[0], e0 / n, 1e-6);
  EXPECT_NEAR(s.smooth_feat()[1], e1 / n, 1e-6);
}

TEST(STrack, UpdatePropagatesDetectionFeature) {
  utils::KalmanFilterXYAH kf;
  STrack::reset_id();
  STrack track({50, 50, 20, 20}, 0.9f, 0, 0);
  track.activate(&kf, 1);
  STrack det({52, 50, 20, 20}, 0.9f, 0, 1);
  det.update_features({1.0f, 0.0f});
  track.update(det, 2);
  ASSERT_TRUE(track.has_feature());
  EXPECT_NEAR(track.curr_feat()[0], 1.0f, 1e-6);
}

TEST(STrack, ReActivatePropagatesDetectionFeature) {
  utils::KalmanFilterXYAH kf;
  STrack::reset_id();
  STrack track({50, 50, 20, 20}, 0.9f, 0, 0);
  track.activate(&kf, 1);
  STrack det({52, 50, 20, 20}, 0.9f, 0, 1);
  det.update_features({2.0f, 0.0f});
  track.re_activate(det, 2, false);
  ASSERT_TRUE(track.has_feature());
  EXPECT_NEAR(track.curr_feat()[0], 1.0f, 1e-6);
}

TEST(STrack, ZeroNormFeatureLeavesHasFeatureFalse) {
  STrack s({50, 50, 20, 20}, 0.9f, 0, 0);
  s.update_features({0.0f, 0.0f});
  EXPECT_FALSE(s.has_feature());
}

TEST(Matching, EmbeddingDistanceCosine) {
  auto a =
      std::make_shared<STrack>(std::array<float, 4>{0, 0, 10, 10}, 0.9f, 0, 0);
  auto b =
      std::make_shared<STrack>(std::array<float, 4>{0, 0, 10, 10}, 0.9f, 0, 1);
  a->update_features({1.0f, 0.0f});
  b->update_features({1.0f, 0.0f});
  auto d = utils::embedding_distance({a}, {b});
  ASSERT_EQ(d.size(), 1u);
  EXPECT_NEAR(d[0][0], 0.0, 1e-6);

  auto orthogonal =
      std::make_shared<STrack>(std::array<float, 4>{0, 0, 10, 10}, 0.9f, 0, 2);
  orthogonal->update_features({0.0f, 1.0f});
  d = utils::embedding_distance({a}, {orthogonal});
  EXPECT_NEAR(d[0][0], 1.0, 1e-6);

  auto opposite =
      std::make_shared<STrack>(std::array<float, 4>{0, 0, 10, 10}, 0.9f, 0, 3);
  opposite->update_features({-1.0f, 0.0f});
  d = utils::embedding_distance({a}, {opposite});
  EXPECT_NEAR(d[0][0], 2.0, 1e-6);
}

TEST(Matching, EmbeddingDistanceMissingFeatureIsMax) {
  auto track =
      std::make_shared<STrack>(std::array<float, 4>{0, 0, 10, 10}, 0.9f, 0, 0);
  auto det =
      std::make_shared<STrack>(std::array<float, 4>{0, 0, 10, 10}, 0.9f, 0, 1);
  det->update_features({1.0f, 0.0f});
  // Track has no feature.
  EXPECT_NEAR(utils::embedding_distance({track}, {det})[0][0], 2.0, 1e-9);
  // Detection has no feature (non-person box).
  EXPECT_NEAR(utils::embedding_distance({det}, {track})[0][0], 2.0, 1e-9);

  auto det3 =
      std::make_shared<STrack>(std::array<float, 4>{0, 0, 10, 10}, 0.9f, 0, 2);
  det3->update_features({1.0f, 0.0f, 0.0f});
  // Dimension mismatch (2-D vs 3-D) also yields the sentinel.
  EXPECT_NEAR(utils::embedding_distance({det}, {det3})[0][0], 2.0, 1e-9);
}

TEST(Matching, FuseAppearanceGatesAndTakesMinimum) {
  const std::vector<std::vector<double>> iou = {{0.4, 0.4, 0.4, 0.4}};
  // /2 -> {0.1, 0.3, 1.0, 0.1}; col1 exceeds appearance_thresh (caps to 1.0),
  // col2 caps too, col3 is <= thresh but masked.
  const std::vector<std::vector<double>> emb = {{0.2, 0.6, 2.0, 0.2}};
  const std::vector<std::vector<bool>> mask = {{false, false, false, true}};
  const auto d = utils::fuse_appearance(iou, emb, mask, 0.25);
  EXPECT_NEAR(d[0][0], 0.1, 1e-9); // min(0.4, 0.1)
  EXPECT_NEAR(d[0][1], 0.4, 1e-9); // capped to 1.0, min(0.4, 1.0)
  EXPECT_NEAR(d[0][2], 0.4, 1e-9); // capped to 1.0, min(0.4, 1.0)
  // emb/2 = 0.1 <= thresh, so only the mask can force 1.0 here: min(0.4, 1.0).
  EXPECT_NEAR(d[0][3], 0.4, 1e-9);
}

TEST(Matching, FuseAppearanceEmptyPassesThrough) {
  const std::vector<std::vector<double>> empty;
  const std::vector<std::vector<bool>> empty_mask;
  EXPECT_TRUE(utils::fuse_appearance(empty, empty, empty_mask, 0.25).empty());
}

TEST(ReIDEncoder, MakeBatchIsNchwRgbZeroTo255) {
  cv::Mat frame(8, 8, CV_8UC3, cv::Scalar(10, 20, 30)); // BGR
  const std::vector<std::array<float, 4>> boxes = {{0, 0, 8, 8}};
  std::vector<float> out;
  engine::ReIDEncoder::make_batch(frame, boxes, 4, 2, out);
  ASSERT_EQ(out.size(), 1u * 3u * 2u * 4u);
  // RGB order on a constant patch: R=30, G=20, B=10, stride H*W=8.
  EXPECT_NEAR(out[0], 30.0f, 1e-3);
  EXPECT_NEAR(out[8], 20.0f, 1e-3);
  EXPECT_NEAR(out[16], 10.0f, 1e-3);
}

TEST(ReIDEncoder, MakeBatchHandlesDegenerateAndOutOfFrameBoxes) {
  cv::Mat frame(8, 8, CV_8UC3, cv::Scalar(10, 20, 30));
  const std::vector<std::array<float, 4>> boxes = {{-5, -5, 4, 4},
                                                   {100, 100, 200, 200}};
  std::vector<float> out;
  ASSERT_NO_THROW(engine::ReIDEncoder::make_batch(frame, boxes, 4, 2, out));
  ASSERT_EQ(out.size(), 2u * 3u * 2u * 4u);
  for (const float value : out) {
    EXPECT_TRUE(std::isfinite(value));
  }
  // Box 1 lies entirely outside the frame: its whole 3*H*W slice stays zero
  // (3 channels * 2 * 4 = 24 floats per box, so [24, 48)).
  for (std::size_t i = 24; i < 48; ++i) {
    EXPECT_FLOAT_EQ(out[i], 0.0f);
  }
}

TEST(ReIDEncoder, MissingModelThrows) {
  EXPECT_THROW(
      engine::ReIDEncoder("/nonexistent/reid-model.onnx", "cpu", "cpu"),
      std::runtime_error);
}

TEST(BotSort, WithReidEmptyModelIsMotionOnly) {
  BotSortParams params;
  params.with_reid = true; // reid_model stays empty
  BotSort tracker(params);
  EXPECT_FALSE(tracker.needs_frame()); // no encoder and no CMC
  std::vector<TrackDetection> dets(1);
  dets[0] = {50, 50, 20, 20, 0.9f, 0, 0};
  EXPECT_EQ(tracker.update(dets).size(), 1u);
}

TEST(BotSort, WithReidAndGmcNeedsFrame) {
  BotSortParams params;
  params.with_reid = true;
  params.gmc_method = "sparseOptFlow";
  BotSort tracker(params);
  EXPECT_TRUE(tracker.needs_frame());
}

TEST(TrackerFactory, CreatesBotSortWithReidParams) {
  BotSortParams params;
  params.with_reid = true;
  params.reid_model = "";
  EXPECT_NE(create_tracker(params), nullptr);
}

TEST(ReIDEncoder, ProducesUnitNormEmbeddingsWhenModelAvailable) {
  const char *env = std::getenv("YOLO_ROS_REID_MODEL");
  const std::string path =
      env != nullptr ? env : "/home/agonzc34/models/osnet_x0_25_reid.onnx";
  std::ifstream probe(path);
  if (!probe.good()) {
    GTEST_SKIP() << "set YOLO_ROS_REID_MODEL to a ReID ONNX model to run this";
  }
  engine::ReIDEncoder encoder(path, "cpu", "cpu");
  cv::Mat frame(240, 320, CV_8UC3, cv::Scalar(10, 20, 30));
  const std::vector<std::array<float, 4>> boxes = {{10, 10, 90, 210},
                                                   {120, 20, 220, 230}};
  const auto feats = encoder.inference(frame, boxes);
  ASSERT_EQ(feats.size(), 2u);
  EXPECT_EQ(feats[0].size(), 512u);
  for (const auto &feature : feats) {
    double norm = 0.0;
    for (const float value : feature) {
      norm += static_cast<double>(value) * value;
    }
    EXPECT_NEAR(std::sqrt(norm), 1.0, 1e-3);
  }
}

} // namespace
} // namespace yolo_ros::tracking
