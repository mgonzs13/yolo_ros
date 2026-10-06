// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "yolo_ros/3d/depth_utils.hpp"

namespace yolo_ros::depth {
namespace {

sensor_msgs::msg::CameraInfo make_camera(int w, int h, double fx, double fy,
                                         double cx, double cy) {
  sensor_msgs::msg::CameraInfo info;
  info.width = static_cast<uint32_t>(w);
  info.height = static_cast<uint32_t>(h);
  info.k = {fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0};
  return info;
}

TEST(Stats, Median) {
  EXPECT_DOUBLE_EQ(median({1.0, 3.0, 2.0}), 2.0);
  EXPECT_DOUBLE_EQ(median({4.0, 1.0, 3.0, 2.0}), 2.5);
  EXPECT_DOUBLE_EQ(median({}), 0.0);
}

TEST(Stats, PercentileSortedLinear) {
  const std::vector<double> v = {1.0, 2.0, 3.0, 4.0};
  EXPECT_DOUBLE_EQ(percentile_sorted(v, 0.0), 1.0);
  EXPECT_DOUBLE_EQ(percentile_sorted(v, 0.5), 2.5);
  EXPECT_DOUBLE_EQ(percentile_sorted(v, 0.25), 1.75);
  EXPECT_DOUBLE_EQ(percentile_sorted(v, 1.0), 4.0);
}

TEST(Stats, WeightedMean) {
  EXPECT_NEAR(weighted_mean({1.0, 2.0}, {1.0, 3.0}), 1.75, 1e-9);
  // Zero total weight falls back to the median.
  EXPECT_DOUBLE_EQ(weighted_mean({5.0, 1.0, 3.0}, {0.0, 0.0, 0.0}), 3.0);
}

TEST(Stats, SpatialWeightsCenterIsMaxAndFloorHolds) {
  const auto center = compute_spatial_weights({50}, {50}, 50, 50, 20, 20);
  ASSERT_EQ(center.size(), 1u);
  EXPECT_NEAR(center[0], 1.0, 1e-9);

  const auto corner = compute_spatial_weights({0}, {0}, 500, 500, 20, 20);
  ASSERT_EQ(corner.size(), 1u);
  EXPECT_NEAR(corner[0], 0.3, 1e-9);

  const auto sym = compute_spatial_weights({40, 60}, {50, 50}, 50, 50, 20, 20);
  ASSERT_EQ(sym.size(), 2u);
  EXPECT_NEAR(sym[0], sym[1], 1e-12);
}

TEST(Stats, DepthBoundsOrdering) {
  const std::vector<double> depth = {1.00, 1.01, 1.02, 1.03, 1.04,
                                     1.05, 1.06, 1.07, 1.08, 1.09};
  const std::vector<double> weight(depth.size(), 1.0);
  const auto b = compute_depth_bounds_weighted(depth, weight);
  EXPECT_LE(b.min, b.center);
  EXPECT_LE(b.center, b.max);
  EXPECT_NEAR(b.center, 1.04, 1e-6);
}

TEST(Stats, DepthBoundsDegenerateConstant) {
  const std::vector<double> depth = {2.0, 2.0, 2.0, 2.0};
  const std::vector<double> weight(depth.size(), 1.0);
  const auto b = compute_depth_bounds_weighted(depth, weight);
  EXPECT_DOUBLE_EQ(b.center, 2.0);
  EXPECT_DOUBLE_EQ(b.min, 2.0);
  EXPECT_DOUBLE_EQ(b.max, 2.0);
}

TEST(Stats, DepthBoundsSmallSample) {
  const std::vector<double> depth = {1.0, 2.0, 3.0};
  const std::vector<double> weight(depth.size(), 1.0);
  const auto b = compute_depth_bounds_weighted(depth, weight);
  EXPECT_DOUBLE_EQ(b.center, 2.0);
  EXPECT_DOUBLE_EQ(b.min, 1.0);
  EXPECT_DOUBLE_EQ(b.max, 3.0);
}

TEST(Stats, AxisBoundsRejectsOutlier) {
  const std::vector<double> val = {0.0, 0.0, 0.0, 0.0, 100.0};
  const std::vector<double> w(val.size(), 1.0);
  const auto b = compute_axis_bounds(val, w, 4.5, 0.06, 0.50);
  EXPECT_NEAR(b.center, 0.0, 1e-9);
  EXPECT_LT(b.max, 1.0);
  EXPECT_GT(b.max, b.min);
}

TEST(Stats, AxisBoundsSmallSample) {
  const std::vector<double> val = {3.0, 1.0, 2.0};
  const std::vector<double> w(val.size(), 1.0);
  const auto b = compute_axis_bounds(val, w, 4.5, 0.06, 0.50);
  EXPECT_DOUBLE_EQ(b.center, 2.0);
  EXPECT_DOUBLE_EQ(b.min, 1.0);
  EXPECT_DOUBLE_EQ(b.max, 3.0);
}

TEST(Pixel, DepthAtPixel16UC1And32FC1) {
  cv::Mat mm16(4, 4, CV_16UC1, cv::Scalar(1000));
  EXPECT_DOUBLE_EQ(depth_at_pixel(mm16, 0, 0, 1000), 1.0);

  cv::Mat m32(4, 4, CV_32FC1, cv::Scalar(1.5f));
  EXPECT_DOUBLE_EQ(depth_at_pixel(m32, 3, 3, 1000), 1.5);
}

TEST(Pixel, DepthAtPixelOutOfRangeIsZero) {
  cv::Mat depth(4, 4, CV_16UC1, cv::Scalar(1000));
  EXPECT_DOUBLE_EQ(depth_at_pixel(depth, -1, 0, 1000), 0.0);
  EXPECT_DOUBLE_EQ(depth_at_pixel(depth, 0, -1, 1000), 0.0);
  EXPECT_DOUBLE_EQ(depth_at_pixel(depth, 4, 0, 1000), 0.0);
  EXPECT_DOUBLE_EQ(depth_at_pixel(depth, 0, 4, 1000), 0.0);
  EXPECT_DOUBLE_EQ(depth_at_pixel(cv::Mat(), 0, 0, 1000), 0.0);
}

TEST(Pixel, DepthAtPixelZeroDivisorIsZero) {
  cv::Mat depth(4, 4, CV_16UC1, cv::Scalar(1000));
  EXPECT_DOUBLE_EQ(depth_at_pixel(depth, 0, 0, 0), 0.0);
}

TEST(Pixel, DepthAtPixelUnsupportedTypeIsZero) {
  cv::Mat depth(4, 4, CV_8UC1, cv::Scalar(10));
  EXPECT_DOUBLE_EQ(depth_at_pixel(depth, 0, 0, 1000), 0.0);
}

TEST(Quaternion, Rotates90AboutZ) {
  constexpr double k = 0.7071067811865476;
  const std::array<double, 4> q = {k, 0.0, 0.0, k};
  const auto v = qv_mult(q, {1.0, 0.0, 0.0});
  EXPECT_NEAR(v[0], 0.0, 1e-9);
  EXPECT_NEAR(v[1], 1.0, 1e-9);
  EXPECT_NEAR(v[2], 0.0, 1e-9);
}

TEST(Lifting, KeypointsBackProject) {
  cv::Mat depth(100, 100, CV_32FC1, cv::Scalar(2.0f));
  const auto info = make_camera(100, 100, 100.0, 100.0, 0.0, 0.0);
  yolo_msgs::msg::Detection det;
  yolo_msgs::msg::KeyPoint2D kp;
  kp.id = 1;
  kp.point.x = 50.0;
  kp.point.y = 50.0;
  kp.score = 0.9f;
  det.keypoints.data.push_back(kp);

  const auto kp3d = convert_keypoints_to_3d(depth, info, det, 1000);
  ASSERT_EQ(kp3d.data.size(), 1u);
  EXPECT_NEAR(kp3d.data[0].point.x, 1.0, 1e-6);
  EXPECT_NEAR(kp3d.data[0].point.y, 1.0, 1e-6);
  EXPECT_NEAR(kp3d.data[0].point.z, 2.0, 1e-6);
  EXPECT_EQ(kp3d.data[0].id, 1);
  EXPECT_FLOAT_EQ(kp3d.data[0].score, 0.9f);
}

TEST(Lifting, KeypointsBackProjectAsymmetric) {
  cv::Mat depth(480, 640, CV_32FC1, cv::Scalar(2.0f));
  const auto info = make_camera(640, 480, 600.0, 500.0, 320.0, 240.0);
  yolo_msgs::msg::Detection det;
  yolo_msgs::msg::KeyPoint2D kp;
  kp.id = 1;
  kp.point.x = 400.0;
  kp.point.y = 100.0;
  kp.score = 0.9f;
  det.keypoints.data.push_back(kp);

  const auto kp3d = convert_keypoints_to_3d(depth, info, det, 1000);
  ASSERT_EQ(kp3d.data.size(), 1u);
  // x is derived from the column (point.x), y from the row (point.y).
  EXPECT_NEAR(kp3d.data[0].point.x, 2.0 * (400.0 - 320.0) / 600.0, 1e-6);
  EXPECT_NEAR(kp3d.data[0].point.y, 2.0 * (100.0 - 240.0) / 500.0, 1e-6);
  EXPECT_NEAR(kp3d.data[0].point.z, 2.0, 1e-6);
}

TEST(Lifting, KeypointsSkipInvalidDepth) {
  cv::Mat depth(100, 100, CV_32FC1, cv::Scalar(0.0f));
  const auto info = make_camera(100, 100, 100.0, 100.0, 50.0, 50.0);
  yolo_msgs::msg::Detection det;
  yolo_msgs::msg::KeyPoint2D kp;
  kp.id = 1;
  kp.point.x = 50.0;
  kp.point.y = 50.0;
  det.keypoints.data.push_back(kp);
  EXPECT_TRUE(convert_keypoints_to_3d(depth, info, det, 1000).data.empty());
}

TEST(Lifting, BoxConstantDepth) {
  cv::Mat depth(100, 100, CV_32FC1, cv::Scalar(2.0f));
  const auto info = make_camera(100, 100, 100.0, 100.0, 50.0, 50.0);
  yolo_msgs::msg::Detection det;
  det.bbox.center.position.x = 50.0;
  det.bbox.center.position.y = 50.0;
  det.bbox.size.x = 20.0;
  det.bbox.size.y = 20.0;

  const auto box = convert_bb_to_3d(depth, info, det, 1000);
  ASSERT_TRUE(box.has_value());
  EXPECT_NEAR(box->center.position.z, 2.0, 1e-6);
  EXPECT_NEAR(box->center.position.x, 0.0, 0.01);
  EXPECT_NEAR(box->center.position.y, 0.0, 0.01);
}

TEST(Lifting, BoxHasIdentityOrientationWhenDisabled) {
  cv::Mat depth(100, 100, CV_32FC1, cv::Scalar(2.0f));
  const auto info = make_camera(100, 100, 100.0, 100.0, 50.0, 50.0);
  yolo_msgs::msg::Detection det;
  det.bbox.center.position.x = 50.0;
  det.bbox.center.position.y = 50.0;
  det.bbox.size.x = 20.0;
  det.bbox.size.y = 20.0;

  // Orientation estimation is disabled by default, so the box must keep the
  // identity quaternion that geometry_msgs/Pose initializes (w = 1).
  const auto box = convert_bb_to_3d(depth, info, det, 1000);
  ASSERT_TRUE(box.has_value());
  EXPECT_DOUBLE_EQ(box->center.orientation.x, 0.0);
  EXPECT_DOUBLE_EQ(box->center.orientation.y, 0.0);
  EXPECT_DOUBLE_EQ(box->center.orientation.z, 0.0);
  EXPECT_DOUBLE_EQ(box->center.orientation.w, 1.0);
}

TEST(Lifting, BoxConstantDepthAsymmetric) {
  cv::Mat depth(480, 640, CV_32FC1, cv::Scalar(2.0f));
  const auto info = make_camera(640, 480, 600.0, 500.0, 320.0, 240.0);
  yolo_msgs::msg::Detection det;
  det.bbox.center.position.x = 340.0;
  det.bbox.center.position.y = 220.0;
  det.bbox.size.x = 20.0;
  det.bbox.size.y = 20.0;

  const auto box = convert_bb_to_3d(depth, info, det, 1000);
  ASSERT_TRUE(box.has_value());
  EXPECT_NEAR(box->center.position.z, 2.0, 1e-6);
  EXPECT_NEAR(box->center.position.x, 2.0 * (340.0 - 320.0) / 600.0, 0.015);
  EXPECT_NEAR(box->center.position.y, 2.0 * (220.0 - 240.0) / 500.0, 0.015);
}

TEST(Lifting, BoxNoValidDepthIsNullopt) {
  cv::Mat depth(100, 100, CV_32FC1, cv::Scalar(0.0f));
  const auto info = make_camera(100, 100, 100.0, 100.0, 50.0, 50.0);
  yolo_msgs::msg::Detection det;
  det.bbox.center.position.x = 50.0;
  det.bbox.center.position.y = 50.0;
  det.bbox.size.x = 20.0;
  det.bbox.size.y = 20.0;
  EXPECT_FALSE(convert_bb_to_3d(depth, info, det, 1000).has_value());
}

TEST(Transform, BoxIdentityAndTranslation) {
  yolo_msgs::msg::BoundingBox3D box;
  box.center.position.x = 1.0;
  box.center.position.y = 2.0;
  box.center.position.z = 3.0;
  box.center.orientation.w = 1.0;
  box.size.x = 4.0;
  box.size.y = 5.0;
  box.size.z = 6.0;

  const auto identity =
      transform_3d_box(box, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0});
  EXPECT_DOUBLE_EQ(identity.center.position.x, 1.0);
  EXPECT_DOUBLE_EQ(identity.center.position.y, 2.0);
  EXPECT_DOUBLE_EQ(identity.center.position.z, 3.0);
  EXPECT_DOUBLE_EQ(identity.size.x, 4.0);

  const auto moved =
      transform_3d_box(box, {10.0, 20.0, 30.0}, {1.0, 0.0, 0.0, 0.0});
  EXPECT_DOUBLE_EQ(moved.center.position.x, 11.0);
  EXPECT_DOUBLE_EQ(moved.center.position.y, 22.0);
  EXPECT_DOUBLE_EQ(moved.center.position.z, 33.0);
}

TEST(Transform, BoxRotatesAxisAlignedSize) {
  yolo_msgs::msg::BoundingBox3D box;
  box.center.orientation.w = 1.0;
  box.size.x = 4.0;
  box.size.y = 2.0;
  box.size.z = 6.0;
  constexpr double k = 0.7071067811865476;
  const std::array<double, 4> qz90 = {k, 0.0, 0.0, k};

  const auto out = transform_3d_box(box, {0.0, 0.0, 0.0}, qz90);
  EXPECT_NEAR(out.size.x, 2.0, 1e-9);
  EXPECT_NEAR(out.size.y, 4.0, 1e-9);
  EXPECT_NEAR(out.size.z, 6.0, 1e-9);
}

TEST(Transform, OrientedBoxKeepsLocalSizeAndComposesOrientation) {
  yolo_msgs::msg::BoundingBox3D box;
  constexpr double k = 0.7071067811865476;
  box.center.orientation.w = k;
  box.center.orientation.z = k;
  box.size.x = 4.0;
  box.size.y = 2.0;
  box.size.z = 6.0;
  const std::array<double, 4> qz90 = {k, 0.0, 0.0, k};

  const auto out = transform_3d_box(box, {0.0, 0.0, 0.0}, qz90);
  EXPECT_DOUBLE_EQ(out.size.x, 4.0);
  EXPECT_DOUBLE_EQ(out.size.y, 2.0);
  EXPECT_DOUBLE_EQ(out.size.z, 6.0);
  // 90 deg frame rotation composed with a 90 deg box orientation = 180 deg z.
  EXPECT_NEAR(out.center.orientation.w, 0.0, 1e-9);
  EXPECT_NEAR(out.center.orientation.z, 1.0, 1e-9);
}

TEST(Transform, KeypointsTranslation) {
  yolo_msgs::msg::KeyPoint3DArray kps;
  yolo_msgs::msg::KeyPoint3D kp;
  kp.point.x = 1.0;
  kp.point.y = 2.0;
  kp.point.z = 3.0;
  kps.data.push_back(kp);

  const auto out =
      transform_3d_keypoints(kps, {10.0, 20.0, 30.0}, {1.0, 0.0, 0.0, 0.0});
  ASSERT_EQ(out.data.size(), 1u);
  EXPECT_DOUBLE_EQ(out.data[0].point.x, 11.0);
  EXPECT_DOUBLE_EQ(out.data[0].point.y, 22.0);
  EXPECT_DOUBLE_EQ(out.data[0].point.z, 33.0);
}

TEST(Transform, KeypointsRotation) {
  yolo_msgs::msg::KeyPoint3DArray kps;
  yolo_msgs::msg::KeyPoint3D kp;
  kp.point.x = 1.0;
  kp.point.y = 0.0;
  kp.point.z = 0.0;
  kps.data.push_back(kp);
  constexpr double k = 0.7071067811865476;

  const auto out =
      transform_3d_keypoints(kps, {0.0, 0.0, 0.0}, {k, 0.0, 0.0, k});
  ASSERT_EQ(out.data.size(), 1u);
  EXPECT_NEAR(out.data[0].point.x, 0.0, 1e-9);
  EXPECT_NEAR(out.data[0].point.y, 1.0, 1e-9);
  EXPECT_NEAR(out.data[0].point.z, 0.0, 1e-9);
}

} // namespace
} // namespace yolo_ros::depth
