// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2022 Nir Aharon
// SPDX-License-Identifier: MIT

#include "yolo_ros/tracking/utils/camera_motion.hpp"
#include "yolo_ros/utils/logs.hpp"
#include "yolo_ros/utils/string_utils.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace yolo_ros::tracking::utils {

namespace {

// Build a KalmanAffine from a 2x3 CV_64F warp, scaling the translation back to
// full-resolution pixels when a downscale factor was applied.
KalmanAffine to_affine(const cv::Mat &warp, int downscale) {
  KalmanAffine out;
  if (warp.empty()) {
    return out;
  }
  out.r[0][0] = warp.at<double>(0, 0);
  out.r[0][1] = warp.at<double>(0, 1);
  out.r[1][0] = warp.at<double>(1, 0);
  out.r[1][1] = warp.at<double>(1, 1);
  const double scale = downscale > 1 ? downscale : 1;
  out.t[0] = warp.at<double>(0, 2) * scale;
  out.t[1] = warp.at<double>(1, 2) * scale;
  return out;
}

} // namespace

CameraMotionCompensator::CameraMotionCompensator(const std::string &method,
                                                 int downscale)
    : method_(yolo_ros::utils::to_lower(method)),
      downscale_(downscale < 1 ? 1 : downscale) {
  if (method_ == "orb") {
    detector_ = cv::FastFeatureDetector::create(20);
    extractor_ = cv::ORB::create();
    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);
  } else if (method_ == "ecc") {
    ecc_criteria_ = cv::TermCriteria(
        cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 5000, 1e-6);
  } else if (method_ != "none" && method_ != "sparseoptflow") {
    YOLO_LOG_WARN("Unknown camera-motion method \"%s\"; disabling "
                  "camera-motion compensation.",
                  method.c_str());
    method_ = "none";
  }
}

void CameraMotionCompensator::reset() {
  prev_frame_.release();
  prev_points_.clear();
  prev_keypoints_.clear();
  prev_descriptors_.release();
  initialized_ = false;
}

KalmanAffine CameraMotionCompensator::apply(const cv::Mat &frame) {
  if (!enabled() || frame.empty()) {
    return KalmanAffine{};
  }
  if (method_ == "sparseoptflow") {
    return apply_sparse_opt_flow(frame);
  }
  if (method_ == "orb") {
    return apply_features(frame);
  }
  if (method_ == "ecc") {
    return apply_ecc(frame);
  }
  return KalmanAffine{};
}

KalmanAffine
CameraMotionCompensator::apply_sparse_opt_flow(const cv::Mat &frame) {
  cv::Mat gray;
  if (frame.channels() == 1) {
    gray = frame;
  } else {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  }
  if (downscale_ > 1) {
    cv::Mat small;
    cv::resize(gray, small,
               cv::Size(gray.cols / downscale_, gray.rows / downscale_));
    gray = small;
  }

  // Reference feature params: maxCorners=1000, qualityLevel=0.01,
  // minDistance=1, blockSize=3, no Harris.
  std::vector<cv::Point2f> points;
  cv::goodFeaturesToTrack(gray, points, 1000, 0.01, 1.0, cv::noArray(), 3,
                          false, 0.04);

  if (!initialized_) {
    prev_frame_ = gray.clone();
    prev_points_ = points;
    initialized_ = true;
    return KalmanAffine{};
  }

  KalmanAffine result;
  if (!prev_points_.empty() && !points.empty()) {
    std::vector<cv::Point2f> matched;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prev_frame_, gray, prev_points_, matched, status,
                             err);
    std::vector<cv::Point2f> prev_good;
    std::vector<cv::Point2f> curr_good;
    for (std::size_t i = 0; i < status.size(); ++i) {
      if (status[i]) {
        prev_good.push_back(prev_points_[i]);
        curr_good.push_back(matched[i]);
      }
    }
    if (prev_good.size() > 4) {
      const cv::Mat warp = cv::estimateAffinePartial2D(
          prev_good, curr_good, cv::noArray(), cv::RANSAC);
      if (!warp.empty()) {
        result = to_affine(warp, downscale_);
      }
    }
  }

  prev_frame_ = gray.clone();
  prev_points_ = points;
  return result;
}

KalmanAffine CameraMotionCompensator::apply_features(const cv::Mat &frame) {
  cv::Mat gray;
  if (frame.channels() == 1) {
    gray = frame;
  } else {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  }
  if (downscale_ > 1) {
    cv::Mat small;
    cv::resize(gray, small,
               cv::Size(gray.cols / downscale_, gray.rows / downscale_));
    gray = small;
  }

  std::vector<cv::KeyPoint> keypoints;
  detector_->detect(gray, keypoints);
  cv::Mat descriptors;
  extractor_->compute(gray, keypoints, descriptors);

  if (!initialized_) {
    prev_keypoints_ = keypoints;
    prev_descriptors_ = descriptors;
    initialized_ = true;
    return KalmanAffine{};
  }

  KalmanAffine result;
  if (!prev_descriptors_.empty() && !descriptors.empty()) {
    std::vector<std::vector<cv::DMatch>> knn;
    matcher_->knnMatch(prev_descriptors_, descriptors, knn, 2);

    std::vector<cv::Point2f> prev_good;
    std::vector<cv::Point2f> curr_good;
    std::vector<cv::Point2f> deltas;
    for (const auto &pair : knn) {
      if (pair.size() != 2 || pair[0].distance >= 0.9 * pair[1].distance) {
        continue;
      }
      const cv::Point2f prev_pt = prev_keypoints_[pair[0].queryIdx].pt;
      const cv::Point2f curr_pt = keypoints[pair[0].trainIdx].pt;
      prev_good.push_back(prev_pt);
      curr_good.push_back(curr_pt);
      deltas.push_back(curr_pt - prev_pt);
    }

    // Spatial outlier rejection: drop matches farther than 2.5 std from the
    // mean displacement.
    if (deltas.size() > 4) {
      cv::Scalar mean;
      cv::Scalar stddev;
      cv::meanStdDev(deltas, mean, stddev);
      std::vector<cv::Point2f> prev_in;
      std::vector<cv::Point2f> curr_in;
      for (std::size_t i = 0; i < deltas.size(); ++i) {
        if (std::abs(deltas[i].x - mean[0]) < 2.5 * stddev[0] &&
            std::abs(deltas[i].y - mean[1]) < 2.5 * stddev[1]) {
          prev_in.push_back(prev_good[i]);
          curr_in.push_back(curr_good[i]);
        }
      }
      prev_good = prev_in;
      curr_good = curr_in;
    }

    if (prev_good.size() > 4) {
      const cv::Mat warp = cv::estimateAffinePartial2D(
          prev_good, curr_good, cv::noArray(), cv::RANSAC);
      if (!warp.empty()) {
        result = to_affine(warp, downscale_);
      }
    }
  }

  prev_keypoints_ = keypoints;
  prev_descriptors_ = descriptors;
  return result;
}

KalmanAffine CameraMotionCompensator::apply_ecc(const cv::Mat &frame) {
  cv::Mat gray;
  if (frame.channels() == 1) {
    gray = frame;
  } else {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  }
  cv::GaussianBlur(gray, gray, cv::Size(3, 3), 1.5);
  if (downscale_ > 1) {
    cv::Mat small;
    cv::resize(gray, small,
               cv::Size(gray.cols / downscale_, gray.rows / downscale_));
    gray = small;
  }

  if (!initialized_) {
    prev_frame_ = gray.clone();
    initialized_ = true;
    return KalmanAffine{};
  }

  cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);
  try {
    cv::findTransformECC(prev_frame_, gray, warp, cv::MOTION_EUCLIDEAN,
                         ecc_criteria_);
  } catch (const cv::Exception &) {
    // Reference behavior: a failed ECC solve keeps the identity warp.
    warp = cv::Mat::eye(2, 3, CV_32F);
  }
  prev_frame_ = gray.clone();

  cv::Mat warp64;
  warp.convertTo(warp64, CV_64F);
  // The reference omits the downscale correction for ECC; apply it here for
  // consistency with the other methods.
  return to_affine(warp64, downscale_);
}

} // namespace yolo_ros::tracking::utils
