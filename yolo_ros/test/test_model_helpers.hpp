// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#ifndef YOLO_ROS__TEST__TEST_MODEL_HELPERS_HPP_
#define YOLO_ROS__TEST__TEST_MODEL_HELPERS_HPP_

#include <unistd.h>

#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <curl/curl.h>
#include <jpeglib.h>
#include <opencv2/core.hpp>

#include "huggingface_hub.h"
#include "yolo_ros/yolo/utils.hpp"

namespace yolo_ros::test {

namespace detail {

struct JpegErrorMgr {
  jpeg_error_mgr pub;
  jmp_buf setjmp_buffer;
};

inline void jpeg_error_exit_cb(j_common_ptr cinfo) {
  auto *err = reinterpret_cast<JpegErrorMgr *>(cinfo->err);
  longjmp(err->setjmp_buffer, 1);
}

inline cv::Mat decode_jpeg(const std::string &path) {
  FILE *fp = std::fopen(path.c_str(), "rb");
  if (fp == nullptr) {
    return {};
  }
  auto cinfo = std::make_unique<jpeg_decompress_struct>();
  JpegErrorMgr jerr;
  cv::Mat image;
  std::vector<JSAMPLE> row;
  cinfo->err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = jpeg_error_exit_cb;
  if (setjmp(jerr.setjmp_buffer)) {
    jpeg_destroy_decompress(cinfo.get());
    std::fclose(fp);
    return {};
  }
  jpeg_create_decompress(cinfo.get());
  jpeg_stdio_src(cinfo.get(), fp);
  jpeg_read_header(cinfo.get(), TRUE);
  cinfo->out_color_space = JCS_RGB;
  jpeg_start_decompress(cinfo.get());
  const int width = static_cast<int>(cinfo->output_width);
  const int height = static_cast<int>(cinfo->output_height);
  image.create(height, width, CV_8UC3);
  row.resize(static_cast<std::size_t>(width) * 3);
  JSAMPROW row_ptr = row.data();
  while (cinfo->output_scanline < cinfo->output_height) {
    const int y = static_cast<int>(cinfo->output_scanline);
    jpeg_read_scanlines(cinfo.get(), &row_ptr, 1);
    for (int x = 0; x < width; ++x) {
      image.at<cv::Vec3b>(y, x) =
          cv::Vec3b(row[x * 3 + 2], row[x * 3 + 1], row[x * 3 + 0]);
    }
  }
  jpeg_finish_decompress(cinfo.get());
  jpeg_destroy_decompress(cinfo.get());
  std::fclose(fp);
  return image;
}

} // namespace detail

inline std::string download_model(const std::string &repo,
                                  const std::string &filename) {
  const char *dir = std::getenv("YOLO_ROS_TEST_MODEL_DIR");
  if (dir != nullptr) {
    const std::string local = std::string(dir) + "/" + filename;
    if (access(local.c_str(), R_OK) == 0) {
      return local;
    }
    return std::string();
  }
  const auto result =
      huggingface_hub::hf_hub_download_with_shards(repo, filename);
  return result.success ? result.path : std::string();
}

inline cv::Mat download_image(const std::string &url,
                              const char *override_env) {
  const char *override_path = std::getenv(override_env);
  if (override_path != nullptr) {
    return detail::decode_jpeg(override_path);
  }
  CURL *curl = curl_easy_init();
  if (curl == nullptr) {
    return {};
  }
  char path_template[] = "/tmp/yolo_ros_test_image_XXXXXX";
  const int fd = mkstemp(path_template);
  if (fd < 0) {
    curl_easy_cleanup(curl);
    return {};
  }
  FILE *fp = fdopen(fd, "wb");
  if (fp == nullptr) {
    close(fd);
    std::remove(path_template);
    curl_easy_cleanup(curl);
    return {};
  }
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
  const CURLcode rc = curl_easy_perform(curl);
  std::fclose(fp);
  curl_easy_cleanup(curl);
  cv::Mat image;
  if (rc == CURLE_OK) {
    image = detail::decode_jpeg(path_template);
  }
  std::remove(path_template);
  return image;
}

inline cv::Mat download_sample_image() {
  return download_image("https://s3.amazonaws.com/images.cocodataset.org/"
                        "val2017/000000039769.jpg",
                        "YOLO_ROS_TEST_IMAGE");
}

inline cv::Mat download_people_image() {
  return download_image("https://s3.amazonaws.com/images.cocodataset.org/"
                        "val2017/000000000139.jpg",
                        "YOLO_ROS_TEST_PEOPLE_IMAGE");
}

inline yolo_ros::yolo::utils::YoloParams make_params(std::string model_path) {
  yolo_ros::yolo::utils::YoloParams params;
  params.model_path = std::move(model_path);
  params.threshold = 0.25f;
  params.iou = 0.45f;
  params.enable = true;
  params.max_det = 300;
  params.image_reliability = 2;
  params.max_fps = 0;
  params.n_threads = 1;
  params.top_k = 5;
  params.provider = "cuda"; // avoid TensorRT engine builds during tests
  return params;
}

} // namespace yolo_ros::test

#endif // YOLO_ROS__TEST__TEST_MODEL_HELPERS_HPP_
