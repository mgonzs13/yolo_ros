// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#ifndef YOLO_ROS__TEST__SUPPORT__FAKE_PLUGIN_HPP_
#define YOLO_ROS__TEST__SUPPORT__FAKE_PLUGIN_HPP_

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "yolo_ros/plugin/plugin.hpp"

namespace yolo_ros::test {

/// @brief Records every lifecycle call for assertions.
struct FakePluginLog {
  std::mutex mutex;
  std::vector<std::string> calls;
  std::map<std::string, std::vector<yolo_ros::CameraInput>> cameras;
  std::atomic<int> runs{0};

  void add(const std::string &entry) {
    std::lock_guard<std::mutex> lock(mutex);
    calls.push_back(entry);
  }
  std::vector<std::string> snapshot() {
    std::lock_guard<std::mutex> lock(mutex);
    return calls;
  }
  bool contains(const std::string &entry) {
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto &call : calls) {
      if (call == entry) {
        return true;
      }
    }
    return false;
  }
};

class FakePlugin : public Plugin {
public:
  explicit FakePlugin(std::shared_ptr<FakePluginLog> log,
                      std::string name = "fake")
      : log_(std::move(log)), name_(std::move(name)) {}

  void declare_params(rclcpp_lifecycle::LifecycleNode &node,
                      const std::string &prefix) override {
    node.declare_parameter<int>(prefix + "value", 7);
    log_->add("declare:" + name_);
  }

  void get_params(const rclcpp_lifecycle::LifecycleNode &node,
                  const std::string &prefix) override {
    value_ = node.get_parameter(prefix + "value").as_int();
    log_->add("get:" + name_);
  }

  std::string output_channel(const std::string &camera) const override {
    return camera + "/fake";
  }

  bool setup(PluginContext &ctx) override {
    {
      std::lock_guard<std::mutex> lock(log_->mutex);
      log_->cameras[ctx.name] = ctx.cameras;
    }
    log_->add("setup:" + name_);
    return true;
  }

  bool activate() override {
    log_->add("activate:" + name_);
    return true;
  }

  void deactivate() override { log_->add("deactivate:" + name_); }

  void run(const std::atomic<bool> &stop) override {
    log_->add("run:" + name_);
    ++log_->runs;
    while (!stop.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }

  int value() const { return value_; }

private:
  std::shared_ptr<FakePluginLog> log_;
  std::string name_;
  int value_ = 0;
};

} // namespace yolo_ros::test

#endif // YOLO_ROS__TEST__SUPPORT__FAKE_PLUGIN_HPP_
