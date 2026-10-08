// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/yolo_node.hpp"

#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "lifecycle_msgs/msg/state.hpp"
#include "support/fake_plugin.hpp"

namespace {

/// @brief Plugin whose activate() always fails, to exercise on_activate.
class FailingActivatePlugin : public yolo_ros::Plugin {
public:
  void declare_params(rclcpp_lifecycle::LifecycleNode &,
                      const std::string &) override {}
  void get_params(const rclcpp_lifecycle::LifecycleNode &,
                  const std::string &) override {}
  std::string output_channel(const std::string &camera) const override {
    return camera + "/failing";
  }
  bool activate() override { return false; }
  void deactivate() override {}
  void run(const std::atomic<bool> &) override {}
};

/// @brief Plugin whose setup() always fails, to poison one configure cycle.
class FailingSetupPlugin : public yolo_ros::test::FakePlugin {
public:
  explicit FailingSetupPlugin(
      const std::shared_ptr<yolo_ros::test::FakePluginLog> &log)
      : yolo_ros::test::FakePlugin(log, "failing") {}

  bool setup(yolo_ros::PluginContext &) override { return false; }
};

} // namespace

class YoloNodeTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }
};

TEST_F(YoloNodeTest, LifecycleRunsPlugins) {
  rclcpp::NodeOptions options;
  options.arguments({"--ros-args", "-r", "__node:=test_lifecycle_plugins"});
  options.parameter_overrides(
      {rclcpp::Parameter("plugins", std::vector<std::string>{"p"}),
       rclcpp::Parameter("p.plugin", "yolo_ros/DetectionPlugin")});
  auto log = std::make_shared<yolo_ros::test::FakePluginLog>();

  yolo_ros::YoloNode node(
      options, [log](const std::string &type, std::string &error) {
        if (type != "yolo_ros/DetectionPlugin") {
          error = "unknown plugin " + type;
          return std::shared_ptr<yolo_ros::Plugin>();
        }
        return std::static_pointer_cast<yolo_ros::Plugin>(
            std::make_shared<yolo_ros::test::FakePlugin>(log, "a"));
      });

  EXPECT_EQ(node.configure().id(),
            lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  EXPECT_EQ(node.activate().id(),
            lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!log->contains("run:a") &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  EXPECT_TRUE(log->contains("run:a"));
  EXPECT_EQ(node.deactivate().id(),
            lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  EXPECT_TRUE(log->contains("deactivate:a"));
  EXPECT_EQ(node.cleanup().id(),
            lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
}

TEST_F(YoloNodeTest, EmptyPluginListConfigures) {
  rclcpp::NodeOptions options;
  options.arguments({"--ros-args", "-r", "__node:=test_empty_plugins"});
  options.parameter_overrides(
      {rclcpp::Parameter("plugins", std::vector<std::string>{})});
  yolo_ros::YoloNode node(options);
  EXPECT_EQ(node.configure().id(),
            lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  EXPECT_EQ(node.deactivate().id(),
            lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  EXPECT_EQ(node.cleanup().id(),
            lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
}

TEST_F(YoloNodeTest, FailedActivateTearsDownAndCanReconfigure) {
  rclcpp::NodeOptions options;
  options.arguments({"--ros-args", "-r", "__node:=test_failed_activate"});
  options.parameter_overrides(
      {rclcpp::Parameter("plugins", std::vector<std::string>{"p"}),
       rclcpp::Parameter("p.plugin", "yolo_ros/DetectionPlugin")});
  yolo_ros::YoloNode node(options, [](const std::string &, std::string &) {
    return std::static_pointer_cast<yolo_ros::Plugin>(
        std::make_shared<FailingActivatePlugin>());
  });

  EXPECT_EQ(node.configure().id(),
            lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_NE(node.host(), nullptr);

  // Jazzy maps a failed on_activate back to inactive (it does NOT go to
  // unconfigured), so assert the actual state; on_activate must nevertheless
  // tear the node down itself, leaving no stale host for the next configure().
  EXPECT_EQ(node.activate().id(),
            lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  EXPECT_EQ(node.host(), nullptr);

  EXPECT_EQ(node.cleanup().id(),
            lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  EXPECT_EQ(node.configure().id(),
            lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  EXPECT_EQ(node.cleanup().id(),
            lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
}

TEST_F(YoloNodeTest, ReconfigureAfterFailedConfigureSucceeds) {
  rclcpp::NodeOptions options;
  options.arguments({"--ros-args", "-r", "__node:=test_reconfigure_failed"});
  options.parameter_overrides(
      {rclcpp::Parameter("plugins", std::vector<std::string>{"p"}),
       rclcpp::Parameter("p.plugin", "yolo_ros/DetectionPlugin")});
  auto failing_log = std::make_shared<yolo_ros::test::FakePluginLog>();
  auto working_log = std::make_shared<yolo_ros::test::FakePluginLog>();
  int created = 0;

  yolo_ros::YoloNode node(options, [&created, failing_log, working_log](
                                       const std::string &, std::string &) {
    if (created++ == 0) {
      return std::static_pointer_cast<yolo_ros::Plugin>(
          std::make_shared<FailingSetupPlugin>(failing_log));
    }
    return std::static_pointer_cast<yolo_ros::Plugin>(
        std::make_shared<yolo_ros::test::FakePlugin>(working_log, "a"));
  });

  EXPECT_EQ(node.configure().id(),
            lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  EXPECT_EQ(node.host(), nullptr);

  EXPECT_EQ(node.configure().id(),
            lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_NE(node.host(), nullptr);
  EXPECT_TRUE(working_log->contains("setup:a"));
  EXPECT_EQ(node.cleanup().id(),
            lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
}

TEST_F(YoloNodeTest, UnknownPluginFailsConfigure) {
  rclcpp::NodeOptions options;
  options.arguments({"--ros-args", "-r", "__node:=test_unknown_plugin"});
  options.parameter_overrides(
      {rclcpp::Parameter("plugins", std::vector<std::string>{"p"}),
       rclcpp::Parameter("p.plugin", "yolo_ros/DoesNotExist")});
  yolo_ros::YoloNode node(options);
  EXPECT_EQ(node.configure().id(),
            lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
}

TEST_F(YoloNodeTest, MissingPluginFieldFailsConfigure) {
  rclcpp::NodeOptions options;
  options.arguments({"--ros-args", "-r", "__node:=test_missing_plugin_field"});
  options.parameter_overrides(
      {rclcpp::Parameter("plugins", std::vector<std::string>{"p"})});
  yolo_ros::YoloNode node(options);
  EXPECT_EQ(node.configure().id(),
            lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
}
