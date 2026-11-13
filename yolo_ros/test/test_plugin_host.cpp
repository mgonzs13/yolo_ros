// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/plugin/plugin_host.hpp"

#include <gtest/gtest.h>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "support/fake_plugin.hpp"
#include "yolo_ros/camera/camera_streams.hpp"

namespace {

/// @brief Node, blackboard, registry and one validated camera for host tests.
struct HostFixture {
  std::shared_ptr<rclcpp_lifecycle::LifecycleNode> node;
  yolo_ros::Blackboard blackboard;
  yolo_ros::TopicRegistry topics;
  std::unique_ptr<yolo_ros::CameraStreams> streams;
  std::string error;

  explicit HostFixture(const std::vector<rclcpp::Parameter> &extra = {},
                       const std::vector<std::string> &cameras = {"cam0"}) {
    std::vector<rclcpp::Parameter> parameters;

    for (const auto &camera : cameras) {
      parameters.emplace_back(camera + ".rgb_topic", "/" + camera + "/image");
    }

    parameters.insert(parameters.end(), extra.begin(), extra.end());
    rclcpp::NodeOptions options;
    options.parameter_overrides(parameters);
    this->node =
        std::make_shared<rclcpp_lifecycle::LifecycleNode>("host_test", options);
    this->streams = std::make_unique<yolo_ros::CameraStreams>(
        this->blackboard, this->node->get_logger());
    EXPECT_TRUE(this->streams->configure(*this->node, cameras, this->error))
        << this->error;
  }
};

/// @brief Factory returning a FakePlugin named "a" for any plugin type.
yolo_ros::PluginHost::Factory
fake_factory(const std::shared_ptr<yolo_ros::test::FakePluginLog> &log) {
  return [log](const std::string &, std::string &) {
    return std::static_pointer_cast<yolo_ros::Plugin>(
        std::make_shared<yolo_ros::test::FakePlugin>(log, "a"));
  };
}

} // namespace

class PluginHostTest : public ::testing::Test {
protected:
  static void SetUpTestCase() {
    if (!rclcpp::ok()) {
      rclcpp::init(0, nullptr);
    }
  }
};

TEST_F(PluginHostTest, ConfigureDeclaresAllThenGetsAllThenSetsUp) {
  HostFixture fixture(
      {rclcpp::Parameter("p.plugin", "yolo_ros/DetectionPlugin")});
  auto log = std::make_shared<yolo_ros::test::FakePluginLog>();

  yolo_ros::PluginHost host(*fixture.node, fixture.blackboard, fixture.topics,
                            *fixture.streams, nullptr, fake_factory(log));

  std::string error;
  ASSERT_TRUE(host.configure({"p"}, error)) << error;
  auto calls = log->snapshot();
  ASSERT_EQ(calls.size(), 3u);
  EXPECT_EQ(calls[0], "declare:a");
  EXPECT_EQ(calls[1], "get:a");
  EXPECT_EQ(calls[2], "setup:a");
  EXPECT_TRUE(fixture.node->has_parameter("p.value"));

  ASSERT_EQ(log->cameras.at("p").size(), 1u);
  EXPECT_EQ(log->cameras.at("p")[0].name, "cam0");
  EXPECT_EQ(log->cameras.at("p")[0].frame_channel, "cam0");
  EXPECT_EQ(log->cameras.at("p")[0].input_channel, "cam0");
}

TEST_F(PluginHostTest, ReconfigureAfterCleanupReappliesOverrides) {
  HostFixture fixture(
      {rclcpp::Parameter("p.plugin", "yolo_ros/DetectionPlugin"),
       rclcpp::Parameter("p.value", 11)});
  std::string error;

  auto first_log = std::make_shared<yolo_ros::test::FakePluginLog>();
  yolo_ros::PluginHost first(*fixture.node, fixture.blackboard, fixture.topics,
                             *fixture.streams, nullptr,
                             fake_factory(first_log));
  ASSERT_TRUE(first.configure({"p"}, error)) << error;
  EXPECT_EQ(fixture.node->get_parameter("p.value").as_int(), 11);
  first.cleanup();

  // Simulate a value changed while inactive: Foxy allows undeclaring the
  // statically typed parameter, so it is gone after cleanup; on the other
  // distributions the override must be reapplied on the next configure.
  if (fixture.node->has_parameter("p.value")) {
    fixture.node->set_parameter(rclcpp::Parameter("p.value", 42));
    EXPECT_EQ(fixture.node->get_parameter("p.value").as_int(), 42);
  }

  auto second_log = std::make_shared<yolo_ros::test::FakePluginLog>();
  yolo_ros::PluginHost second(*fixture.node, fixture.blackboard, fixture.topics,
                              *fixture.streams, nullptr,
                              fake_factory(second_log));
  ASSERT_TRUE(second.configure({"p"}, error)) << error;
  EXPECT_EQ(fixture.node->get_parameter("p.value").as_int(), 11);
  EXPECT_TRUE(second_log->contains("get:a"));
  EXPECT_TRUE(second_log->contains("setup:a"));
}

TEST_F(PluginHostTest,
       ConfigureDeclaresAllInstancesBeforeAnyGetAndDeactivatesInReverse) {
  HostFixture fixture(
      {rclcpp::Parameter("p.plugin", "yolo_ros/DetectionPlugin"),
       rclcpp::Parameter("q.plugin", "yolo_ros/TrackingPlugin")});
  auto log = std::make_shared<yolo_ros::test::FakePluginLog>();
  int created = 0;

  yolo_ros::PluginHost host(
      *fixture.node, fixture.blackboard, fixture.topics, *fixture.streams,
      nullptr, [log, &created](const std::string &, std::string &) {
        const std::string name = (created++ == 0) ? "a" : "b";
        return std::static_pointer_cast<yolo_ros::Plugin>(
            std::make_shared<yolo_ros::test::FakePlugin>(log, name));
      });

  std::string error;
  ASSERT_TRUE(host.configure({"p", "q"}, error)) << error;
  const std::vector<std::string> expected{"declare:a", "declare:b", "get:a",
                                          "get:b",     "setup:a",   "setup:b"};
  EXPECT_EQ(log->snapshot(), expected);

  ASSERT_TRUE(host.activate(error)) << error;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!(log->contains("run:a") && log->contains("run:b")) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  ASSERT_TRUE(log->contains("run:a"));
  ASSERT_TRUE(log->contains("run:b"));

  host.deactivate();
  const auto calls = log->snapshot();
  ASSERT_GE(calls.size(), 2u);
  EXPECT_EQ(calls[calls.size() - 2], "deactivate:b");
  EXPECT_EQ(calls[calls.size() - 1], "deactivate:a");
}

TEST_F(PluginHostTest, ActivateStartsThreadsAndDeactivateJoins) {
  HostFixture fixture(
      {rclcpp::Parameter("p.plugin", "yolo_ros/DetectionPlugin")});
  auto log = std::make_shared<yolo_ros::test::FakePluginLog>();
  auto instance = std::make_shared<yolo_ros::test::FakePlugin>(log, "a");

  yolo_ros::PluginHost host(
      *fixture.node, fixture.blackboard, fixture.topics, *fixture.streams,
      nullptr, [instance](const std::string &, std::string &) {
        return std::static_pointer_cast<yolo_ros::Plugin>(instance);
      });

  std::string error;
  ASSERT_TRUE(host.configure({"p"}, error)) << error;
  ASSERT_TRUE(host.activate(error)) << error;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!log->contains("run:a") &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  ASSERT_TRUE(log->contains("run:a"));
  host.deactivate();
  EXPECT_TRUE(log->contains("deactivate:a"));
}

TEST_F(PluginHostTest, MalformedSpecFailsConfigure) {
  HostFixture fixture;
  yolo_ros::PluginHost host(*fixture.node, fixture.blackboard, fixture.topics,
                            *fixture.streams, nullptr,
                            [](const std::string &, std::string &) {
                              return std::shared_ptr<yolo_ros::Plugin>();
                            });
  std::string error;
  EXPECT_FALSE(host.configure({"bad.name"}, error));
  EXPECT_FALSE(error.empty());
}

TEST_F(PluginHostTest, MissingPluginFieldFailsConfigure) {
  HostFixture fixture;
  yolo_ros::PluginHost host(*fixture.node, fixture.blackboard, fixture.topics,
                            *fixture.streams, nullptr,
                            [](const std::string &, std::string &) {
                              return std::shared_ptr<yolo_ros::Plugin>();
                            });
  std::string error;
  EXPECT_FALSE(host.configure({"noplugin"}, error));
  EXPECT_NE(error.find("noplugin.plugin"), std::string::npos) << error;
}

TEST_F(PluginHostTest, WiresChainInOrder) {
  rclcpp::NodeOptions options;
  options.parameter_overrides(
      {rclcpp::Parameter("cam0.rgb_topic", "/camera0/image"),
       rclcpp::Parameter("det.plugin", "yolo_ros/DetectionPlugin"),
       rclcpp::Parameter("dbg.plugin", "yolo_ros/DebugPlugin")});
  auto node =
      std::make_shared<rclcpp_lifecycle::LifecycleNode>("host_test", options);
  yolo_ros::Blackboard blackboard;
  yolo_ros::TopicRegistry topics;
  yolo_ros::CameraStreams streams(blackboard, node->get_logger());
  std::string error;
  ASSERT_TRUE(streams.configure(*node, {"cam0"}, error)) << error;

  auto log = std::make_shared<yolo_ros::test::FakePluginLog>();
  yolo_ros::PluginHost host(
      *node, blackboard, topics, streams, nullptr,
      [log](const std::string &type, std::string &) {
        const std::string name =
            type == "yolo_ros/DetectionPlugin" ? "det" : "dbg";
        return std::static_pointer_cast<yolo_ros::Plugin>(
            std::make_shared<yolo_ros::test::FakePlugin>(log, name));
      });

  ASSERT_TRUE(host.configure({"det", "dbg"}, error)) << error;
  const auto &contexts = log->cameras;
  ASSERT_EQ(contexts.at("det").size(), 1u);
  EXPECT_EQ(contexts.at("det")[0].input_channel, "cam0");
  EXPECT_TRUE(contexts.at("det")[0].upstream_channels.empty());
  ASSERT_EQ(contexts.at("dbg").size(), 1u);
  EXPECT_EQ(contexts.at("dbg")[0].input_channel, "cam0/fake");
  ASSERT_EQ(contexts.at("dbg")[0].upstream_channels.size(), 1u);
  EXPECT_EQ(contexts.at("dbg")[0].upstream_channels[0], "cam0/fake");
}

TEST_F(PluginHostTest, RejectsUnknownCamera) {
  HostFixture fixture(
      {rclcpp::Parameter("p.plugin", "yolo_ros/DetectionPlugin"),
       rclcpp::Parameter("p.cameras", std::vector<std::string>{"missing"})});
  auto log = std::make_shared<yolo_ros::test::FakePluginLog>();
  yolo_ros::PluginHost host(*fixture.node, fixture.blackboard, fixture.topics,
                            *fixture.streams, nullptr, fake_factory(log));
  std::string error;
  EXPECT_FALSE(host.configure({"p"}, error));
  EXPECT_NE(error.find("unknown camera"), std::string::npos) << error;
}

TEST_F(PluginHostTest, RejectsDuplicateCameraEntries) {
  HostFixture fixture(
      {rclcpp::Parameter("p.plugin", "yolo_ros/DetectionPlugin"),
       rclcpp::Parameter("p.cameras",
                         std::vector<std::string>{"cam0", "cam0"})});
  auto log = std::make_shared<yolo_ros::test::FakePluginLog>();
  yolo_ros::PluginHost host(*fixture.node, fixture.blackboard, fixture.topics,
                            *fixture.streams, nullptr, fake_factory(log));
  std::string error;
  EXPECT_FALSE(host.configure({"p"}, error));
  EXPECT_NE(error.find("duplicate camera"), std::string::npos) << error;
}

TEST_F(PluginHostTest, EmptyCameraListExpandsToAllNodeCameras) {
  HostFixture fixture(
      {rclcpp::Parameter("p.plugin", "yolo_ros/DetectionPlugin")},
      {"cam0", "cam1"});
  auto log = std::make_shared<yolo_ros::test::FakePluginLog>();
  yolo_ros::PluginHost host(*fixture.node, fixture.blackboard, fixture.topics,
                            *fixture.streams, nullptr, fake_factory(log));
  std::string error;
  ASSERT_TRUE(host.configure({"p"}, error)) << error;
  ASSERT_EQ(log->cameras.at("p").size(), 2u);
  EXPECT_EQ(log->cameras.at("p")[0].name, "cam0");
  EXPECT_EQ(log->cameras.at("p")[1].name, "cam1");
}

TEST_F(PluginHostTest, RejectsCameraNotProducedByPreviousPlugin) {
  HostFixture fixture(
      {rclcpp::Parameter("p.plugin", "yolo_ros/DetectionPlugin"),
       rclcpp::Parameter("q.plugin", "yolo_ros/TrackingPlugin"),
       rclcpp::Parameter("p.cameras", std::vector<std::string>{"cam0"}),
       rclcpp::Parameter("q.cameras", std::vector<std::string>{"cam1"})},
      {"cam0", "cam1"});
  auto log = std::make_shared<yolo_ros::test::FakePluginLog>();
  yolo_ros::PluginHost host(*fixture.node, fixture.blackboard, fixture.topics,
                            *fixture.streams, nullptr, fake_factory(log));
  std::string error;
  EXPECT_FALSE(host.configure({"p", "q"}, error));
  EXPECT_NE(error.find("is not produced by plugin 'p'"), std::string::npos)
      << error;
}

TEST_F(PluginHostTest, RejectsChainViolations) {
  auto factory = [](const std::string &, std::string &) {
    return std::static_pointer_cast<yolo_ros::Plugin>(
        std::make_shared<yolo_ros::test::FakePlugin>(
            std::make_shared<yolo_ros::test::FakePluginLog>(), "x"));
  };

  {
    rclcpp::NodeOptions options;
    options.parameter_overrides(
        {rclcpp::Parameter("cam0.rgb_topic", "/camera0/image"),
         rclcpp::Parameter("p.plugin", "yolo_ros/TrackingPlugin")});
    auto node =
        std::make_shared<rclcpp_lifecycle::LifecycleNode>("host_test", options);
    yolo_ros::Blackboard blackboard;
    yolo_ros::TopicRegistry topics;
    yolo_ros::CameraStreams streams(blackboard, node->get_logger());
    std::string stream_error;
    ASSERT_TRUE(streams.configure(*node, {"cam0"}, stream_error))
        << stream_error;
    yolo_ros::PluginHost host(*node, blackboard, topics, streams, nullptr,
                              factory);
    std::string error;
    EXPECT_FALSE(host.configure({"p"}, error));
    EXPECT_NE(error.find("first plugin"), std::string::npos) << error;
  }

  {
    rclcpp::NodeOptions options;
    options.parameter_overrides(
        {rclcpp::Parameter("cam0.rgb_topic", "/camera0/image"),
         rclcpp::Parameter("p.plugin", "yolo_ros/DetectionPlugin"),
         rclcpp::Parameter("q.plugin", "yolo_ros/DetectionPlugin")});
    auto node =
        std::make_shared<rclcpp_lifecycle::LifecycleNode>("host_test", options);
    yolo_ros::Blackboard blackboard;
    yolo_ros::TopicRegistry topics;
    yolo_ros::CameraStreams streams(blackboard, node->get_logger());
    std::string stream_error;
    ASSERT_TRUE(streams.configure(*node, {"cam0"}, stream_error))
        << stream_error;
    yolo_ros::PluginHost host(*node, blackboard, topics, streams, nullptr,
                              factory);
    std::string error;
    EXPECT_FALSE(host.configure({"p", "q"}, error));
    EXPECT_NE(error.find("duplicate plugin type"), std::string::npos) << error;
  }

  {
    rclcpp::NodeOptions options;
    options.parameter_overrides(
        {rclcpp::Parameter("cam0.rgb_topic", "/camera0/image"),
         rclcpp::Parameter("p.plugin", "yolo_ros/DetectionPlugin"),
         rclcpp::Parameter("q.plugin", "yolo_ros/DebugPlugin"),
         rclcpp::Parameter("r.plugin", "yolo_ros/DetectionPlugin")});
    auto node =
        std::make_shared<rclcpp_lifecycle::LifecycleNode>("host_test", options);
    yolo_ros::Blackboard blackboard;
    yolo_ros::TopicRegistry topics;
    yolo_ros::CameraStreams streams(blackboard, node->get_logger());
    std::string stream_error;
    ASSERT_TRUE(streams.configure(*node, {"cam0"}, stream_error))
        << stream_error;
    yolo_ros::PluginHost host(*node, blackboard, topics, streams, nullptr,
                              factory);
    std::string error;
    EXPECT_FALSE(host.configure({"p", "q", "r"}, error));
    EXPECT_NE(error.find("last plugin"), std::string::npos) << error;
  }
}
