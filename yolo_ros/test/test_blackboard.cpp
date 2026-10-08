// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/blackboard/blackboard.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>

using namespace std::chrono_literals;

TEST(Blackboard, FanOutToAllConsumers) {
  yolo_ros::Blackboard bb;
  auto a = bb.subscribe<int>("numbers", 4);
  auto b = bb.subscribe<int>("numbers", 4);
  bb.publish<int>("numbers", std::make_shared<const int>(7));
  std::shared_ptr<const int> out_a;
  std::shared_ptr<const int> out_b;
  EXPECT_TRUE(a.try_pop(out_a));
  EXPECT_TRUE(b.try_pop(out_b));
  EXPECT_EQ(*out_a, 7);
  EXPECT_EQ(*out_b, 7);
}

TEST(Blackboard, KeepLatestDropsPrevious) {
  yolo_ros::Blackboard bb;
  auto reader = bb.subscribe<int>("c", 1);
  bb.publish<int>("c", std::make_shared<const int>(1));
  bb.publish<int>("c", std::make_shared<const int>(2));
  std::shared_ptr<const int> out;
  ASSERT_TRUE(reader.try_pop(out));
  EXPECT_EQ(*out, 2);
  EXPECT_FALSE(reader.try_pop(out));
}

TEST(Blackboard, DropOldestKeepsNewest) {
  yolo_ros::Blackboard bb;
  auto reader = bb.subscribe<int>("c", 2);
  bb.publish<int>("c", std::make_shared<const int>(1));
  bb.publish<int>("c", std::make_shared<const int>(2));
  bb.publish<int>("c", std::make_shared<const int>(3));
  std::shared_ptr<const int> out;
  ASSERT_TRUE(reader.try_pop(out));
  EXPECT_EQ(*out, 2);
  EXPECT_EQ(reader.dropped(), 1u);
  ASSERT_TRUE(reader.try_pop(out));
  EXPECT_EQ(*out, 3);
}

TEST(Blackboard, WaitTimesOutAndWakesOnPublish) {
  yolo_ros::Blackboard bb;
  auto reader = bb.subscribe<int>("c", 1);
  std::shared_ptr<const int> out;
  EXPECT_FALSE(reader.wait(out, 20ms));
  std::thread producer([&bb] {
    std::this_thread::sleep_for(10ms);
    bb.publish<int>("c", std::make_shared<const int>(42));
  });
  EXPECT_TRUE(reader.wait(out, 500ms));
  EXPECT_EQ(*out, 42);
  producer.join();
}

TEST(Blackboard, WakeUnblocksWithoutClosing) {
  yolo_ros::Blackboard bb;
  auto reader = bb.subscribe<int>("c", 1);
  std::thread waker([&bb] {
    std::this_thread::sleep_for(10ms);
    bb.wake_all();
  });
  std::shared_ptr<const int> out;
  EXPECT_FALSE(reader.wait(out, 500ms));
  bb.publish<int>("c", std::make_shared<const int>(5));
  EXPECT_TRUE(reader.wait(out, 100ms));
  waker.join();
}

TEST(Blackboard, CloseAllEndsWaits) {
  yolo_ros::Blackboard bb;
  auto reader = bb.subscribe<int>("c", 1);
  bb.close_all();
  std::shared_ptr<const int> out;
  EXPECT_FALSE(reader.wait(out, 10ms));
}

TEST(Blackboard, LatestAndHasProducer) {
  yolo_ros::Blackboard bb;
  EXPECT_FALSE(bb.has_producer("x"));
  bb.declare_channel<int>("x");
  EXPECT_TRUE(bb.has_producer("x"));
  bb.publish<int>("x", std::make_shared<const int>(9));
  ASSERT_TRUE(bb.latest<int>("x"));
  EXPECT_EQ(*bb.latest<int>("x"), 9);
  EXPECT_EQ(bb.latest<int>("other"), nullptr);
}

TEST(Blackboard, TypeMismatchThrows) {
  yolo_ros::Blackboard bb;
  bb.declare_channel<int>("typed");
  EXPECT_THROW(bb.subscribe<double>("typed", 1), std::runtime_error);
  EXPECT_THROW(bb.publish<double>("typed", std::make_shared<const double>(1.0)),
               std::runtime_error);
}

TEST(Blackboard, ResetClearsChannels) {
  yolo_ros::Blackboard bb;
  bb.declare_channel<int>("x");
  bb.close_all();
  bb.reset();
  EXPECT_FALSE(bb.has_producer("x"));
  auto reader = bb.subscribe<int>("x", 1);
  bb.publish<int>("x", std::make_shared<const int>(1));
  std::shared_ptr<const int> out;
  EXPECT_TRUE(reader.try_pop(out));
}

TEST(Blackboard, WaitForActivityWakesOnPublish) {
  yolo_ros::Blackboard bb;
  std::atomic<bool> result{false};
  std::thread waiter([&bb, &result] { result = bb.wait_for_activity(500ms); });
  std::this_thread::sleep_for(10ms);
  bb.publish<int>("c", std::make_shared<const int>(1));
  waiter.join();
  EXPECT_TRUE(result.load());
}

TEST(Blackboard, WaitForActivityTimesOut) {
  yolo_ros::Blackboard bb;
  EXPECT_FALSE(bb.wait_for_activity(20ms));
}

TEST(Blackboard, PublishHookSeesChannelAndType) {
  yolo_ros::Blackboard bb;
  bool called = false;
  std::string seen_channel;
  const std::type_info *seen_type = nullptr;
  bb.set_publish_hook([&called, &seen_channel,
                       &seen_type](const std::string &channel,
                                   const std::type_info &type,
                                   const std::shared_ptr<const void> &) {
    called = true;
    seen_channel = channel;
    seen_type = &type;
  });
  bb.publish<int>("hooked", std::make_shared<const int>(3));
  EXPECT_TRUE(called);
  EXPECT_EQ(seen_channel, "hooked");
  EXPECT_EQ(seen_type, &typeid(int));
  called = false;
  bb.set_publish_hook(nullptr);
  bb.publish<int>("hooked", std::make_shared<const int>(4));
  EXPECT_FALSE(called);
}

TEST(Blackboard, CloseWakesBlockedReader) {
  yolo_ros::Blackboard bb;
  auto reader = bb.subscribe<int>("c", 1);
  std::atomic<bool> result{true};
  std::thread blocked([&reader, &result] {
    std::shared_ptr<const int> out;
    result = reader.wait(out, 5s);
  });
  std::this_thread::sleep_for(10ms);
  const auto start = std::chrono::steady_clock::now();
  bb.close_all();
  blocked.join();
  const auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_FALSE(result.load());
  EXPECT_LT(elapsed, 1s);
}

TEST(Blackboard, WakeUnblocksBlockedReader) {
  yolo_ros::Blackboard bb;
  auto reader = bb.subscribe<int>("c", 1);
  std::atomic<bool> result{true};
  std::thread blocked([&reader, &result] {
    std::shared_ptr<const int> out;
    result = reader.wait(out, 5s);
  });
  std::this_thread::sleep_for(10ms);
  const auto start = std::chrono::steady_clock::now();
  bb.wake_all();
  blocked.join();
  const auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_FALSE(result.load());
  EXPECT_LT(elapsed, 1s);
}

TEST(Blackboard, LatestTypeMismatchReturnsNullptr) {
  yolo_ros::Blackboard bb;
  bb.declare_channel<int>("typed");
  EXPECT_EQ(bb.latest<double>("typed"), nullptr);
}
