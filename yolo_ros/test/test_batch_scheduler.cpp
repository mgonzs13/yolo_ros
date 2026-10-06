// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <mutex>
#include <utility>
#include <vector>

#include "yolo_ros/engine/batch_scheduler.hpp"

namespace yolo_ros {
namespace {

using Batch = std::vector<std::pair<std::size_t, int>>;
using BatchScheduler = yolo_ros::engine::BatchScheduler<int>;

class Collector {
public:
  void add(Batch batch) {
    std::lock_guard<std::mutex> lock(mutex_);
    batch_sizes_.push_back(batch.size());
    for (auto &item : batch) {
      items_.push_back(item);
    }
    cond_.notify_all();
  }
  bool wait_for(std::size_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cond_.wait_for(lock, std::chrono::seconds(5),
                          [&] { return items_.size() >= count; });
  }
  std::vector<std::pair<std::size_t, int>> items() {
    std::lock_guard<std::mutex> lock(mutex_);
    return items_;
  }
  std::vector<std::size_t> batch_sizes() {
    std::lock_guard<std::mutex> lock(mutex_);
    return batch_sizes_;
  }

private:
  std::mutex mutex_;
  std::condition_variable cond_;
  std::vector<std::pair<std::size_t, int>> items_;
  std::vector<std::size_t> batch_sizes_;
};

// A camera that never publishes must not hold up the others.
TEST(BatchScheduler, BatchesOnlyReadyCameras) {
  BatchScheduler scheduler(3, 8);
  Collector collector;
  scheduler.start([&](Batch batch) { collector.add(std::move(batch)); });
  scheduler.push(0, 10);
  scheduler.push(2, 30);
  ASSERT_TRUE(collector.wait_for(2));
  const auto items = collector.items();
  ASSERT_EQ(items.size(), 2u);
  for (const auto &item : items) {
    EXPECT_NE(item.first, 1u); // the idle camera never appears
  }
  scheduler.stop();
}

// A camera faster than the batch cycle overwrites its slot: the stale frame
// must never be processed.
TEST(BatchScheduler, LatestFrameWins) {
  BatchScheduler scheduler(1, 8);
  scheduler.push(0, 1);
  scheduler.push(0, 2);
  Collector collector;
  scheduler.start([&](Batch batch) { collector.add(std::move(batch)); });
  ASSERT_TRUE(collector.wait_for(1));
  const auto items = collector.items();
  ASSERT_EQ(items.size(), 1u);
  EXPECT_EQ(items[0].first, 0u);
  EXPECT_EQ(items[0].second, 2);
  scheduler.stop();
}

TEST(BatchScheduler, RespectsMaxBatch) {
  BatchScheduler scheduler(3, 2);
  Collector collector;
  scheduler.start([&](Batch batch) { collector.add(std::move(batch)); });
  scheduler.push(0, 0);
  scheduler.push(1, 1);
  scheduler.push(2, 2);
  ASSERT_TRUE(collector.wait_for(3));
  for (const std::size_t size : collector.batch_sizes()) {
    EXPECT_LE(size, 2u);
  }
  scheduler.stop();
}

TEST(BatchScheduler, StopJoinsWhileBusy) {
  BatchScheduler scheduler(2, 8);
  std::atomic<int> calls{0};
  std::promise<void> first_call;
  auto future = first_call.get_future();
  scheduler.start([&](Batch) {
    if (calls.fetch_add(1) == 0) {
      first_call.set_value();
    }
  });
  scheduler.push(0, 1);
  ASSERT_EQ(future.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  scheduler.stop(); // must join without deadlocking
  SUCCEED();
}

} // namespace
} // namespace yolo_ros
