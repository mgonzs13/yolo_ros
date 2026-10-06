// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

/// @file
/// @brief Latest-frame-wins micro-batcher used by the multi-camera node.

#ifndef YOLO_ROS__ENGINE__BATCH_SCHEDULER_HPP_
#define YOLO_ROS__ENGINE__BATCH_SCHEDULER_HPP_

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

/// @addtogroup yolo_engine
/// @{
namespace yolo_ros::engine {

/// @brief Collects the latest frame per camera and runs a callback on a batch.
///
/// Each camera owns one slot: a newer frame overwrites the pending one, so a
/// camera faster than the batch cycle cannot build up latency, and a silent
/// camera never withholds the others. A worker thread drains every ready slot
/// (up to @p max_batch) in one callback. @p T must be movable.
/// @tparam T Payload stored per camera.
template <typename T> class BatchScheduler {
public:
  /// @brief One (camera index, payload) pair in a drained batch.
  using Batch = std::vector<std::pair<std::size_t, T>>;
  /// @brief Callback run on the worker thread for each drained batch.
  using Callback = std::function<void(Batch)>;

  /// @brief Construct with a fixed camera count and batch ceiling.
  /// @param num_cameras Number of slots (cameras).
  /// @param max_batch Maximum frames per callback (clamped to >= 1).
  BatchScheduler(std::size_t num_cameras, std::size_t max_batch)
      : slots_(num_cameras), ready_(num_cameras, false),
        max_batch_(std::max<std::size_t>(1, max_batch)) {}

  /// @brief Stop the worker (if running) and destroy the scheduler.
  ~BatchScheduler() { this->stop(); }

  BatchScheduler(const BatchScheduler &) = delete;
  BatchScheduler &operator=(const BatchScheduler &) = delete;

  /// @brief Start the worker thread with @p callback.
  /// @param callback Invoked for each drained batch.
  void start(Callback callback) {
    {
      std::lock_guard<std::mutex> lock(this->mutex_);
      this->callback_ = std::move(callback);
      this->stopping_ = false;
    }
    this->worker_ = std::thread([this] { this->run(); });
  }

  /// @brief Signal the worker to stop and join it. Safe to call twice.
  void stop() {
    {
      std::lock_guard<std::mutex> lock(this->mutex_);

      if (!this->worker_.joinable()) {
        return;
      }

      this->stopping_ = true;
    }
    this->cond_.notify_all();
    this->worker_.join();
  }

  /// @brief Store @p frame as the latest pending frame for @p camera.
  /// @param camera Camera index; out-of-range values are ignored.
  /// @param frame Payload to enqueue (overwrites any pending one).
  void push(std::size_t camera, T frame) {
    if (camera >= this->slots_.size()) {
      return;
    }

    {
      std::lock_guard<std::mutex> lock(this->mutex_);
      this->slots_[camera] = std::move(frame);
      this->ready_[camera] = true;
    }
    this->cond_.notify_one();
  }

private:
  /// @brief Worker loop: wait for a ready slot, drain a batch, invoke the
  /// callback.
  void run() {
    for (;;) {
      Batch batch;
      {
        std::unique_lock<std::mutex> lock(this->mutex_);
        this->cond_.wait(
            lock, [this] { return this->stopping_ || this->any_ready(); });

        if (this->stopping_) {
          return; // drop pending frames on shutdown
        }

        for (std::size_t i = 0;
             i < this->slots_.size() && batch.size() < this->max_batch_; ++i) {
          if (this->ready_[i]) {
            batch.emplace_back(i, std::move(this->slots_[i]));
            this->ready_[i] = false;
          }
        }
      }
      this->callback_(std::move(batch));
    }
  }

  /// @brief Whether at least one slot holds a pending frame.
  bool any_ready() const {
    for (const bool ready : this->ready_) {
      if (ready) {
        return true;
      }
    }

    return false;
  }

  /// @brief Latest pending payload per camera.
  std::vector<T> slots_;
  /// @brief Per-camera flag: true when a pending payload is present.
  std::vector<bool> ready_;
  /// @brief Maximum frames drained per callback.
  std::size_t max_batch_;
  /// @brief Guards slots_, ready_, callback_ and stopping_.
  std::mutex mutex_;
  /// @brief Signalled on every push and on stop.
  std::condition_variable cond_;
  /// @brief Invoked for each drained batch.
  Callback callback_;
  /// @brief Drains ready slots on its own thread.
  std::thread worker_;
  /// @brief Set by stop() to make the worker exit.
  bool stopping_ = false;
};

} // namespace yolo_ros::engine
/// @}

#endif // YOLO_ROS__ENGINE__BATCH_SCHEDULER_HPP_
