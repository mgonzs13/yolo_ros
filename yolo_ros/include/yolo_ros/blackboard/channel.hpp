// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief Per-consumer queues used by the Blackboard.

#ifndef YOLO_ROS__BLACKBOARD__CHANNEL_HPP_
#define YOLO_ROS__BLACKBOARD__CHANNEL_HPP_

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <utility>

namespace yolo_ros {

/// @brief Bounded, thread-safe queue filled by the publisher and drained by
/// exactly one consumer thread.
///
/// Depth 1 keeps only the latest item (images); depth > 1 drops the oldest
/// item on overflow (detections), mirroring a ROS QoS keep-last policy.
template <typename T> class ConsumerQueue {
public:
  /// @brief Create a queue with @p depth slots.
  /// @param[in] depth Buffer depth; 0 is clamped to 1.
  explicit ConsumerQueue(std::size_t depth) : depth_(depth == 0 ? 1 : depth) {}

  /// @brief Enqueue @p item, dropping the oldest entry when full.
  /// @param[in] item Item to enqueue; ignored once the queue is closed.
  void push(std::shared_ptr<const T> item) {
    {
      std::lock_guard<std::mutex> lock(this->mutex_);

      if (this->closed_) {
        return;
      }

      if (this->depth_ == 1) {
        if (!this->items_.empty()) {
          ++this->dropped_;
        }

        this->items_.clear();
      } else if (this->items_.size() >= this->depth_) {
        this->items_.pop_front();
        ++this->dropped_;

        if (this->dropped_ % 1000 == 1) {
          fprintf(stderr, "[blackboard] queue overflow: %llu item(s) dropped\n",
                  static_cast<unsigned long long>(this->dropped_));
        }
      }

      this->items_.push_back(std::move(item));
    }
    this->cv_.notify_all();
  }

  /// @brief Wait for an item, a wake/close or @p timeout.
  /// @param[out] out Receives the popped item on success.
  /// @param[in] timeout Maximum time to block.
  /// @return True when an item was popped.
  bool wait(std::shared_ptr<const T> &out, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(this->mutex_);
    const std::uint64_t generation = this->wake_generation_;
    this->cv_.wait_for(lock, timeout, [this, generation] {
      return !this->items_.empty() || this->closed_ ||
             this->wake_generation_ != generation;
    });

    if (this->items_.empty()) {
      return false;
    }

    out = std::move(this->items_.front());
    this->items_.pop_front();
    return true;
  }

  /// @brief Pop an item without blocking.
  /// @param[out] out Receives the popped item on success.
  /// @return True when an item was popped.
  bool try_pop(std::shared_ptr<const T> &out) {
    std::lock_guard<std::mutex> lock(this->mutex_);

    if (this->items_.empty()) {
      return false;
    }

    out = std::move(this->items_.front());
    this->items_.pop_front();
    return true;
  }

  /// @brief Wake blocked readers without closing (deactivation).
  void wake() {
    {
      std::lock_guard<std::mutex> lock(this->mutex_);
      ++this->wake_generation_;
    }
    this->cv_.notify_all();
  }

  /// @brief Permanently close the queue; further waits return false.
  void close() {
    {
      std::lock_guard<std::mutex> lock(this->mutex_);
      this->closed_ = true;
      ++this->wake_generation_;
    }
    this->cv_.notify_all();
  }

  /// @brief Items dropped on overflow since construction.
  /// @return The dropped-item count.
  std::uint64_t dropped() const {
    std::lock_guard<std::mutex> lock(this->mutex_);
    return this->dropped_;
  }

private:
  /// @brief Guards every field below.
  mutable std::mutex mutex_;
  /// @brief Signalled on push, wake and close.
  std::condition_variable cv_;
  /// @brief Buffered items, oldest first.
  std::deque<std::shared_ptr<const T>> items_;
  /// @brief Buffer depth; 1 keeps only the latest item.
  std::size_t depth_;
  /// @brief True once close() has been called.
  bool closed_ = false;
  /// @brief Bumped by wake()/close() to interrupt blocked readers.
  std::uint64_t wake_generation_ = 0;
  /// @brief Items dropped because the queue was full.
  std::uint64_t dropped_ = 0;
};

/// @brief Consumer-side handle over a blackboard channel.
template <typename T> class ChannelReader {
public:
  /// @brief Create an invalid reader with no queue attached.
  ChannelReader() = default;
  /// @brief Create a reader draining @p queue.
  /// @param[in] queue Queue to read from.
  explicit ChannelReader(std::shared_ptr<ConsumerQueue<T>> queue)
      : queue_(std::move(queue)) {}

  /// @brief Wait for an item, a wake/close or @p timeout.
  /// @param[out] out Receives the popped item on success.
  /// @param[in] timeout Maximum time to block.
  /// @return True when an item was popped.
  bool wait(std::shared_ptr<const T> &out, std::chrono::milliseconds timeout) {
    return this->queue_ && this->queue_->wait(out, timeout);
  }

  /// @brief Pop an item without blocking.
  /// @param[out] out Receives the popped item on success.
  /// @return True when an item was popped.
  bool try_pop(std::shared_ptr<const T> &out) {
    return this->queue_ && this->queue_->try_pop(out);
  }

  /// @brief Whether a queue is attached.
  /// @return True when the reader is usable.
  bool valid() const { return static_cast<bool>(this->queue_); }

  /// @brief Close the underlying queue, waking blocked readers.
  void close() {
    if (this->queue_) {
      this->queue_->close();
    }
  }

  /// @brief Items dropped by the underlying queue.
  /// @return The dropped-item count; 0 without a queue.
  std::uint64_t dropped() const {
    return this->queue_ ? this->queue_->dropped() : 0;
  }

private:
  /// @brief Queue being drained; null when default-constructed.
  std::shared_ptr<ConsumerQueue<T>> queue_;
};

} // namespace yolo_ros

#endif // YOLO_ROS__BLACKBOARD__CHANNEL_HPP_
