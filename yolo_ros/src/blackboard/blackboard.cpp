// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/blackboard/blackboard.hpp"

#include <utility>

namespace yolo_ros {

bool Blackboard::has_producer(const std::string &channel) const {
  std::lock_guard<std::mutex> lock(this->mutex_);
  auto it = this->channels_.find(channel);
  return it != this->channels_.end() && it->second->has_producer;
}

bool Blackboard::wait_for_activity(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(this->activity_mutex_);
  const std::uint64_t generation = this->activity_generation_;
  return this->activity_cv_.wait_for(lock, timeout, [this, generation] {
    return this->activity_generation_ != generation || this->closed_;
  });
}

bool Blackboard::closed() const {
  std::lock_guard<std::mutex> lock(this->activity_mutex_);
  return this->closed_;
}

void Blackboard::wake_all() {
  {
    // Sleepers wake under mutex_ so concurrent subscribe() calls cannot race
    // the consumers vector; lock order is Blackboard::mutex_ then queue mutex.
    std::lock_guard<std::mutex> lock(this->mutex_);

    for (auto &entry : this->channels_) {
      entry.second->wake();
    }
  }
  this->notify_activity();
}

void Blackboard::close_all() {
  {
    // See wake_all() for the lock order rationale.
    std::lock_guard<std::mutex> lock(this->mutex_);

    for (auto &entry : this->channels_) {
      entry.second->close();
    }
  }
  {
    std::lock_guard<std::mutex> lock(this->activity_mutex_);
    this->closed_ = true;
  }
  this->activity_cv_.notify_all();
}

void Blackboard::reset() {
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->channels_.clear();
    this->publish_hook_ = nullptr;
  }
  {
    std::lock_guard<std::mutex> lock(this->activity_mutex_);
    this->closed_ = false;
  }
}

void Blackboard::set_publish_hook(PublishHook hook) {
  std::lock_guard<std::mutex> lock(this->mutex_);
  this->publish_hook_ = std::move(hook);
}

void Blackboard::notify_activity() {
  {
    std::lock_guard<std::mutex> lock(this->activity_mutex_);
    ++this->activity_generation_;
  }
  this->activity_cv_.notify_all();
}

} // namespace yolo_ros
