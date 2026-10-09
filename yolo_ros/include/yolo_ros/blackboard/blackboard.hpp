// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief Typed publish/subscribe bus shared by plugins.

#ifndef YOLO_ROS__BLACKBOARD__BLACKBOARD_HPP_
#define YOLO_ROS__BLACKBOARD__BLACKBOARD_HPP_

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "yolo_ros/blackboard/channel.hpp"

namespace yolo_ros {

/// @brief Thread-safe typed channel bus.
///
/// Channels are created on first use. Each consumer gets an independent
/// bounded queue so fan-out never blocks the producer. A plugin declares the
/// channels it will produce with declare_channel() so the TopicRegistry can
/// decide whether a ROS subscription is needed.
class Blackboard {
public:
  /// @brief Called after every publish with the type-erased payload. The
  /// TopicRegistry uses it to forward channel data to ROS publishers.
  using PublishHook =
      std::function<void(const std::string &channel, const std::type_info &type,
                         const std::shared_ptr<const void> &msg)>;

  /// @brief Register @p channel as produced by a plugin.
  template <typename T> void declare_channel(const std::string &channel) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    this->get_or_create<T>(channel)->has_producer = true;
  }

  /// @brief Deliver @p msg to every consumer of @p channel and run the hook.
  /// @param[in] channel Channel name to publish on.
  /// @param[in] msg Message to deliver.
  template <typename T>
  void publish(const std::string &channel, std::shared_ptr<const T> msg) {
    std::shared_ptr<Channel<T>> channel_state;
    std::vector<std::shared_ptr<ConsumerQueue<T>>> consumers;
    PublishHook hook;
    {
      std::lock_guard<std::mutex> lock(this->mutex_);
      channel_state = this->get_or_create<T>(channel);
      channel_state->has_producer = true;
      channel_state->latest = msg;
      consumers = channel_state->consumers;
      hook = this->publish_hook_;
    }

    for (auto &consumer : consumers) {
      consumer->push(msg);
    }

    if (hook) {
      hook(channel, typeid(T), std::static_pointer_cast<const void>(msg));
    }

    this->notify_activity();
  }

  /// @brief Create an independent consumer queue of @p depth entries.
  /// @param[in] channel Channel to subscribe to.
  /// @param[in] depth Queue depth for the new consumer.
  /// @return Reader draining the new queue.
  template <typename T>
  ChannelReader<T> subscribe(const std::string &channel, std::size_t depth) {
    std::lock_guard<std::mutex> lock(this->mutex_);
    auto channel_state = this->get_or_create<T>(channel);
    auto queue = std::make_shared<ConsumerQueue<T>>(depth);
    channel_state->consumers.push_back(queue);
    return ChannelReader<T>(queue);
  }

  /// @brief Last published value of @p channel, or nullptr.
  /// @param[in] channel Channel to query.
  /// @return The latest value, or nullptr when absent or type-mismatched.
  template <typename T>
  std::shared_ptr<const T> latest(const std::string &channel) const {
    std::lock_guard<std::mutex> lock(this->mutex_);
    auto it = this->channels_.find(channel);

    if (it == this->channels_.end() ||
        it->second->type != std::type_index(typeid(T))) {
      return nullptr;
    }

    return std::static_pointer_cast<const T>(it->second->latest);
  }

  /// @brief Whether some plugin declared @p channel as produced.
  /// @param[in] channel Channel to query.
  /// @return True when a producer was declared or published.
  bool has_producer(const std::string &channel) const;

  /// @brief Block until a publish/wake/close happens or @p timeout elapses.
  /// @param[in] timeout Maximum time to block.
  /// @return True when activity was observed before the timeout.
  bool wait_for_activity(std::chrono::milliseconds timeout);

  /// @brief Whether close_all() has been called since the last reset().
  /// @return True while the blackboard is closed.
  bool closed() const;

  /// @brief Wake every blocked reader without closing the queues.
  void wake_all();

  /// @brief Close every consumer queue and wake all readers.
  void close_all();

  /// @brief Drop all channels and the publish hook (cleanup).
  ///
  /// Readers created before close_all() keep their own queue state (usually
  /// closed) because they hold the queue directly; new subscriptions create
  /// fresh channels and queues. Future publishes start from empty state, so
  /// reset() reverses close_all() for traffic created afterwards.
  void reset();

  /// @brief Install @p hook, invoked after every publish.
  /// @param[in] hook Callback receiving the channel, payload type and value.
  void set_publish_hook(PublishHook hook);

private:
  /// @brief Type-erased base of every typed channel.
  struct ChannelBase {
    /// @brief Create a channel carrying values of type @p t.
    /// @param[in] t Runtime type of the channel payload.
    explicit ChannelBase(std::type_index t) : type(t) {}

    /// @brief Destroy the channel.
    virtual ~ChannelBase() = default;

    /// @brief Close every consumer queue of this channel.
    virtual void close() = 0;

    /// @brief Wake every blocked reader of this channel.
    virtual void wake() = 0;

    /// @brief Runtime type of the channel payload.
    std::type_index type;
    /// @brief True once a producer declared or published this channel.
    bool has_producer = false;
    /// @brief Last published payload, type-erased.
    std::shared_ptr<const void> latest;
  };

  /// @brief Typed channel holding one queue per consumer.
  template <typename T> struct Channel : ChannelBase {
    /// @brief Create an empty channel for payload type @c T.
    Channel() : ChannelBase(std::type_index(typeid(T))) {}

    /// @brief Consumer queues fed on every publish.
    std::vector<std::shared_ptr<ConsumerQueue<T>>> consumers;

    /// @brief Close every consumer queue of this channel.
    void close() override {
      for (auto &consumer : this->consumers) {
        consumer->close();
      }
    }

    /// @brief Wake every blocked reader of this channel.
    void wake() override {
      for (auto &consumer : this->consumers) {
        consumer->wake();
      }
    }
  };

  /// @brief Find or create the channel, throwing on a type mismatch.
  /// @param[in] channel Channel name to look up.
  /// @return The existing or newly created typed channel.
  /// @throws std::runtime_error when the channel exists with another type.
  template <typename T>
  std::shared_ptr<Channel<T>> get_or_create(const std::string &channel) {
    auto it = this->channels_.find(channel);

    if (it == this->channels_.end()) {
      auto created = std::make_shared<Channel<T>>();
      this->channels_.emplace(channel, created);
      return created;
    }

    if (it->second->type != std::type_index(typeid(T))) {
      throw std::runtime_error("Blackboard channel '" + channel +
                               "' already has a different payload type");
    }

    return std::static_pointer_cast<Channel<T>>(it->second);
  }

  /// @brief Bump the activity generation and wake activity waiters.
  void notify_activity();

  /// @brief Guards @c channels_ and @c publish_hook_.
  ///
  /// Lock order: mutex_ may be held while taking a ConsumerQueue mutex
  /// (wake_all/close_all iterate consumers under mutex_). Queues never call
  /// back into the Blackboard, so the reverse order cannot occur.
  mutable std::mutex mutex_;
  /// @brief Channels keyed by channel name.
  std::unordered_map<std::string, std::shared_ptr<ChannelBase>> channels_;
  /// @brief Installed publish hook, may be empty.
  PublishHook publish_hook_;

  /// @brief Guards the activity state below.
  mutable std::mutex activity_mutex_;
  /// @brief Signalled on every publish, wake and close.
  std::condition_variable activity_cv_;
  /// @brief Bumped on every activity to detect new publishes.
  std::uint64_t activity_generation_ = 0;
  /// @brief True while close_all() is in effect.
  bool closed_ = false;
};

} // namespace yolo_ros

#endif // YOLO_ROS__BLACKBOARD__BLACKBOARD_HPP_
