// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief Approximate-time synchronization over blackboard channels.

#ifndef YOLO_ROS__BLACKBOARD__CHANNEL_SYNC_HPP_
#define YOLO_ROS__BLACKBOARD__CHANNEL_SYNC_HPP_

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "yolo_ros/blackboard/blackboard.hpp"
#include "yolo_ros/utils/message_filters_compat.hpp"

namespace yolo_ros {

/// @brief message_filters source fed from a blackboard consumer queue.
template <typename M>
class BlackboardFilter : public message_filters::SimpleFilter<M> {
public:
  /// @brief Emit @p msg to the registered synchronizer callback.
  /// @param[in] msg Message to signal.
  void feed(const std::shared_ptr<const M> &msg) { this->signalMessage(msg); }
};

/// @brief Synchronizes N blackboard channels with the ApproximateTime policy.
///
/// next() must be called from a single (the owning plugin's) thread; it drains
/// the channels, feeds the synchronizer and returns a matched tuple.
template <typename... Msgs> class ChannelSync {
  static_assert(sizeof...(Msgs) >= 2 && sizeof...(Msgs) <= 9,
                "ChannelSync requires between 2 and 9 message types "
                "(message_filters supports 2..9)");

public:
  /// @brief Matched tuple of the synchronized messages.
  using Result = std::tuple<std::shared_ptr<const Msgs>...>;
  /// @brief ApproximateTime synchronization policy.
  using Policy = message_filters::sync_policies::ApproximateTime<Msgs...>;
  /// @brief Synchronizer instantiated with @p Policy.
  using Synchronizer = message_filters::Synchronizer<Policy>;

  /// @brief Subscribe @p channels and build the synchronizer.
  /// @param[in] blackboard Bus providing the channel queues.
  /// @param[in] channels One channel name per message type.
  /// @param[in] queue Consumer queue depth for every channel.
  /// @throws std::invalid_argument when the channel count mismatches @p Msgs.
  ChannelSync(Blackboard &blackboard, const std::vector<std::string> &channels,
              std::size_t queue = 10)
      : blackboard_(blackboard) {
    if (channels.size() != sizeof...(Msgs)) {
      throw std::invalid_argument(
          "ChannelSync expects " + std::to_string(sizeof...(Msgs)) +
          " channels, got " + std::to_string(channels.size()));
    }

    this->setup(channels, queue, std::index_sequence_for<Msgs...>{});
  }

  /// @brief Return the next approximately synchronized set.
  ///
  /// Each drain feeds every queued message to the synchronizer, which may
  /// produce several matches per batch; only the most recent match is returned
  /// and earlier matches from the same batch are intentionally dropped
  /// (real-time latest-wins coalescing). The timeout is a lower bound: a match
  /// observed during the final wait is still returned after @p timeout has
  /// elapsed. A non-positive @p timeout only drains once and checks for a
  /// match, returning immediately (cheap polling).
  /// @param[out] out Receives the matched tuple on success.
  /// @param[in] timeout Maximum time to wait for a match.
  /// @return False once @p timeout elapsed with no match, or as soon as the
  /// blackboard is closed.
  bool next(Result &out, std::chrono::milliseconds timeout) {
    if (timeout <= std::chrono::milliseconds::zero()) {
      return this->consume(out);
    }

    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
      if (this->consume(out)) {
        return true;
      }

      this->blackboard_.wait_for_activity(std::chrono::milliseconds(20));

      if (this->blackboard_.closed()) {
        return this->consume(out);
      }
    } while (std::chrono::steady_clock::now() < deadline);

    return this->consume(out);
  }

private:
  /// @brief Drain every channel and report whether a match was produced.
  /// @param[out] out Receives the most recent match when one is available.
  /// @return True when a match was produced.
  bool consume(Result &out) {
    this->drain(std::index_sequence_for<Msgs...>{});

    if (!this->matched_) {
      return false;
    }

    out = *this->matched_;
    this->matched_.reset();
    return true;
  }

  /// @brief Subscribe one channel and store its reader.
  /// @tparam I Index of the message type/channel to initialize.
  /// @param[in] channel Channel name for slot @p I.
  /// @param[in] queue Consumer queue depth for slot @p I.
  template <std::size_t I>
  void init_one(const std::string &channel, std::size_t queue) {
    /// Message type of slot @p I.
    using M = typename std::tuple_element<I, std::tuple<Msgs...>>::type;
    std::get<I>(this->readers_) =
        this->blackboard_.subscribe<M>(channel, queue);
  }

  /// @brief Drain the queue of slot @p I into its synchronizer filter.
  /// @tparam I Index of the message type to drain.
  template <std::size_t I> void drain_one() {
    /// Message type of slot @p I.
    using M = typename std::tuple_element<I, std::tuple<Msgs...>>::type;
    std::shared_ptr<const M> msg;

    while (std::get<I>(this->readers_).try_pop(msg)) {
      std::get<I>(this->filters_).feed(msg);
    }
  }

  /// @brief Drain every queue in index order.
  /// @tparam I Message type indices.
  template <std::size_t... I> void drain(std::index_sequence<I...>) {
    (this->drain_one<I>(), ...);
  }

  /// @brief Subscribe every channel and register the match callback.
  /// @tparam I Message type indices.
  /// @param[in] channels Channel names, one per message type.
  /// @param[in] queue Consumer queue depth for every channel.
  template <std::size_t... I>
  void setup(const std::vector<std::string> &channels, std::size_t queue,
             std::index_sequence<I...>) {
    (this->init_one<I>(channels[I], queue), ...);
    this->synchronizer_ = std::make_shared<Synchronizer>(
        Policy(queue), std::get<I>(this->filters_)...);
    auto callback = [this](const std::shared_ptr<const Msgs> &...msgs) {
      this->matched_ = std::make_tuple(msgs...);
    };
#if defined(MESSAGE_FILTERS_LEGACY_API) || defined(MESSAGE_FILTERS_OLD_API)
    // Old-API Signal9 registers callbacks through a nine-slot std::bind padded
    // with NullType, so a plain N-argument lambda is not invocable. Binding the
    // lambda to the matching placeholders lets std::bind discard the padding;
    // this wrapper is only used on distros that take that padded path.
    const auto placeholders = std::make_tuple(
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
        std::placeholders::_4, std::placeholders::_5, std::placeholders::_6,
        std::placeholders::_7, std::placeholders::_8, std::placeholders::_9);
    this->synchronizer_->registerCallback(
        std::bind(callback, std::get<I>(placeholders)...));
#else
    // Newer APIs invoke the callback with exactly N arguments.
    this->synchronizer_->registerCallback(callback);
#endif
  }

  /// @brief Bus providing the synchronized channels.
  Blackboard &blackboard_;
  /// @brief One consumer reader per message type.
  std::tuple<ChannelReader<Msgs>...> readers_;
  /// @brief One message_filters source per message type.
  std::tuple<BlackboardFilter<Msgs>...> filters_;
  /// @brief ApproximateTime synchronizer producing the matches.
  std::shared_ptr<Synchronizer> synchronizer_;
  /// @brief Most recent match, empty when none is pending.
  std::optional<Result> matched_;
};

} // namespace yolo_ros

#endif // YOLO_ROS__BLACKBOARD__CHANNEL_SYNC_HPP_
