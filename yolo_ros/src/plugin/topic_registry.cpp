// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/plugin/topic_registry.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace yolo_ros {

namespace {

/// @brief Human-readable owner label used in conflict diagnostics.
/// @param owner Plugin instance name (may be empty).
/// @return The instance name, or "unknown" when it is empty.
std::string owner_name(const std::string &owner) {
  return owner.empty() ? std::string("unknown") : owner;
}

} // namespace

void TopicRegistry::validate() {
  // Reject conflicting requests before creating any entity.
  for (std::size_t i = 0; i < this->publications_.size(); ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      if (this->publications_[j].channel == this->publications_[i].channel &&
          this->publications_[j].topic != this->publications_[i].topic) {
        throw std::runtime_error(
            "Channel '" + this->publications_[i].channel +
            "' is exposed on multiple topics ('" +
            this->publications_[j].topic + "' by plugin '" +
            owner_name(this->publications_[j].owner) + "' vs '" +
            this->publications_[i].topic + "' by plugin '" +
            owner_name(this->publications_[i].owner) + "')");
      }

      if (this->publications_[j].topic == this->publications_[i].topic &&
          this->publications_[j].type != this->publications_[i].type) {
        throw std::runtime_error(
            "Topic '" + this->publications_[i].topic +
            "' is exposed by plugin '" +
            owner_name(this->publications_[i].owner) +
            "' with incompatible types (plugin '" +
            owner_name(this->publications_[j].owner) + "' uses " +
            this->publications_[j].type_name + ", plugin '" +
            owner_name(this->publications_[i].owner) + "' uses " +
            this->publications_[i].type_name + ")");
      }

      if (this->publications_[j].topic == this->publications_[i].topic &&
          this->publications_[j].qos != this->publications_[i].qos) {
        throw std::runtime_error("Topic '" + this->publications_[i].topic +
                                 "' is exposed by plugin '" +
                                 owner_name(this->publications_[i].owner) +
                                 "' with incompatible QoS (plugin '" +
                                 owner_name(this->publications_[j].owner) +
                                 "' uses a different QoS)");
      }
    }
  }
}

void TopicRegistry::create_entities(rclcpp_lifecycle::LifecycleNode &node,
                                    Blackboard &blackboard) {
  this->destroy_entities();
  this->blackboard_ = &blackboard;

  // One ROS publisher per (topic, type, QoS); map every exposed channel to it.
  for (const auto &request : this->publications_) {
    rclcpp::PublisherBase::SharedPtr publisher;

    for (auto &group : this->publisher_groups_) {
      if (group.topic == request.topic && group.type == request.type &&
          group.qos == request.qos) {
        publisher = group.publisher;
        break;
      }
    }

    if (!publisher) {
      publisher = request.create(node);

      if (!publisher) {
        throw std::runtime_error("Failed to create ROS publisher for topic '" +
                                 request.topic + "'");
      }

      this->publisher_groups_.push_back(
          {request.topic, request.type, request.qos, publisher});
      this->publisher_entities_.push_back(publisher);
    }

    ExposedChannel exposed;
    exposed.type = request.type;
    exposed.publisher = publisher;
    exposed.publish = request.publish;
    this->exposed_channels_[request.channel] = std::move(exposed);
  }

  blackboard.set_publish_hook([this](const std::string &channel,
                                     const std::type_info &type,
                                     const std::shared_ptr<const void> &msg) {
    auto it = this->exposed_channels_.find(channel);

    if (it == this->exposed_channels_.end() ||
        it->second.type != std::type_index(type) || !it->second.publisher) {
      return;
    }

    it->second.publish(it->second.publisher, msg);
  });
}

void TopicRegistry::destroy_entities() {
  if (this->blackboard_) {
    this->blackboard_->set_publish_hook(nullptr);
  }

  this->publisher_entities_.clear();
  this->publisher_groups_.clear();
  this->exposed_channels_.clear();
  this->blackboard_ = nullptr;
}

void TopicRegistry::reset() {
  this->destroy_entities();
  this->publications_.clear();
}

} // namespace yolo_ros
