// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief Deduplicating bridge between blackboard channels and ROS topics.

#ifndef YOLO_ROS__PLUGIN__TOPIC_REGISTRY_HPP_
#define YOLO_ROS__PLUGIN__TOPIC_REGISTRY_HPP_

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "rclcpp/qos.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "yolo_ros/blackboard/blackboard.hpp"

namespace yolo_ros {

/// @brief Output registry: deduplicates ROS publishers for plugin output
/// channels.
///
/// expose() -> blackboard channel -> ROS topic (via the publish hook)
///
/// @par Thread safety
/// expose() and validate() run on the node executor during configure.
/// create_entities() and destroy_entities() run during activate/deactivate,
/// when no plugin worker threads are running. The publish hook runs on plugin
/// worker threads and must not run concurrently with entity creation or
/// destruction.
class TopicRegistry {
public:
  /// @brief Construct an empty registry.
  TopicRegistry() = default;

  /// @brief Declare that @p channel must be mirrored on @p topic.
  /// @tparam T Payload type carried by the channel and topic.
  /// @param channel Blackboard channel mirrored to ROS.
  /// @param topic ROS topic name.
  /// @param qos Requested QoS profile.
  /// @param owner plugin instance name reported in conflict errors.
  template <typename T>
  void expose(const std::string &channel, const std::string &topic,
              const rclcpp::QoS &qos, const std::string &owner = "") {
    PublicationRequest request;
    request.channel = channel;
    request.topic = topic;
    request.type = std::type_index(typeid(T));
    request.type_name = typeid(T).name();
    request.qos = qos;
    request.owner = owner;
    request.create = [topic, qos](rclcpp_lifecycle::LifecycleNode &node) {
      auto publisher = node.create_publisher<T>(topic, qos);
      publisher->on_activate();
      return static_cast<rclcpp::PublisherBase::SharedPtr>(publisher);
    };
    request.publish = [](const rclcpp::PublisherBase::SharedPtr &base,
                         const std::shared_ptr<const void> &msg) {
      auto typed = std::static_pointer_cast<const T>(msg);
      std::static_pointer_cast<rclcpp::Publisher<T>>(base)->publish(*typed);
    };
    this->publications_.push_back(std::move(request));
  }

  /// @brief Check publication conflicts.
  /// @throws std::runtime_error on type/QoS conflicts.
  void validate();

  /// @brief Create the deduplicated ROS entities and install the hook.
  void create_entities(rclcpp_lifecycle::LifecycleNode &node,
                       Blackboard &blackboard);

  /// @brief Destroy entities and remove the hook (idempotent).
  void destroy_entities();

  /// @brief Drop all requests (cleanup).
  void reset();

  /// @brief Number of ROS publishers currently created.
  /// @return Created publisher entity count.
  std::size_t publisher_entity_count() const {
    return this->publisher_entities_.size();
  }

private:
  /// @brief Deleted copy constructor.
  TopicRegistry(const TopicRegistry &) = delete;
  /// @brief Deleted copy assignment.
  TopicRegistry &operator=(const TopicRegistry &) = delete;
  /// @brief Deleted move constructor.
  TopicRegistry(TopicRegistry &&) = delete;
  /// @brief Deleted move assignment.
  TopicRegistry &operator=(TopicRegistry &&) = delete;

  /// @brief One expose() declaration awaiting entity creation.
  struct PublicationRequest {
    /// @brief Blackboard channel mirrored to the topic.
    std::string channel;
    /// @brief ROS topic name.
    std::string topic;
    /// @brief Payload type used for conflict checks.
    std::type_index type{typeid(void)};
    /// @brief Human-readable payload type name for error messages.
    std::string type_name;
    /// @brief Requested QoS profile.
    rclcpp::QoS qos{1};
    /// @brief Plugin instance that requested the publication.
    std::string owner;
    /// @brief Factory creating the ROS publisher.
    std::function<rclcpp::PublisherBase::SharedPtr(
        rclcpp_lifecycle::LifecycleNode &)>
        create;
    /// @brief Type-erased publish callback for the created publisher.
    std::function<void(const rclcpp::PublisherBase::SharedPtr &,
                       const std::shared_ptr<const void> &)>
        publish;
  };

  /// @brief Runtime state of one exposed blackboard channel.
  struct ExposedChannel {
    /// @brief Payload type used to match the publish hook type.
    std::type_index type{typeid(void)};
    /// @brief Publisher shared with every channel on the same topic.
    rclcpp::PublisherBase::SharedPtr publisher;
    /// @brief Type-erased publish callback.
    std::function<void(const rclcpp::PublisherBase::SharedPtr &,
                       const std::shared_ptr<const void> &)>
        publish;
  };

  /// @brief Key and publisher for one deduplicated ROS publisher.
  struct PublisherGroup {
    /// @brief ROS topic name.
    std::string topic;
    /// @brief Payload type.
    std::type_index type{typeid(void)};
    /// @brief QoS profile.
    rclcpp::QoS qos{1};
    /// @brief Created publisher shared by matching requests.
    rclcpp::PublisherBase::SharedPtr publisher;
  };

  /// @brief All expose() declarations, in registration order.
  std::vector<PublicationRequest> publications_;
  /// @brief Created ROS publishers with their (topic, type, QoS) keys.
  std::vector<PublisherGroup> publisher_groups_;
  /// @brief Created ROS publishers (deduplicated).
  std::vector<rclcpp::PublisherBase::SharedPtr> publisher_entities_;
  /// @brief Exposed channels mapped to their publisher and publish hook.
  std::map<std::string, ExposedChannel> exposed_channels_;
  /// @brief Blackboard the publish hook is installed on, or nullptr.
  Blackboard *blackboard_ = nullptr;
};

} // namespace yolo_ros

#endif // YOLO_ROS__PLUGIN__TOPIC_REGISTRY_HPP_
