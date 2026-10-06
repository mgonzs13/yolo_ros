// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

/// @file
/// @brief message_filters compatibility helpers across distros.
///
/// Humble..Kilted define MESSAGE_FILTERS_OLD_API (set in CMakeLists.txt): there
/// message_filters::Subscriber takes the node type as a second template
/// argument and subscribe() expects an rmw_qos_profile_t. Lyrical and newer
/// reduce Subscriber to the message type and make subscribe() take an
/// rclcpp::QoS. MessageFilterSubscriber normalises both so call sites stay
/// identical.

#ifndef YOLO_ROS__UTILS__MESSAGE_FILTERS_COMPAT_HPP_
#define YOLO_ROS__UTILS__MESSAGE_FILTERS_COMPAT_HPP_

#include "rclcpp/qos.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

#ifdef MESSAGE_FILTERS_OLD_API
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"
#else
#include "message_filters/subscriber.hpp"
#include "message_filters/sync_policies/approximate_time.hpp"
#include "message_filters/synchronizer.hpp"
#endif

/// @addtogroup yolo_utils
/// @{
namespace yolo_ros::utils {

/// @brief message_filters::Subscriber specialised for a lifecycle node.
///
/// Exposes subscribe(node, topic, rclcpp::QoS) on every distro, so call sites
/// look the same regardless of the installed message_filters API:
///   yolo_ros::utils::MessageFilterSubscriber<MsgType> subscription_;
///   subscription_.subscribe(shared_from_this(), topic, qos);
#ifdef MESSAGE_FILTERS_OLD_API
template <typename MessageT>
class MessageFilterSubscriber
    : public message_filters::Subscriber<MessageT,
                                         rclcpp_lifecycle::LifecycleNode> {
public:
  /// @brief Subscribe to a topic.
  /// @param node Node used to create the subscription.
  /// @param topic Topic name to subscribe to.
  /// @param qos QoS profile for the subscription.
  template <typename NodeT>
  void subscribe(NodeT node, const std::string &topic, const rclcpp::QoS &qos) {
    message_filters::Subscriber<MessageT, rclcpp_lifecycle::LifecycleNode>::
        subscribe(node, topic, qos.get_rmw_qos_profile());
  }
};
#else
template <typename MessageT>
class MessageFilterSubscriber : public message_filters::Subscriber<MessageT> {
public:
  /// @brief Subscribe to a topic.
  /// @param node Node used to create the subscription.
  /// @param topic Topic name to subscribe to.
  /// @param qos QoS profile for the subscription.
  template <typename NodeT>
  void subscribe(NodeT node, const std::string &topic, const rclcpp::QoS &qos) {
    message_filters::Subscriber<MessageT>::subscribe(node, topic, qos);
  }
};
#endif

} // namespace yolo_ros::utils
/// @}

#endif // YOLO_ROS__UTILS__MESSAGE_FILTERS_COMPAT_HPP_
