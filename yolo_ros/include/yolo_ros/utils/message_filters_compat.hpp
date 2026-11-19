// Copyright (C) 2026 Alejandro González Cantón
// Copyright (C) 2026 Miguel Ángel González Santamarta
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/// @file
/// @brief message_filters compatibility helpers across distros.
///
/// Foxy and Galactic define MESSAGE_FILTERS_LEGACY_API (set in CMakeLists.txt):
/// there message_filters::Subscriber takes only the message type and
/// subscribe() accepts an rclcpp::Node (or shared pointer) plus an
/// rmw_qos_profile_t. Humble..Kilted define MESSAGE_FILTERS_OLD_API: there
/// message_filters::Subscriber takes the node type as a second template
/// argument and subscribe() expects an rmw_qos_profile_t. Lyrical and newer
/// reduce Subscriber to the message type and make subscribe() take an
/// rclcpp::QoS. MessageFilterSubscriber normalises all of them so call sites
/// stay identical.

#ifndef YOLO_ROS__UTILS__MESSAGE_FILTERS_COMPAT_HPP_
#define YOLO_ROS__UTILS__MESSAGE_FILTERS_COMPAT_HPP_

#include "rclcpp/qos.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

#if defined(MESSAGE_FILTERS_LEGACY_API) || defined(MESSAGE_FILTERS_OLD_API)
#include "message_filters/simple_filter.h"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"
#else
#include "message_filters/simple_filter.hpp"
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
#if defined(MESSAGE_FILTERS_LEGACY_API)
template <typename MessageT>
class MessageFilterSubscriber : public message_filters::SimpleFilter<MessageT> {
public:
  /// @brief Subscribe to a topic.
  /// @param node Node used to create the subscription.
  /// @param topic Topic name to subscribe to.
  /// @param qos QoS profile for the subscription.
  template <typename NodeT>
  void subscribe(NodeT node, const std::string &topic, const rclcpp::QoS &qos) {
    this->subscription_ = node->template create_subscription<MessageT>(
        topic, qos, [this](const std::shared_ptr<const MessageT> msg) {
          this->signalMessage(msg);
        });
  }

  /// @brief Force immediate unsubscription from the topic.
  void unsubscribe() { this->subscription_.reset(); }

  /// @brief Returns the internal rclcpp::Subscription<MessageT>::SharedPtr.
  const typename rclcpp::Subscription<MessageT>::SharedPtr
  getSubscriber() const {
    return this->subscription_;
  }

private:
  typename rclcpp::Subscription<MessageT>::SharedPtr subscription_;
};
#elif defined(MESSAGE_FILTERS_OLD_API)
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
