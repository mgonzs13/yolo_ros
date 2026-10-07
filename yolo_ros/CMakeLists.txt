cmake_minimum_required(VERSION 3.8)
project(yolo_ros)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(yolo_msgs REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yolo_onnxruntime_vendor REQUIRED)
find_package(yolo_hfhub_vendor REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)

# Some dependencies renamed headers or changed their C++ API across distros.
# ROS_DISTRO is exported by the sourced ROS environment; pick the variant with a
# compile definition, mirroring the CV_BRIDGE_H handling in llama_ros.
if("$ENV{ROS_DISTRO}" STREQUAL "foxy" OR "$ENV{ROS_DISTRO}" STREQUAL "galactic" OR "$ENV{ROS_DISTRO}" STREQUAL "humble")
  # cv_bridge < 4.x (Foxy..Humble) ships cv_bridge/cv_bridge.h; 4.x (Iron and
  # newer) renamed it to cv_bridge/cv_bridge.hpp.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCV_BRIDGE_H")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCV_BRIDGE_HPP")
endif()
if("$ENV{ROS_DISTRO}" STREQUAL "foxy" OR "$ENV{ROS_DISTRO}" STREQUAL "galactic" OR "$ENV{ROS_DISTRO}" STREQUAL "humble" OR "$ENV{ROS_DISTRO}" STREQUAL "iron" OR "$ENV{ROS_DISTRO}" STREQUAL "jazzy" OR "$ENV{ROS_DISTRO}" STREQUAL "kilted")
  # Foxy..Kilted ship the .h headers and subscribe() expects an
  # rmw_qos_profile_t.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMESSAGE_FILTERS_OLD_API")
endif()

if("$ENV{ROS_DISTRO}" STREQUAL "foxy" OR "$ENV{ROS_DISTRO}" STREQUAL "galactic")
  # Foxy and Galactic predate the NodeType template argument in
  # message_filters::Subscriber and only accept rclcpp::Node instances.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMESSAGE_FILTERS_LEGACY_API")
  # cv_bridge only exports the cv_bridge::cv_bridge imported target since
  # Humble; Foxy and Galactic export the legacy ${cv_bridge_LIBRARIES} variable.
  set(CV_BRIDGE_TARGET ${cv_bridge_LIBRARIES})
  include_directories(${cv_bridge_INCLUDE_DIRS})
else()
  set(CV_BRIDGE_TARGET cv_bridge::cv_bridge)
endif()
if("$ENV{ROS_DISTRO}" STREQUAL "foxy")
  # Foxy's rclcpp::QoS::reliability() only accepts
  # rmw_qos_reliability_policy_t; rclcpp::ReliabilityPolicy landed in Galactic.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DRCLCPP_LEGACY_QOS")
endif()

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${yolo_onnxruntime_vendor_INCLUDE_DIRS}
)

# Install yolo_ros
add_executable(yolo_node
  src/node/yolo_node.cpp
  src/yolo_main.cpp

  src/yolo/segment.cpp
  src/yolo/detect.cpp
  src/yolo/pose.cpp
  src/yolo/classify.cpp
  src/yolo/obb.cpp
  src/yolo/utils.cpp
  src/yolo/model_task.cpp
  src/yolo/model_factory.cpp

  src/engine/model.cpp
  src/engine/provider.cpp
  src/utils/logs.cpp
)

target_include_directories(yolo_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(yolo_node
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  ${sensor_msgs_TARGETS}
  ${std_srvs_TARGETS}
  ${CV_BRIDGE_TARGET}
  ament_index_cpp::ament_index_cpp
  ${yolo_msgs_TARGETS}
  ${yolo_onnxruntime_vendor_LIBRARIES}
  ${yolo_hfhub_vendor_TARGETS}
  ${OpenCV_LIBS}
)

# Install debug node
add_executable(debug_node
  src/debug_main.cpp
  src/node/debug_node.cpp
)

target_include_directories(debug_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(debug_node
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  ${sensor_msgs_TARGETS}
  ${CV_BRIDGE_TARGET}
  ${visualization_msgs_TARGETS}
  message_filters::message_filters
  ${yolo_msgs_TARGETS}
  ${OpenCV_LIBS}
)

# Install tracking node
add_executable(tracking_node
  src/node/tracking_node.cpp
  src/tracking_main.cpp

  src/tracking/tracker.cpp
  src/tracking/byte_tracker.cpp
  src/tracking/bot_sort.cpp
  src/tracking/strack.cpp

  src/tracking/utils/kalman_filter.cpp
  src/tracking/utils/camera_motion.cpp
  src/tracking/utils/lapjv.cpp
  src/tracking/utils/matching.cpp

  src/engine/reid_encoder.cpp
  src/engine/provider.cpp
  src/utils/logs.cpp
)

target_include_directories(tracking_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(tracking_node
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  ${sensor_msgs_TARGETS}
  ${CV_BRIDGE_TARGET}
  message_filters::message_filters
  ${yolo_msgs_TARGETS}
  ${yolo_onnxruntime_vendor_LIBRARIES}
  ${OpenCV_LIBS}
)

# Install 3D detection node
add_executable(detect_3d_node
  src/node/detect_3d_node.cpp
  src/detect_3d_main.cpp
  src/3d/depth_utils.cpp
)

target_include_directories(detect_3d_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(detect_3d_node
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  ${sensor_msgs_TARGETS}
  ${CV_BRIDGE_TARGET}
  message_filters::message_filters
  tf2_ros::tf2_ros
  ${geometry_msgs_TARGETS}
  ${yolo_msgs_TARGETS}
  ${OpenCV_LIBS}
)

# Install multi-camera batched inference node
add_executable(yolo_batch_node
  src/node/batch_node.cpp
  src/batch_main.cpp

  src/yolo/model_factory.cpp
  src/yolo/model_task.cpp
  src/yolo/segment.cpp
  src/yolo/detect.cpp
  src/yolo/pose.cpp
  src/yolo/classify.cpp
  src/yolo/obb.cpp
  src/yolo/utils.cpp

  src/engine/model.cpp
  src/engine/provider.cpp
  src/utils/logs.cpp
)

target_include_directories(yolo_batch_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(yolo_batch_node
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  ${sensor_msgs_TARGETS}
  ${std_srvs_TARGETS}
  ${CV_BRIDGE_TARGET}
  ament_index_cpp::ament_index_cpp
  ${yolo_msgs_TARGETS}
  ${yolo_onnxruntime_vendor_LIBRARIES}
  ${yolo_hfhub_vendor_TARGETS}
  ${OpenCV_LIBS}
)

install(TARGETS
  yolo_node debug_node tracking_node detect_3d_node yolo_batch_node
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  LICENSE
  THIRD_PARTY_NOTICES.md
  DESTINATION share/${PROJECT_NAME}
)

# Install the class-name vocabulary used as a fallback when the ONNX graph
# carries no "names" metadata (resolved at runtime via the ament index).
install(DIRECTORY conf
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_utils
    test/test_utils.cpp
    src/yolo/utils.cpp
  )
  target_include_directories(test_utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test>
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(test_utils
    ${yolo_msgs_TARGETS}
    ${yolo_onnxruntime_vendor_LIBRARIES}
    ${OpenCV_LIBS}
  )

  ament_add_gtest(test_provider
    test/test_provider.cpp
    src/engine/provider.cpp
    src/utils/logs.cpp
  )
  target_include_directories(test_provider PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test>
  )
  target_link_libraries(test_provider
    ${yolo_onnxruntime_vendor_LIBRARIES}
  )

  ament_add_gtest(test_tracking
    test/test_tracking.cpp
    src/tracking/tracker.cpp
    src/tracking/byte_tracker.cpp
    src/tracking/bot_sort.cpp
    src/tracking/strack.cpp
    src/tracking/utils/kalman_filter.cpp
    src/tracking/utils/camera_motion.cpp
    src/tracking/utils/lapjv.cpp
    src/tracking/utils/matching.cpp

    src/engine/reid_encoder.cpp
    src/engine/provider.cpp
    src/utils/logs.cpp
  )
  target_include_directories(test_tracking PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test>
  )
  target_link_libraries(test_tracking
    ${yolo_onnxruntime_vendor_LIBRARIES}
    ${OpenCV_LIBS}
  )

  ament_add_gtest(test_depth_utils
    test/test_depth_utils.cpp
    src/3d/depth_utils.cpp
  )
  target_include_directories(test_depth_utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test>
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(test_depth_utils
    ${sensor_msgs_TARGETS}
    ${yolo_msgs_TARGETS}
    ${OpenCV_LIBS}
  )

  find_package(CURL REQUIRED)
  find_package(JPEG REQUIRED)

  ament_add_gtest(test_yolo26n
    test/test_yolo26n.cpp
    src/yolo/detect.cpp
    src/yolo/segment.cpp
    src/yolo/pose.cpp
    src/yolo/classify.cpp
    src/yolo/utils.cpp
    src/engine/model.cpp
    src/engine/provider.cpp
    src/utils/logs.cpp
  )
  set_tests_properties(test_yolo26n PROPERTIES TIMEOUT 300)
  target_include_directories(test_yolo26n PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test>
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(test_yolo26n
    rclcpp::rclcpp
    ${sensor_msgs_TARGETS}
    ${CV_BRIDGE_TARGET}
    ament_index_cpp::ament_index_cpp
    ${yolo_msgs_TARGETS}
    ${yolo_onnxruntime_vendor_LIBRARIES}
    ${yolo_hfhub_vendor_TARGETS}
    ${OpenCV_LIBS}
    CURL::libcurl
    JPEG::JPEG
  )

  ament_add_gtest(test_batch
    test/test_batch.cpp
    src/yolo/detect.cpp
    src/yolo/utils.cpp
    src/engine/model.cpp
    src/engine/provider.cpp
    src/utils/logs.cpp
  )
  target_include_directories(test_batch PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test>
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(test_batch
    rclcpp::rclcpp
    ${sensor_msgs_TARGETS}
    ${CV_BRIDGE_TARGET}
    ament_index_cpp::ament_index_cpp
    ${yolo_msgs_TARGETS}
    ${yolo_onnxruntime_vendor_LIBRARIES}
    ${yolo_hfhub_vendor_TARGETS}
    ${OpenCV_LIBS}
    CURL::libcurl
    JPEG::JPEG
  )

  ament_add_gtest(test_batch_scheduler
    test/test_batch_scheduler.cpp
  )
  target_include_directories(test_batch_scheduler PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )

  ament_add_gtest(test_model_factory
    test/test_model_factory.cpp
    src/yolo/model_task.cpp
  )
  target_include_directories(test_model_factory PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(test_model_factory
    ${yolo_msgs_TARGETS}
    ${yolo_onnxruntime_vendor_LIBRARIES}
    ${CV_BRIDGE_TARGET}
  )
endif()

ament_package()
