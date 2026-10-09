cmake_minimum_required(VERSION 3.10)
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
find_package(pluginlib REQUIRED)

# Optional CUDA runtime: enables CUDA graph execution on the CUDA EP. The
# feature compiles out when the toolkit is unavailable.
set(YOLO_ROS_CUDA_RUNTIME_LIBS "")
set(YOLO_ROS_CUDA_RUNTIME_DEFS "")
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
  find_package(CUDAToolkit QUIET)
endif()
if(CUDAToolkit_FOUND)
  set(YOLO_ROS_CUDA_RUNTIME_LIBS CUDA::cudart)
  set(YOLO_ROS_CUDA_RUNTIME_DEFS YOLO_ROS_HAS_CUDA_RUNTIME)
endif()

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
  # Foxy's pluginlib/class_loader.hpp includes boost/shared_ptr.hpp.
  find_package(Boost REQUIRED)
endif()

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${yolo_onnxruntime_vendor_INCLUDE_DIRS}
)

# Shared library with the framework and the engine/task/tracking/3D sources.
add_library(yolo_ros_core SHARED
  src/blackboard/blackboard.cpp
  src/camera/camera_frame.cpp
  src/camera/camera_streams.cpp
  src/plugin/topic_registry.cpp
  src/plugin/plugin_host.cpp
  src/node/yolo_node.cpp

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
  src/engine/model.cpp
  src/engine/input_shape.cpp
  src/utils/logs.cpp
  src/utils/cpu_utils.cpp

  src/3d/depth_utils.cpp

  src/yolo/segment.cpp
  src/yolo/detect.cpp
  src/yolo/pose.cpp
  src/yolo/classify.cpp
  src/yolo/obb.cpp
  src/yolo/utils.cpp
  src/yolo/model_task.cpp
  src/yolo/model_factory.cpp
)

target_include_directories(yolo_ros_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(yolo_ros_core
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  pluginlib::pluginlib
  tf2_ros::tf2_ros
  message_filters::message_filters
  ${sensor_msgs_TARGETS}
  ${yolo_msgs_TARGETS}
  ${CV_BRIDGE_TARGET}
  ament_index_cpp::ament_index_cpp
  ${yolo_onnxruntime_vendor_LIBRARIES}
  ${yolo_hfhub_vendor_TARGETS}
  ${OpenCV_LIBS}
)

if("$ENV{ROS_DISTRO}" STREQUAL "foxy")
  # Boost headers pulled in by Foxy's pluginlib; propagate to the plugins and
  # test targets that link yolo_ros_core.
  target_link_libraries(yolo_ros_core Boost::boost)
endif()

if(YOLO_ROS_CUDA_RUNTIME_DEFS)
  target_compile_definitions(yolo_ros_core PRIVATE ${YOLO_ROS_CUDA_RUNTIME_DEFS})
  target_link_libraries(yolo_ros_core ${YOLO_ROS_CUDA_RUNTIME_LIBS})
endif()

# Plugin implementations loaded by the single YoloNode.
add_library(yolo_ros_plugins SHARED
  src/plugins/debug_plugin.cpp
  src/plugins/tracking_plugin.cpp
  src/plugins/detect_3d_plugin.cpp
  src/plugins/detection_plugin.cpp
)

target_include_directories(yolo_ros_plugins PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(yolo_ros_plugins
  yolo_ros_core
  ${sensor_msgs_TARGETS}
  ${visualization_msgs_TARGETS}
  message_filters::message_filters
  ${yolo_msgs_TARGETS}
  ${CV_BRIDGE_TARGET}
  ${yolo_onnxruntime_vendor_LIBRARIES}
  ${yolo_hfhub_vendor_TARGETS}
  ament_index_cpp::ament_index_cpp
  ${OpenCV_LIBS}
)

pluginlib_export_plugin_description_file(yolo_ros plugins.xml)

add_executable(yolo_node src/yolo_main.cpp)
target_link_libraries(yolo_node yolo_ros_core)

install(TARGETS
  yolo_ros_core
  yolo_ros_plugins
  yolo_node
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_export_include_directories(include)
ament_export_libraries(yolo_ros_core)
ament_export_dependencies(
  rclcpp
  rclcpp_lifecycle
  pluginlib
  sensor_msgs
  visualization_msgs
  message_filters
  tf2_ros
  geometry_msgs
  yolo_msgs
  ament_index_cpp
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

  ament_add_gtest(test_blackboard
    test/test_blackboard.cpp
  )
  target_link_libraries(test_blackboard
    yolo_ros_core
  )

  ament_add_gtest(test_channel_sync
    test/test_channel_sync.cpp
  )
  target_link_libraries(test_channel_sync
    yolo_ros_core
    message_filters::message_filters
    ${sensor_msgs_TARGETS}
    ${yolo_msgs_TARGETS}
  )

  ament_add_gtest(test_cpu_utils
    test/test_cpu_utils.cpp
  )
  target_link_libraries(test_cpu_utils
    yolo_ros_core
  )

  ament_add_gtest(test_camera_frame
    test/test_camera_frame.cpp
  )
  target_link_libraries(test_camera_frame
    yolo_ros_core
    ${sensor_msgs_TARGETS}
    ${yolo_msgs_TARGETS}
    message_filters::message_filters
  )

  ament_add_gtest(test_camera_streams
    test/test_camera_streams.cpp
  )
  target_link_libraries(test_camera_streams
    yolo_ros_core
    message_filters::message_filters
    ${sensor_msgs_TARGETS}
    ${yolo_msgs_TARGETS}
  )

  ament_add_gtest(test_topic_registry
    test/test_topic_registry.cpp
  )
  target_link_libraries(test_topic_registry
    yolo_ros_core
    ${sensor_msgs_TARGETS}
    ${yolo_msgs_TARGETS}
  )

  ament_add_gtest(test_plugin_host
    test/test_plugin_host.cpp
  )
  target_include_directories(test_plugin_host PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/test
  )
  target_link_libraries(test_plugin_host
    yolo_ros_core
  )

  ament_add_gtest(test_yolo_node
    test/test_yolo_node.cpp
  )
  target_include_directories(test_yolo_node PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/test
  )
  target_link_libraries(test_yolo_node
    yolo_ros_core
    tf2_ros::tf2_ros
  )

  ament_add_gtest(test_debug_plugin
    test/test_debug_plugin.cpp
  )
  target_link_libraries(test_debug_plugin
    yolo_ros_plugins
    ${sensor_msgs_TARGETS}
    ${yolo_msgs_TARGETS}
  )

  ament_add_gtest(test_tracking_plugin
    test/test_tracking_plugin.cpp
  )
  target_link_libraries(test_tracking_plugin
    yolo_ros_plugins
    ${sensor_msgs_TARGETS}
    ${yolo_msgs_TARGETS}
  )

  ament_add_gtest(test_detect_3d_plugin
    test/test_detect_3d_plugin.cpp
  )
  target_link_libraries(test_detect_3d_plugin
    yolo_ros_plugins
    tf2_ros::tf2_ros
    ${geometry_msgs_TARGETS}
    ${sensor_msgs_TARGETS}
    ${yolo_msgs_TARGETS}
  )

  ament_add_gtest(test_detection_plugin
    test/test_detection_plugin.cpp
  )
  target_link_libraries(test_detection_plugin
    yolo_ros_plugins
    ${yolo_msgs_TARGETS}
  )

  ament_add_gtest(test_input_shape
    test/test_input_shape.cpp
    src/engine/input_shape.cpp
  )
  target_include_directories(test_input_shape PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )

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
  src/utils/cpu_utils.cpp
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
  src/utils/cpu_utils.cpp
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

  ament_add_gtest(test_ort_compat
    test/test_ort_compat.cpp
  )
  target_include_directories(test_ort_compat PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test>
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(test_ort_compat
    ${yolo_msgs_TARGETS}
    ${yolo_onnxruntime_vendor_LIBRARIES}
    ${yolo_hfhub_vendor_TARGETS}
    ${OpenCV_LIBS}
    CURL::libcurl
    JPEG::JPEG
  )
  set_tests_properties(test_ort_compat PROPERTIES TIMEOUT 300)

  ament_add_gtest(test_yolo26n
    test/test_yolo26n.cpp
    src/yolo/detect.cpp
    src/yolo/segment.cpp
    src/yolo/pose.cpp
    src/yolo/classify.cpp
    src/yolo/utils.cpp
    src/engine/model.cpp
    src/engine/input_shape.cpp
    src/engine/provider.cpp
    src/utils/logs.cpp
  src/utils/cpu_utils.cpp
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

  if(YOLO_ROS_CUDA_RUNTIME_DEFS)
    target_compile_definitions(test_yolo26n PRIVATE
      ${YOLO_ROS_CUDA_RUNTIME_DEFS})
    target_link_libraries(test_yolo26n ${YOLO_ROS_CUDA_RUNTIME_LIBS})
  endif()

  ament_add_gtest(test_batch
    test/test_batch.cpp
    src/yolo/detect.cpp
    src/yolo/utils.cpp
    src/engine/model.cpp
    src/engine/input_shape.cpp
    src/engine/provider.cpp
    src/utils/logs.cpp
  src/utils/cpu_utils.cpp
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

  if(YOLO_ROS_CUDA_RUNTIME_DEFS)
    target_compile_definitions(test_batch PRIVATE
      ${YOLO_ROS_CUDA_RUNTIME_DEFS})
    target_link_libraries(test_batch ${YOLO_ROS_CUDA_RUNTIME_LIBS})
  endif()

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
