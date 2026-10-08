# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.10)
project(yolo_hfhub_vendor)

include(FetchContent)
find_package(ament_cmake REQUIRED)
find_package(CURL REQUIRED)

# Fetch the huggingface-hub-cpp library (an unofficial C++ client for the
# Hugging Face Hub), used at runtime to download YOLO models from the Hub.
FetchContent_Declare(
  yolo_hfhub
  GIT_REPOSITORY https://github.com/agonzc34/huggingface-hub-cpp
  GIT_TAG        1.1.5
  GIT_SHALLOW    TRUE
)
# Built as a static archive, but downstream code (yolo_ros_plugins) links it
# into a shared library, which requires position-independent code.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
FetchContent_MakeAvailable(yolo_hfhub)

# Export the hfhub target and its include directories for downstream packages.
install(
  TARGETS hfhub
  EXPORT export_yolo_hfhub
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include
)

ament_export_include_directories(include)
ament_export_dependencies(CURL)
ament_export_targets(export_yolo_hfhub HAS_LIBRARY_TARGET)

install(FILES
  LICENSE
  THIRD_PARTY_NOTICES.md
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
