# Copyright (C) 2026 Alejandro González Cantón
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
  THIRD_PARTY_NOTICES.md
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
