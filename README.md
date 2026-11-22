# yolo_ros

ROS 2 wrap for YOLO models from [Ultralytics](https://github.com/ultralytics/ultralytics) to perform object detection and tracking, instance segmentation, human pose estimation, oriented bounding box (OBB) and image classification. It also provides 3D versions of object detection, instance segmentation and human pose estimation from depth images.

The pipeline is a pure **C++ / ONNX Runtime** implementation: there is no Python runtime and no `ultralytics` dependency at run time. It runs any Ultralytics-exported ONNX model (see the [model guide](docs/models.md)) and is licensed under **MIT**. Every stage runs inside a single `yolo_node` executable as a **pluginlib** plugin — detection, tracking, 3D lifting and debug rendering — exchanging data over an in-memory typed blackboard.

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit) [![GitHub release](https://img.shields.io/github/release/mgonzs13/yolo_ros.svg)](https://github.com/mgonzs13/yolo_ros/releases) [![Code Size](https://img.shields.io/github/languages/code-size/mgonzs13/yolo_ros.svg?branch=main)](https://github.com/mgonzs13/yolo_ros?branch=main) [![Last Commit](https://img.shields.io/github/last-commit/mgonzs13/yolo_ros.svg?branch=main)](https://github.com/mgonzs13/yolo_ros/commits/main) [![GitHub issues](https://img.shields.io/github/issues/mgonzs13/yolo_ros)](https://github.com/mgonzs13/yolo_ros/issues) [![GitHub pull requests](https://img.shields.io/github/issues-pr/mgonzs13/yolo_ros)](https://github.com/mgonzs13/yolo_ros/pulls) [![Contributors](https://img.shields.io/github/contributors/mgonzs13/yolo_ros.svg)](https://github.com/mgonzs13/yolo_ros/graphs/contributors) [![Doxygen Deployment](https://github.com/mgonzs13/yolo_ros/actions/workflows/doxygen-deployment.yml/badge.svg?branch=main)](https://github.com/mgonzs13/yolo_ros/actions/workflows/doxygen-deployment.yml?branch=main)

| ROS 2 Distro |                          Branch                          |                                                                                                                          Build status                                                                                                                          |                                                                Docker Image                                                                 |
| :----------: | :------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------: |
|   **Foxy**   | [`main`](https://github.com/mgonzs13/yolo_ros/tree/main) |         [![Foxy Build](https://img.shields.io/github/actions/workflow/status/mgonzs13/yolo_ros/foxy-build-test.yml?branch=main&event=push&label=Foxy)](https://github.com/mgonzs13/yolo_ros/actions/workflows/foxy-build-test.yml?query=branch%3Amain)         |     [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-foxy-blue)](https://hub.docker.com/r/mgons/yolo_ros/tags?name=foxy)     |
| **Galactic** | [`main`](https://github.com/mgonzs13/yolo_ros/tree/main) | [![Galactic Build](https://img.shields.io/github/actions/workflow/status/mgonzs13/yolo_ros/galactic-build-test.yml?branch=main&event=push&label=Galactic)](https://github.com/mgonzs13/yolo_ros/actions/workflows/galactic-build-test.yml?query=branch%3Amain) | [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-galactic-blue)](https://hub.docker.com/r/mgons/yolo_ros/tags?name=galactic) |
|  **Humble**  | [`main`](https://github.com/mgonzs13/yolo_ros/tree/main) |     [![Humble Build](https://img.shields.io/github/actions/workflow/status/mgonzs13/yolo_ros/humble-build-test.yml?branch=main&event=push&label=Humble)](https://github.com/mgonzs13/yolo_ros/actions/workflows/humble-build-test.yml?query=branch%3Amain)     |   [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-humble-blue)](https://hub.docker.com/r/mgons/yolo_ros/tags?name=humble)   |
|   **Iron**   | [`main`](https://github.com/mgonzs13/yolo_ros/tree/main) |         [![Iron Build](https://img.shields.io/github/actions/workflow/status/mgonzs13/yolo_ros/iron-build-test.yml?branch=main&event=push&label=Iron)](https://github.com/mgonzs13/yolo_ros/actions/workflows/iron-build-test.yml?query=branch%3Amain)         |     [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-iron-blue)](https://hub.docker.com/r/mgons/yolo_ros/tags?name=iron)     |
|  **Jazzy**   | [`main`](https://github.com/mgonzs13/yolo_ros/tree/main) |       [![Jazzy Build](https://img.shields.io/github/actions/workflow/status/mgonzs13/yolo_ros/jazzy-build-test.yml?branch=main&event=push&label=Jazzy)](https://github.com/mgonzs13/yolo_ros/actions/workflows/jazzy-build-test.yml?query=branch%3Amain)       |    [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-jazzy-blue)](https://hub.docker.com/r/mgons/yolo_ros/tags?name=jazzy)    |
|  **Kilted**  | [`main`](https://github.com/mgonzs13/yolo_ros/tree/main) |     [![Kilted Build](https://img.shields.io/github/actions/workflow/status/mgonzs13/yolo_ros/kilted-build-test.yml?branch=main&event=push&label=Kilted)](https://github.com/mgonzs13/yolo_ros/actions/workflows/kilted-build-test.yml?query=branch%3Amain)     |   [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-kilted-blue)](https://hub.docker.com/r/mgons/yolo_ros/tags?name=kilted)   |
| **Lyrical**  | [`main`](https://github.com/mgonzs13/yolo_ros/tree/main) |   [![Lyrical Build](https://img.shields.io/github/actions/workflow/status/mgonzs13/yolo_ros/lyrical-build-test.yml?branch=main&event=push&label=Lyrical)](https://github.com/mgonzs13/yolo_ros/actions/workflows/lyrical-build-test.yml?query=branch%3Amain)   |  [![Docker Image](https://img.shields.io/badge/Docker%20Image%20-lyrical-blue)](https://hub.docker.com/r/mgons/yolo_ros/tags?name=lyrical)  |

</div>

## Quick start

```shell
cd ~/ros2_ws/src
git clone https://github.com/mgonzs13/yolo_ros.git

cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash

ros2 launch yolo_bringup yolo.launch.py
```

The default build uses the ROS 2 `onnxruntime_vendor` CPU runtime. For CUDA, TensorRT or QNN, pass `-DONNXRUNTIME_ROOT=<prefix>` to CMake; see the [installation guide](docs/installation.md).

## Documentation

| Guide                                       | Contents                                                                               |
| ------------------------------------------- | -------------------------------------------------------------------------------------- |
| [Installation](docs/installation.md)        | Prerequisites, CPU build, custom ONNX Runtime, QNN/Ventuno setup                       |
| [Build options](docs/build.md)              | CUDA/TensorRT/QNN details, cuDNN/TensorRT versions, source-build knobs, offline builds |
| [Docker](docs/docker.md)                    | CPU and GPU images, NVIDIA runtime, TensorRT engine cache                              |
| [Models](docs/models.md)                    | Supported model families, ONNX export, dynamic batch, Hugging Face Hub, ReID encoders  |
| [Usage](docs/usage.md)                      | Pipeline YAML, architecture, 3D detection, multi-camera behavior, topics and services  |
| [Parameters](docs/parameters.md)            | Every node/plugin parameter, default, CLI alias and YAML-only key                      |
| [Multi-camera pipelines](docs/pipelines.md) | Full multi-camera example, chain semantics, per-camera outputs, batching scaling       |
| [Writing a plugin](docs/plugins.md)         | Plugin interface, registration and loading                                             |
| [Demos](docs/demos.md)                      | Detection, segmentation, pose, OBB, classification and 3D launch commands with videos  |
| [Benchmark](docs/benchmark.md)              | CPU vs CUDA vs TensorRT latency and resource usage                                     |

The C++ API reference is generated with Doxygen and published to GitHub Pages on every release: <https://mgonzs13.github.io/yolo_ros/>. To build it locally:

```shell
sudo apt install doxygen graphviz
doxygen .github/Doxyfile
# output: docs/doxygen/index.html
```

## License

The repository is licensed under the **MIT License**; see the root [`LICENSE`](LICENSE) file and the [license and third-party notices](docs/license.md). Model weights and exported ONNX files remain subject to their own licenses; Ultralytics models are **AGPL-3.0**, not MIT.
