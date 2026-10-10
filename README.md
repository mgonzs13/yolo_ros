# yolo_ros

ROS 2 wrap for YOLO models from [Ultralytics](https://github.com/ultralytics/ultralytics) to perform object detection and tracking, instance segmentation, human pose estimation, Oriented Bounding Box (OBB) and image classification. There are also 3D versions of object detection, instance segmentation and human pose estimation based on depth images.

The pipeline is a pure **C++ / ONNX Runtime** implementation: there is no Python runtime and no `ultralytics` dependency at run time. It runs any Ultralytics-exported ONNX model (see [Models](#models)), and the whole repository is licensed under **MIT**. Every stage runs inside a single `yolo_node` executable as a **pluginlib** plugin — detection, tracking, 3D lifting and debug rendering — exchanging data over an in-memory typed blackboard.

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

## Table of Contents

1. [Installation](#installation)
2. [Docker](#docker)
3. [Models](#models)
4. [Usage](#usage)
5. [Demos](#demos)
6. [Documentation](#documentation)
7. [License](#license)

## Installation

The node builds as a standard ROS 2 `colcon` workspace. ONNX Runtime 1.20.0 is downloaded automatically by `yolo_onnxruntime_vendor` at configure time — the CPU tarball by default, the GPU tarball with `-DONNX_GPU=ON` — so the runtime is not something you install by hand.

Prerequisites and checkout, common to every backend:

```shell
# Clone this repo
cd ~/ros2_ws/src
git clone https://github.com/mgonzs13/yolo_ros.git

# Install rosdep dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

### CPU (x64 and aarch64)

```shell
colcon build --symlink-install
source install/setup.bash
```

`yolo_onnxruntime_vendor` downloads the CPU ONNX Runtime for the host architecture (x86_64 or aarch64) and the node runs on the CPU execution provider — no CUDA or cuDNN required.

### CUDA / TensorRT (x64)

The vendor package detects the CUDA major and downloads the matching ONNX Runtime GPU build, which needs the cuDNN major shown:

| CUDA | ONNX Runtime | cuDNN |
| ---- | ------------ | ----- |
| 11   | 1.18.0       | 8     |
| 12   | 1.20.0       | 9     |
| 13   | 1.28.0       | 9     |

```shell
# Scripted: detects CUDA + Ubuntu, installs the matching cuDNN, runs ldconfig
# and verifies it (--tensorrt for the TensorRT EP, --ubuntu 2004|2204|2404 to
# override the detected release, --dry-run to preview).
sudo yolo_onnxruntime_vendor/scripts/install_gpu_deps.sh

colcon build --symlink-install --cmake-args -DONNX_GPU=ON
source install/setup.bash
```

Or manually (only add the keyring when apt does not know `libcudnn9-*`, e.g. systems set up from a local `cuda-repo-*` archive; use the URL matching your Ubuntu release — `ubuntu2004` / `ubuntu2204` / `ubuntu2404`):

```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# cuDNN matching the detected CUDA major. The -dev package is recommended:
# ONNX Runtime 1.18/1.20 link the versioned libcudnn soname directly, while
# 1.28 resolves it at run time (probing libcudnn.so.9, then libcudnn.so).
sudo apt-get install libcudnn9-dev-cuda-13   # CUDA 13; libcudnn9-dev-cuda-12 for CUDA 12; libcudnn8-dev for CUDA 11
sudo ldconfig
```

If startup reports `Using execution provider: cpu`, or the CUDA provider fails with `cuDNN is unavailable or disabled ... dlopen failed for libcudnn.so`, the runtime libraries are missing or the wrong major — install the cuDNN matching the table (cuDNN 8 for CUDA 11, cuDNN 9 for CUDA 12/13, with the CUDA-major-specific package).

For `provider: tensorrt`, the ONNX Runtime builds need the TensorRT **10** runtime (`libnvinfer.so.10`, `libnvonnxparser.so.10`): install `libnvinfer10` and `libnvonnxparsers10`, or run the script with `--tensorrt`. The `tensorrt` meta package may point at a newer major (TensorRT 11) and will not satisfy it; in that case the provider logs `Failed to load library .../libonnxruntime_providers_tensorrt.so ... libnvinfer.so.10: cannot open shared object file` and `provider: tensorrt` falls back to CUDA.

`-DONNX_GPU=ON` detects the CUDA major (`CUDA_VERSION` env, `$CUDA_HOME/version.json`, `/usr/local/cuda*/version.json`, then `nvcc`) and downloads the matching ONNX Runtime GPU build with its cuDNN major; the host also needs an NVIDIA driver and the same CUDA series at run time. Override the selection with `-DONNX_CUDA_MAJOR=11|12|13`, `-DONNXRUNTIME_VERSION=...`, `-DONNX_GPU_SUFFIX=...` or a full `-DONNXRUNTIME_URL=...`. TensorRT is opt-in: the default `provider: auto` runs on the CUDA execution provider and falls back to CPU; set `provider: tensorrt` to prefer the TensorRT EP (TensorRT → CUDA → CPU) when it is installed. `provider` and `device` are chosen at runtime — see [Parameters](#parameters). The prebuilt GPU tarballs are x64-only and cover only the table above; on aarch64 or for any other CUDA/cuDNN combination build from source (see the [build guide](docs/build.md)).

Launch from the workspace root so relative source paths resolve.

## Docker

Two Dockerfiles are provided: `Dockerfile` (CPU) and `Dockerfile.gpu` (CUDA / TensorRT, built on `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04` with `-DONNX_GPU=ON`).

```shell
docker build -t yolo_ros .                        # CPU image
docker build -f Dockerfile.gpu -t yolo_ros:gpu .  # CUDA / TensorRT image
```

If you want to use CUDA, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and add `--gpus all`:

```shell
docker run -it --rm --gpus all yolo_ros:gpu
```

See the [Docker guide](docs/docker.md) for the GPU details, the TensorRT engine cache and the NVIDIA runtime.

## Models

The C++ pipeline runs any Ultralytics-exported **ONNX** model whose output matches one of the YOLO layouts below. The compatible model families are:

- [YOLOv3](https://docs.ultralytics.com/models/yolov3/) (`yolov3u`, Ultralytics' updated anchor-free head)
- [YOLOv5](https://docs.ultralytics.com/models/yolov5/) (`yolov5u`)
- [YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- [YOLOv9](https://docs.ultralytics.com/models/yolov9/)
- [YOLOv10](https://docs.ultralytics.com/models/yolov10/)
- [YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [YOLO12](https://docs.ultralytics.com/models/yolo12/)
- [YOLO26](https://docs.ultralytics.com/models/yolo26/)

Models are exported for one task — **detection** (`detect`), **instance segmentation** (`segment`), **human pose** (`pose`), **oriented bounding box** (`obb`) or **image classification** (`classify`). See the [model export guide](docs/models.md) for how to export a `.pt` checkpoint to ONNX with the `ultralytics` package, the export notes and the Hugging Face Hub download.

> **License note**: Ultralytics models and pretrained weights are **not** MIT licensed. They are released under the **AGPL-3.0** license (with commercial / enterprise licensing available from Ultralytics), so the MIT license of this repository does **not** cover them. Check the terms of the specific model you use at <https://ultralytics.com/license> — especially if you ship or deploy the model.

## Usage

Run from the workspace root (so the workspace is sourced as an overlay). `yolo.launch.py` starts one `yolo_node` executable that loads every pipeline stage as a **pluginlib** plugin; a single YAML file defines the whole pipeline — the node-level `cameras`, the ordered `plugins` chain and every plugin's parameters. The task is selected by the **model**, not by a separate executable or launch.

The `plugins` parameter is an ordered list of plugin instance names; each instance selects its pluginlib class with a `<instance>.plugin` parameter, and the instance name is the prefix of every parameter the plugin reads (`detection.*` for `yolo_ros/DetectionPlugin`, `tracking.*` for `yolo_ros/TrackingPlugin`, `detection3d.*` for `yolo_ros/Detect3DPlugin`, `debug.*` for `yolo_ros/DebugPlugin`). The list order **is** the data-flow chain: the first plugin must be `yolo_ros/DetectionPlugin`, no plugin type may appear twice and `yolo_ros/DebugPlugin` must be last. Each plugin also takes a `cameras` list selecting which cameras it processes and consumes the previous plugin's output for each camera; every camera a plugin selects must also be selected by the previous plugin in the chain (the first plugin consumes the camera frames directly), otherwise configure fails with `plugin '<instance>': camera '<cam>' is not produced by plugin '<previous instance>'`. Omit a plugin's `cameras` key to process every defined camera, and never write `cameras: []` (see the YAML note below). The plugins exchange data in memory over typed **blackboard channels** and each runs on its own worker thread. `yolo.launch.py` is a thin loader: it passes its YAML params file plus scalar command-line overrides and makes no topic remaps. The default `config/yolo.yaml` looks like:

```yaml
/yolo/yolo_node:
  ros__parameters:
    plugins: ["detection", "tracking", "detection3d", "debug"]
    cameras: ["cam0"]
    cam0:
      rgb_topic: /camera/rgb/image_raw
      depth_topic: /camera/depth/image_raw # optional pair, set together
      depth_info_topic: /camera/depth/camera_info
      image_reliability: 1
      depth_reliability: 1
    detection:
      plugin: yolo_ros/DetectionPlugin
      cameras: ["cam0"]
      model_repo: unileon-robotics/YOLO26-ONNX
      model_filename: yolo26s.onnx
      threshold: 0.7
      iou: 0.45
      n_threads: -1
      provider: cuda
    tracking:
      plugin: yolo_ros/TrackingPlugin
      cameras: ["cam0"]
      tracker_type: bytetrack
    detection3d:
      plugin: yolo_ros/Detect3DPlugin
      cameras: ["cam0"]
      target_frame: base_link
    debug:
      plugin: yolo_ros/DebugPlugin
      cameras: ["cam0"]
      marker_lifetime: 0.5
```

Each camera declares its `rgb_topic` and optionally the `depth_topic` + `depth_info_topic` pair (set together); cameras are synchronized by the node and stay internal. For an RGB-only setup remove the depth fields and the `detection3d` instance — a camera without depth cannot be selected by `detection3d`. Several cameras are declared in the same `cameras` list, each with its own block (see [Multi-camera pipelines](#multi-camera-pipelines)).

> **YAML note (Jazzy)**: never write an empty sequence in the params file — `cameras: []`, `plugins: []`. rcl's YAML parser produces no value for `[]` on Jazzy, so the node fails to start. Omit the key instead: an omitted plugin `cameras` key means "all defined cameras" (or list every camera explicitly).

Pick the model with `model_filename:=` (a file in the Hugging Face mirror set by `model_repo`) or `model_path:=` (a local path). The task is inferred from the file name (`model_type: auto`): `-seg`/`segment` → segmentation, `-pose`/`pose` → pose, `-obb`/`obb` → OBB, `-cls`/`classify` → classification, anything else detection. Each task also ships a preset params file (`config/yolo_segment.yaml`, `yolo_pose.yaml`, `yolo_obb.yaml`, `yolo_classify.yaml`) that pins `model_type` and task-specific tuning — pass it with `params_file:=`:

```shell
ros2 launch yolo_bringup yolo.launch.py params_file:=$(ros2 pkg prefix yolo_bringup)/share/yolo_bringup/config/yolo_segment.yaml
```

Runnable launch commands for each task (detection, segmentation, pose, OBB and classification) are collected in [Demos](#demos).

### Architecture

`yolo_node` is a single process. One lifecycle node owns `CameraStreams`, which creates one synchronized subscription set per declared camera (RGB alone, or RGB + depth + `CameraInfo` through ApproximateTime) and publishes each camera's frames as a zero-copy `CameraFrame` on the internal channel named after the camera. The plugins listed in `plugins` then run in order, each consuming the previous plugin's output for every camera it selects, and `TopicRegistry` exposes the plugin output channels as ROS publishers only (it creates no subscriptions). Camera frames and inter-plugin channels stay internal; each plugin runs on its own worker thread.

```text
+----------------------------------------------------------------------+
| yolo_node (single process, single lifecycle node)                    |
|                                                                      |
| cameras: ["cam0", ...]    plugins: ["detection", "tracking", ...]    |
|                                                                      |
| CameraStreams: one synchronized rgb[/depth/depth_info] subscription  |
| set per camera -> shared_ptr<const CameraFrame> on channel <cam>     |
|                                                                      |
| ordered chain, per camera:                                           |
|   CameraFrame -> DetectionPlugin -> TrackingPlugin -> Detect3DPlugin |
|               -> DebugPlugin                                         |
|   channels: <cam>/detections -> <cam>/tracking -> <cam>/detections_3d|
|             <cam>/debug_image, <cam>/debug_bb_markers,               |
|             <cam>/debug_kp_markers                                   |
|                                                                      |
| TopicRegistry: one ROS publisher per exposed output channel/topic;   |
| camera frames are never subscribed from ROS                          |
|                                                                      |
| each plugin runs on its own worker thread                            |
+----------------------------------------------------------------------+
```

### 3D Detection

A camera gets a depth stream by setting both `depth_topic` and `depth_info_topic` in its block; adding `detection3d` (class `yolo_ros/Detect3DPlugin`) to the `plugins` chain then synchronizes that camera's depth image + `CameraInfo` with the upstream 2D stream and publishes `<cam>/detections_3d`. Cameras selected by `detection3d` must have depth and must also be selected by the plugin whose detections it lifts (tracking, or detection when tracking is omitted), and `detection3d` must come after that plugin. The shipped `config/yolo.yaml` already includes `detection3d` over a depth camera; for RGB-only setups remove the depth fields and the `detection3d` entry.

For segmentation the depth ROI is driven by the mask polygon; for pose the 2D keypoints are back-projected to 3D (`<cam>/debug_kp_markers`). The 3D plugin can also estimate the orientation of each box (an oriented bounding box fit by PCA to a strided depth sample) when `enable_orientation` is set. When enabled, `detections_3d` carries a non-identity quaternion in each box's `center.orientation` and the box `size` is expressed along the object's own axes.

### Multi-camera pipelines

To run **one shared detector batched across several cameras**, declare every camera on the node and list the ones each plugin should process. A plugin's `cameras` list selects a subset of the cameras its predecessor also selects — omit the list to select every defined camera (never write `cameras: []`) — so tracking, 3D and debug can run on different subsets:

```yaml
cameras: ["cam0", "cam1"]
cam0:
  rgb_topic: /cam0/color/image_raw
  depth_topic: /cam0/depth/image_raw
  depth_info_topic: /cam0/depth/camera_info
cam1:
  rgb_topic: /cam1/color/image_raw
detection:
  plugin: yolo_ros/DetectionPlugin
  cameras: ["cam0", "cam1"]
tracking:
  plugin: yolo_ros/TrackingPlugin
  cameras: ["cam0", "cam1"]
detection3d:
  plugin: yolo_ros/Detect3DPlugin
  cameras: ["cam0"] # every camera here must have depth
debug:
  plugin: yolo_ros/DebugPlugin
  cameras: ["cam0"] # subset of detection3d's cameras
```

`DetectionPlugin` runs directly on the single-camera case and batches (up to `max_batch_size`) when it selects several cameras; the shared model must be a dynamic-batch ONNX export (the mirror's `dynamic/` files are). Outputs stay camera-prefixed (`cam0/detections`, `cam1/detections`, ...). See the [multi-camera pipelines guide](docs/pipelines.md) for the full example and the batching-scaling numbers.

### Topics

All output topics are published under the launch namespace (default `yolo`) and are **camera-prefixed**: a camera named `cam0` publishes under `/<namespace>/cam0/`. Whether a topic exists depends on the `plugins` chain and on the camera being in each plugin's `cameras` list:

- **<cam>/detections**: Objects detected by YOLO using the camera's RGB images. Each object contains a bounding box and a class name, plus a mask or a list of keypoints for segmentation/pose models.
- **<cam>/tracking**: Objects detected and tracked by ByteTrack (or BoT-SORT with `tracking.tracker_type: botsort`). Each object is assigned a stable tracking ID.
- **<cam>/detections_3d**: 3D objects detected by `detection3d`. YOLO results are used to crop the depth image and create 3D bounding boxes and keypoints.
- **<cam>/debug_image**: Debug image showing the detected and tracked objects. It can be visualized with `rqt_image_view` or `rviz2`.
- **<cam>/debug_bb_markers** / **<cam>/debug_kp_markers**: RViz `MarkerArray`s built from the debug plugin's input detection stream; they carry content when `detection3d` precedes debug in the chain (`bbox3d`/`keypoints3d` data).

Synchronized camera frames and the inter-plugin blackboard channels are in-process only — the topics above are the only ROS outputs.

### Services

There are no runtime services. Runtime enable/disable and class filtering were removed entirely: the old `/yolo/enable` and `/yolo/set_classes` services and their `detection.enable` / `detection.classes` startup replacements are gone, so inference always runs and every class is published.

### Parameters

Configuration is file-driven: `yolo.launch.py` declares `params_file` (default `config/yolo.yaml`) and `namespace` (default `yolo`) and passes the YAML params file plus any command-line overrides as `parameters=[params_file, overrides]` (with no topic remaps). The default is `config/yolo.yaml`; the per-task presets (`config/yolo_segment.yaml`, `yolo_pose.yaml`, `yolo_obb.yaml`, `yolo_classify.yaml`) are selected with `params_file:=`. The YAML holds the node-level `cameras` list with the per-camera fields (`<cam>.rgb_topic`, ...), the ordered `plugins` chain, each plugin's `cameras` list and the plugin parameters grouped by instance (`detection.*`, `tracking.*`, `detection3d.*`, `debug.*`); each instance also sets its `<instance>.plugin` class. In addition, **every scalar parameter can be overridden from the command line**; an argument left unset keeps the YAML value. Array parameters (the node `cameras` list and every plugin `cameras` list) and the per-camera blocks are **YAML-only** and cannot be passed as command-line overrides:

```bash
ros2 launch yolo_bringup yolo.launch.py model_path:=/path/model.onnx threshold:=0.5
ros2 launch yolo_bringup yolo.launch.py model_filename:=yolo26l-seg.onnx threshold:=0.6
```

Never write an empty sequence in the YAML (`cameras: []`, `plugins: []`): rcl's YAML parser produces no value for `[]` on Jazzy and the node fails to start. Omit the key instead; for a plugin's `cameras`, omitting it selects every defined camera (or list every camera explicitly).

Override arguments use their parameter name (`threshold` → `detection.threshold`, `tracker_type` → `tracking.tracker_type`, ...). Because an empty value means "not provided", a non-empty YAML string cannot be overridden to empty from the CLI. Run `ros2 launch yolo_bringup yolo.launch.py --show-args` for the full list.

Sections are keyed by the node's fully qualified name, so the key must include the launch namespace (default `yolo`):

```yaml
/yolo/yolo_node:
  ros__parameters:
    plugins: ["detection"]
    cameras: ["cam0"]
    cam0:
      rgb_topic: /camera/rgb/image_raw
    detection:
      plugin: yolo_ros/DetectionPlugin
      cameras: ["cam0"]
      model_type: auto
```

If you change `namespace:=`, update the matching config block names for any value you do not override on the command line. The key parameters are listed below; see `yolo_bringup/config/yolo*.yaml` for the complete set.

#### Plugin instances (`plugins`)

- **plugins**: Ordered list of plugin instance names loaded by the node; the order defines the data-flow chain. Each instance selects its pluginlib class with a `<instance>.plugin` parameter (the conventional instances map `detection` → `yolo_ros/DetectionPlugin`, `tracking` → `yolo_ros/TrackingPlugin`, `detection3d` → `yolo_ros/Detect3DPlugin` and `debug` → `yolo_ros/DebugPlugin`) and reads its parameters under `<instance>.*`. The first entry must be `yolo_ros/DetectionPlugin`, no plugin type may appear twice and `yolo_ros/DebugPlugin` must be last; any subset starting with detection is valid (`["detection"]`, `["detection", "debug"]`, `["detection", "detection3d", "debug"]`, ...). Camera selection follows the same chain: each plugin's `cameras` list must be a subset of the previous plugin's selection (the first plugin consumes the camera frames directly). A camera selected only downstream is a configure error — e.g. `plugin 'tracking': camera 'cam1' is not produced by plugin 'detection'` — and a camera listed twice in one instance is rejected as a duplicate.

#### Cameras (`cameras` and `<cam>.*`)

The node-level `cameras` parameter lists the camera names. Each name declares its own fields:

- **<cam>.rgb_topic** (required): RGB `sensor_msgs/Image` topic.
- **<cam>.depth_topic** / **<cam>.depth_info_topic**: optional depth image and its `sensor_msgs/CameraInfo`; they must be set together (setting only one is a configure error).
- **<cam>.image_reliability** / **<cam>.depth_reliability**: QoS reliability as an integer: `0`=system default, `1`=Reliable, `2`=Best Effort (defaults: `2`).

Camera names must be non-empty, unique and free of `.`, `:`, `/`, and no topic may be used by two cameras. Camera frames are synchronized and stay in-process, so these topics are the node's only ROS subscriptions.

#### Detection plugin (`detection.*`, `yolo_ros/DetectionPlugin`)

- **model_type**: Pipeline to run: `YOLO`/`Detect`, `Segment`, `Pose`, `OBB`, `Classify` or `auto` (default: `auto`, which infers the task from the model file name: `-seg`/`segment`, `-pose`/`pose`, `-obb`/`obb`, `-cls`/`classify`).
- **model_path**: Path to the ONNX model (default: empty; the shipped configs set `model_repo`/`model_filename` instead).
- **model_repo** / **model_filename** / **force_download** / **cache_dir**: Hugging Face Hub download (used instead of `model_path` when set; defaults: empty / empty / `false` / `~/.cache/huggingface/hub`). The shipped configs default to the `unileon-robotics/YOLO26-ONNX` mirror; clear `model_repo` to fall back to the local `model_path`.
- **provider**: Execution provider: `auto` (CUDA → CPU fallback chain), or force `tensorrt`/`trt` (TensorRT → CUDA → CPU), `cuda` (CUDA → CPU), `cpu` (default: `auto`).
- **device**: CUDA/TensorRT device ordinal, e.g. `cuda:0`, `trt:1`, `1` (default: `cuda:0`). The `cuda:`/`trt:` prefix is accepted but `provider` selects the execution provider.
- **cuda_graph_enable**: Capture the fixed-shape model as a CUDA graph on the CUDA provider, cutting per-kernel launch overhead (default: `true`). Only applies to fixed-batch models; dynamic-batch exports, CPU and TensorRT sessions keep the plain path.
- **trt_fp16_enable**: TensorRT FP16 precision (default: `true`).
- **trt_engine_cache_enable**: Persist built TensorRT engines (default: `true`).
- **trt_engine_cache_path**: TensorRT engine cache base directory; empty → `~/.cache/yolo_ros/trt_engines/<model>` (default: empty).
- **threshold**: Detection confidence threshold (default: `0.7`).
- **iou**: IoU threshold for the C++ NMS applied to raw-output exports (detection without a baked-NMS head, segmentation, pose and OBB); it has no effect on exports that bake NMS into the graph (default: `0.45`).
- **max_det**: Maximum number of detections per image (default: `300`).
- **cameras**: Cameras to run inference on; omit the list to select every defined camera (never write `cameras: []`). Detection is first in the chain, so it can select any defined camera. One camera uses the direct path, several run the dynamic-batch path.
- **max_batch_size**: Maximum batch size for the multi-camera path (default: `8`).
- **n_threads**: CPU execution-provider threads; `-1` (or any value <= 0) = auto: performance cores on hybrid CPUs, otherwise physical cores (default: `-1`).
- **max_fps**: Cap the inference/publish rate in Hz; `0` = unlimited (default: `0`).
- **img_width** / **img_height**: Network input size for models exported with a dynamic input (`dynamic=True`) (defaults: `640`/`480`). Models with a static input keep the size baked into the ONNX graph (a warning is logged when the parameters differ); non-positive values fail configure and values not divisible by 32 log a warning.
- **top_k**: Classification only — number of top classes to publish (default: `5`).

#### Tracking plugin (`tracking.*`, `yolo_ros/TrackingPlugin`)

- **cameras**: Cameras to track; omit the list to select every defined camera (never write `cameras: []`). Every selected camera must also be selected by the previous plugin in the chain.
- **tracker_type**: C++ implementation: `bytetrack` (default) or `botsort` (XYWH Kalman filter + camera-motion compensation). The tracker knobs below live in this same instance block; an unknown `tracker_type` logs a warning and passes detections through.
- **track_high_thresh**: First-stage association threshold (default: `0.25`).
- **track_low_thresh**: Second-stage threshold for low-score matches (default: `0.1`).
- **new_track_thresh**: Score above which a new track is started (default: `0.25`).
- **track_buffer**: Frames a lost track is kept alive (default: `30`).
- **match_thresh**: Association similarity threshold (IoU/cost) (default: `0.8`).
- **fuse_score**: Fuse detection score with IoU cost for matching (default: `true`).
- **gmc_method** (BoT-SORT only): Camera-motion method: `none` (default), `sparseOptFlow`, `orb` or `ecc`.
- **gmc_downscale** (BoT-SORT only): Camera-motion downscale factor (default: `2`).
- **with_reid** / **reid_model** / **proximity_thresh** / **appearance_thresh** (BoT-SORT only): ReID appearance model and its matching thresholds (defaults: `false` / empty / `0.5` / `0.25`). Set `with_reid: true` and point `reid_model` at an ONNX encoder (export recipes in the [model export guide](docs/models.md#reid-encoder-for-bot-sort-reid)).
- **provider** / **device** (BoT-SORT + ReID only): ReID model execution provider and device (defaults: `auto` / `cuda:0`).

#### 3D detection plugin (`detection3d.*`, `yolo_ros/Detect3DPlugin`)

- **cameras**: Cameras to lift to 3D; omit the list to select every defined camera (never write `cameras: []`), every selected camera must have the `depth_topic` + `depth_info_topic` pair, and it must also be selected by the previous plugin in the chain.
- **target_frame**: Frame to transform the 3D boxes into (default: `base_link`).
- **depth_image_units_divisor**: Divisor to convert the depth image to meters (default: `1000`).
- **enable_orientation**: Estimate and publish the 3D box orientation (default: `false`).
- **min_seg_points_for_orientation**: Minimum valid depth points required per detection for orientation estimation (default: `20`).

#### Debug plugin (`debug.*`, `yolo_ros/DebugPlugin`)

- **cameras**: Cameras to render; omit the list to select every defined camera (never write `cameras: []`) and every selected camera must also be selected by the previous plugin. Debug is terminal, so the plugin must be last in the chain.
- **marker_lifetime**: RViz marker lifetime in seconds (default: `0.5`).

The plugin syncs each camera frame with its chain input (the previous plugin's detections) and publishes `<cam>/debug_image`; the RViz markers `<cam>/debug_bb_markers` and `<cam>/debug_kp_markers` are built whenever those detections carry `bbox3d`/`keypoints3d` data (i.e. when `detection3d` precedes debug in the chain).

### Writing a plugin

Pipeline stages are **pluginlib** plugins. To add your own:

1. Derive from `yolo_ros::Plugin` and implement `declare_params`, `get_params`, `output_channel`, `setup`, `activate`, `deactivate` and `run` (the worker loop, which must poll its `stop` argument and return promptly). The two parameter methods receive the owning lifecycle node and the instance `prefix`, a runtime string ending in a dot (e.g. `"detection."`), so a plugin declares and reads every parameter under its `<instance>.*` namespace without hard-coding the instance name:

   ```cpp
   void declare_params(rclcpp_lifecycle::LifecycleNode &node,
                       const std::string &prefix) override {
     node.declare_parameter<double>(prefix + "threshold", 0.7);
   }

   void get_params(const rclcpp_lifecycle::LifecycleNode &node,
                   const std::string &prefix) override {
     node.get_parameter(prefix + "threshold", this->threshold_);
   }
   ```

2. Implement `output_channel(camera)` returning the per-camera channel/topic your plugin produces (the chain wires it as the next plugin's input). In `setup`, read the cameras to process from `ctx.cameras` — each `CameraInput` carries the camera name, the internal frame channel and the upstream `input_channel` — and declare/expose outputs with `ctx.blackboard.declare_channel<T>(channel)` and `ctx.topics.expose<T>(channel, topic, qos, ctx.name)`. Plugin inputs are always in-process channels wired by the node; `TopicRegistry` only publishes.
3. Register the class with `PLUGINLIB_EXPORT_CLASS(my_ns::MyPlugin, yolo_ros::Plugin)` and add a matching `<class>` entry to [`yolo_ros/plugins.xml`](yolo_ros/plugins.xml).
4. Load it by adding `"<instance>"` to the node's `plugins` parameter (respecting the chain rules: detection first, no duplicate types, debug last) and setting `<instance>.plugin: yolo_ros/MyPlugin` plus an optional `<instance>.cameras` list; its parameters are then declared and read under `<instance>.*`.

See `yolo_ros/src/plugins/debug_plugin.cpp` for a complete example.

## Demos

### Object Detection

Standard behavior including ByteTrack (or BoT-SORT) object tracking.

```shell
ros2 launch yolo_bringup yolo.launch.py
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1gTQt6soSIq1g2QmK7locHDiZ-8MqVl2w)](https://drive.google.com/file/d/1gTQt6soSIq1g2QmK7locHDiZ-8MqVl2w/view?usp=sharing)

### Instance Segmentation

Instance masks are the borders of the detected objects, not all the pixels inside the masks.

```shell
ros2 launch yolo_bringup yolo.launch.py model_filename:=yolo26l-seg.onnx
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1dwArjDLSNkuOGIB0nSzZR6ABIOCJhAFq)](https://drive.google.com/file/d/1dwArjDLSNkuOGIB0nSzZR6ABIOCJhAFq/view?usp=sharing)

### Human Pose

Visible persons are detected along with their skeleton keypoints.

```shell
ros2 launch yolo_bringup yolo.launch.py model_filename:=yolo26m-pose.onnx
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1pRy9lLSXiFEVFpcbesMCzmTMEoUXGWgr)](https://drive.google.com/file/d/1pRy9lLSXiFEVFpcbesMCzmTMEoUXGWgr/view?usp=sharing)

### Oriented Bounding Box

Rotated boxes are estimated for oriented objects.

```shell
ros2 launch yolo_bringup yolo.launch.py model_filename:=yolo26m-obb.onnx
```

<!-- Demo recording pending: add ./docs/media/demo_obb.gif to display it here.
<p align="center">
  <img src="./docs/media/demo_obb.gif" alt="Oriented bounding box demo" width="100%" />
</p>
-->

### Image Classification

Image-level ImageNet-1k labels are published as detections with an empty bbox, so the `yolo_classify.yaml` preset chains only detection + debug (no tracker to run).

```shell
ros2 launch yolo_bringup yolo.launch.py params_file:=$(ros2 pkg prefix yolo_bringup)/share/yolo_bringup/config/yolo_classify.yaml
```

<!-- Demo recording pending: add ./docs/media/demo_classify.gif to display it here.
<p align="center">
  <img src="./docs/media/demo_classify.gif" alt="Image classification demo" width="100%" />
</p>
-->

### 3D Object Detection

The 3D bounding boxes are calculated by filtering the depth image data from an RGB-D camera using the 2D bounding box. The default `config/yolo.yaml` chains `detection3d` over a camera with depth, so `<cam>/detections_3d` is published as soon as the camera streams depth.

```shell
ros2 launch yolo_bringup yolo.launch.py
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1ZcN_u9RB9_JKq37mdtpzXx3b44tlU-pr)](https://drive.google.com/file/d/1ZcN_u9RB9_JKq37mdtpzXx3b44tlU-pr/view?usp=sharing)

### 3D Object Detection (Using Instance Segmentation Masks)

The depth image data is filtered using the instance mask polygon. Only objects with a 3D bounding box are visualized in the 2D image (the preset keeps the default `detection3d` chain over a depth camera).

```shell
ros2 launch yolo_bringup yolo.launch.py model_filename:=yolo26l-seg.onnx
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1wVZgi5GLkAYxv3GmTxX5z-vB8RQdwqLP)](https://drive.google.com/file/d/1wVZgi5GLkAYxv3GmTxX5z-vB8RQdwqLP/view?usp=sharing)

### 3D Human Pose

Each keypoint is back-projected to 3D using the depth image. Only objects with a 3D bounding box are visualized in the 2D image (the preset keeps the default `detection3d` chain over a depth camera).

```shell
ros2 launch yolo_bringup yolo.launch.py model_filename:=yolo26m-pose.onnx
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1j4VjCAsOCx_mtM2KFPOLkpJogM0t227r)](https://drive.google.com/file/d/1j4VjCAsOCx_mtM2KFPOLkpJogM0t227r/view?usp=sharing)

## Documentation

The C++ API reference is generated with Doxygen and published to GitHub Pages on every release:

- Latest: <https://mgonzs13.github.io/yolo_ros/>

Build it locally (Doxygen + Graphviz):

```shell
sudo apt install doxygen graphviz
doxygen .github/Doxyfile
# output: docs/doxygen/index.html
```

The guides in [`docs/`](./docs) — [build](docs/build.md), [Docker](docs/docker.md), [model export](docs/models.md), [multi-camera pipelines](docs/pipelines.md) and [benchmark](docs/benchmark.md) — are plain Markdown served directly by GitHub.

## License

The whole repository is licensed under the **MIT License**. See the root [`LICENSE`](./LICENSE) file; it applies across the repository where a package does not provide a more specific license file.

Specifically, the repository contains independently licensed ROS 2 packages:

- `yolo_ros`, `yolo_msgs`, `yolo_bringup`, `yolo_onnxruntime_vendor` and `yolo_hfhub_vendor` are licensed under **MIT**. See each package's `LICENSE` file; third-party notices are installed with the applicable packages (`yolo_ros/THIRD_PARTY_NOTICES.md`).

The C++ pipeline adapts behavior from the original `yolo_ros` Python nodes; those contributions were authorized by their copyright holder for release in the MIT-licensed C++ pipeline (see `THIRD_PARTY_NOTICES.md`).

Model weights and exported ONNX files are separate artifacts and remain subject to their respective licenses; the MIT license for the C++ pipeline does not relicense them. In particular, Ultralytics models are AGPL-3.0 — see the [Models](#models) section.
