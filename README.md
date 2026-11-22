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

The node builds as a standard ROS 2 `colcon` workspace. `yolo_ros` uses the released ROS 2 `onnxruntime_vendor` package by default, so a normal build gets the CPU ONNX Runtime from the ROS distribution. For CUDA, TensorRT or QNN, pass `-DONNXRUNTIME_ROOT=<prefix>` as a CMake argument; see [Custom ONNX Runtime](#custom-onnx-runtime-cuda-tensorrt-qnn).

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

`yolo_ros` links the ROS 2 `onnxruntime_vendor` CPU runtime and the node runs on the CPU execution provider — no CUDA or cuDNN required.

### Custom ONNX Runtime (CUDA, TensorRT, QNN)

By default `yolo_ros` uses the CPU ONNX Runtime from the released ROS 2
`onnxruntime_vendor` package. For CUDA/TensorRT/QNN, build or obtain an ONNX
Runtime prefix and pass it to CMake with `-DONNXRUNTIME_ROOT`:

```shell
# CUDA + TensorRT on x86_64/Jetson
src/yolo_ros/scripts/build_ort_from_source.sh 1.28.0 --ep cuda --cuda-arch 87

# QNN on the Ventuno Q
src/yolo_ros/scripts/build_ort_from_source.sh 1.28.0 --ep qnn --qnn-home /usr

# Build yolo_ros against the resulting prefix (default: src/yolo_ros/ort-1.28.0)
colcon build --symlink-install --packages-up-to yolo_ros \
    --cmake-args -DONNXRUNTIME_ROOT=$PWD/src/yolo_ros/ort-1.28.0
```

The prefix must contain `include/onnxruntime_cxx_api.h` and either
`lib/libonnxruntime.so` or `lib64/libonnxruntime.so`. Select the provider with
`provider: cuda`, `provider: tensorrt` or `provider: qnn`.

The CUDA/cuDNN/TensorRT versions must match the ONNX Runtime build. See the
[build guide](docs/build.md) for the version tables, source-build knobs and
troubleshooting.

### QNN (Hexagon NPU, VENTUNO Q)

The QNN execution provider is **not in the default ONNX Runtime**, so it always needs a custom source build. On the VENTUNO Q: install the board's QAIRT packages, build ONNX Runtime with `--ep qnn` and pass the prefix to CMake — details, version notes and troubleshooting are in the [QNN build guide](docs/build.md#qnn-hexagon-npu-onnx-runtime-build):

```shell
sudo apt install qairt-libs qairt-dsp-binaries qairt-headers

src/yolo_ros/scripts/build_ort_from_source.sh 1.28.0 --ep qnn --qnn-home /usr

colcon build --symlink-install --packages-up-to yolo_ros \
    --cmake-args -DONNXRUNTIME_ROOT=$PWD/src/yolo_ros/ort-1.28.0

ros2 launch yolo_bringup yolo.launch.py \
    params_file:=$(ros2 pkg prefix yolo_bringup)/share/yolo_bringup/config/yolo_qnn.yaml
```

Then export a QNN model with Ultralytics (`model.export(format="qnn", name="iq-8275")`) and launch the Hexagon NPU preset (`config/yolo_qnn.yaml`) — see the [QNN model guide](docs/models.md#qualcomm-qnn-hexagon-npu).

Launch from the workspace root so relative source paths resolve.

## Docker

Two Dockerfiles are provided: `Dockerfile` (CPU, using the ROS 2 `onnxruntime_vendor`) and `Dockerfile.gpu` (CUDA / TensorRT, built on `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04` with a custom GPU ONNX Runtime supplied through `-DONNXRUNTIME_ROOT`).

```shell
docker build -t yolo_ros .                        # CPU image
docker build -f Dockerfile.gpu -t yolo_ros:gpu .  # CUDA / TensorRT image
```

For the GPU image, put a flat CUDA/TensorRT ONNX Runtime prefix in `./onnxruntime` first (for example `cp -a ~/onnxruntime-1.20.0-gpu ./onnxruntime`); see the [Docker guide](docs/docker.md).

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

The `plugins` parameter is an ordered list of plugin instance names; each instance selects its pluginlib class with a `<instance>.plugin` parameter, and the instance name is the prefix of every parameter the plugin reads (`detection.*` for `yolo_ros/DetectionPlugin`, `tracking.*` for `yolo_ros/TrackingPlugin`, `detection3d.*` for `yolo_ros/Detect3DPlugin`, `debug.*` for `yolo_ros/DebugPlugin`). Parameters are grouped by concern below that prefix — for example `detection.model.path`, `detection.inference.threshold`, `tracking.track.high_thresh` and `detection3d.orientation.enable`. The list order **is** the data-flow chain: the first plugin must be `yolo_ros/DetectionPlugin`, no plugin type may appear twice and `yolo_ros/DebugPlugin` must be last. Each plugin also takes a `cameras` list selecting which cameras it processes and consumes the previous plugin's output for each camera; every camera a plugin selects must also be selected by the previous plugin in the chain (the first plugin consumes the camera frames directly), otherwise configure fails with <code>plugin '&lt;instance&gt;': camera '&lt;cam&gt;' is not produced by plugin '&lt;previous instance&gt;'</code>. Omit a plugin's `cameras` key to process every defined camera, and never write `cameras: []` (see the YAML note below). The plugins exchange data in memory over typed **blackboard channels** and each runs on its own worker thread. `yolo.launch.py` is a thin loader: it passes its YAML params file plus scalar command-line overrides and makes no topic remaps. The default `config/yolo.yaml` looks like:

```yaml
/yolo/yolo_node:
  ros__parameters:
    plugins: ["detection", "tracking", "detection3d", "debug"]
    cameras: ["cam0"]
    cam0:
      rgb:
        topic: /camera/rgb/image_raw
        reliability: 1
      depth:
        topic: /camera/depth/image_raw # optional pair, set together
        info_topic: /camera/depth/camera_info
        reliability: 1
    detection:
      plugin: yolo_ros/DetectionPlugin
      cameras: ["cam0"]
      model:
        repo: unileon-robotics/YOLO26-ONNX
        filename: yolo26s.onnx
      inference:
        threshold: 0.7
        iou: 0.45
      cpu:
        n_threads: -1
      input:
        width: 640
        height: 480
      provider: cuda
    tracking:
      plugin: yolo_ros/TrackingPlugin
      cameras: ["cam0"]
      tracker_type: bytetrack
      track:
        high_thresh: 0.25
        low_thresh: 0.1
        new_thresh: 0.25
        buffer: 30
        match_thresh: 0.8
        fuse_score: true
    detection3d:
      plugin: yolo_ros/Detect3DPlugin
      cameras: ["cam0"]
      target_frame: base_link
    debug:
      plugin: yolo_ros/DebugPlugin
      cameras: ["cam0"]
      marker_lifetime: 0.5
      compressed:
        enable: true
        quality: 80
```

Each camera declares its `<cam>.rgb.topic` and optionally the `<cam>.depth.topic` + `<cam>.depth.info_topic` pair (set together); cameras are synchronized by the node and stay internal. For an RGB-only setup remove the depth fields and the `detection3d` instance — a camera without depth cannot be selected by `detection3d`. Several cameras are declared in the same `cameras` list, each with its own block (see [Multi-camera pipelines](#multi-camera-pipelines)).

> **YAML note (Jazzy)**: never write an empty sequence in the params file — `cameras: []`, `plugins: []`. rcl's YAML parser produces no value for `[]` on Jazzy, so the node fails to start. Omit the key instead: an omitted plugin `cameras` key means "all defined cameras" (or list every camera explicitly).

Pick the model with `model_filename:=` (a file in the Hugging Face mirror set by `detection.model.repo`) or `model_path:=` (a local path). These launch arguments keep their historical flat names and map to `detection.model.filename` and `detection.model.path`. `detection.model.repo` and `detection.model.filename` must both be non-empty for the Hub download; if the download fails, the node logs the error and falls back to `detection.model.path`. The task is inferred from the file name when `detection.model.type` is `auto` (the default): a path containing `segment`/`-seg`/`_seg` → segmentation, `pose` → pose, `obb` → OBB, `cls`/`classify` → classification, anything else detection. `detection.model.type` is case-insensitive and also accepts explicit aliases: `yolo`/`detect`/`det`/`detection` for detection, any value containing `segment` for segmentation (use `segment`, not `seg`, which is only a filename heuristic), any value containing `pose`/`keypoint` or exactly `kpt` for pose, any value containing `obb`/`rotat` for OBB, and any value containing `class` or exactly `cls`/`clf` for classification.

Each task ships a preset params file that pins `detection.model.type` and task-specific tuning — `config/yolo_segment.yaml`, `yolo_pose.yaml`, `yolo_obb.yaml`, `yolo_classify.yaml` — plus the VENTUNO Q Hexagon NPU preset (`config/yolo_qnn.yaml`, see [QNN](#qnn-hexagon-npu-ventuno-q)). Pass one with `params_file:=`:

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
|             <cam>/debug_image, <cam>/debug_image/compressed,         |
|             <cam>/debug_bb_markers, <cam>/debug_kp_markers           |
|                                                                      |
| TopicRegistry: one ROS publisher per exposed output channel/topic;   |
| camera frames are never subscribed from ROS                          |
|                                                                      |
| each plugin runs on its own worker thread                            |
+----------------------------------------------------------------------+
```

### 3D Detection

A camera gets a depth stream by setting both `depth.topic` and `depth.info_topic` in its block; `CameraStreams` then synchronizes its RGB, depth and `CameraInfo` into one `CameraFrame`. Adding `detection3d` (class `yolo_ros/Detect3DPlugin`) to the `plugins` chain synchronizes that frame with the upstream 2D detections and publishes `<cam>/detections_3d`. Cameras selected by `detection3d` must have depth and must also be selected by the plugin whose detections it lifts (tracking, or detection when tracking is omitted), and `detection3d` must come after that plugin. The shipped `config/yolo.yaml` already includes `detection3d` over a depth camera; for RGB-only setups remove the depth fields and the `detection3d` entry.

For segmentation the depth ROI is driven by the mask polygon; for pose the 2D keypoints are back-projected to 3D (`<cam>/debug_kp_markers`). The 3D plugin can also estimate the orientation of each box (an oriented bounding box fit by PCA to a strided depth sample) when `orientation.enable` is set. When enabled, `detections_3d` carries a non-identity quaternion in each box's `center.orientation` and the box `size` is expressed along the object's own axes.

### Multi-camera pipelines

To run **one shared detector batched across several cameras**, declare every camera on the node and list the ones each plugin should process. A plugin's `cameras` list selects a subset of the cameras its predecessor also selects — omit the list to select every defined camera (never write `cameras: []`) — so tracking, 3D and debug can run on different subsets:

```yaml
cameras: ["cam0", "cam1"]
cam0:
  rgb:
    topic: /cam0/color/image_raw
  depth:
    topic: /cam0/depth/image_raw
    info_topic: /cam0/depth/camera_info
cam1:
  rgb:
    topic: /cam1/color/image_raw
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

`DetectionPlugin` runs directly on the single-camera case and batches (up to `batch.max_size`) when it selects several cameras. A dynamic-batch ONNX export gives true batching (the mirror's `dynamic/` files are); a fixed-batch model still works by chunking to the batch axis baked into the graph, so a stock batch-1 export simply processes one image per inference. Outputs stay camera-prefixed (`cam0/detections`, `cam1/detections`, ...). See the [multi-camera pipelines guide](docs/pipelines.md) for the full example and the batching-scaling numbers.

With TensorRT, ONNX Runtime 1.6 only reuses a cached engine when the input batch size is inside the optimization profile saved with it. The scheduler produces variable batch sizes (the first frame often arrives alone), so the engine can be rebuilt for several minutes on the first inference. Set `batch.pad_to` (CLI `pad_to_batch:=`) to a stable size (usually the camera count, e.g. `4`) to keep the profile valid and the cached engine reusable.

### Topics

All output topics are published under the launch namespace (default `yolo`) and are **camera-prefixed**: a camera named `cam0` publishes under `/<namespace>/cam0/`. Whether a topic exists depends on the `plugins` chain and on the camera being in each plugin's `cameras` list:

- **&lt;cam&gt;/detections**: Objects detected by YOLO using the camera's RGB images. Each object contains a bounding box and a class name, plus a mask or a list of keypoints for segmentation/pose models.
- **&lt;cam&gt;/tracking**: Objects detected and tracked by ByteTrack (or BoT-SORT with `tracking.tracker_type: botsort`). Each object is assigned a stable tracking ID.
- **&lt;cam&gt;/detections_3d**: 3D objects detected by `detection3d`. YOLO results are used to crop the depth image and create 3D bounding boxes and keypoints.
- **&lt;cam&gt;/debug_image**: Debug image showing the detected and tracked objects. It can be visualized with `rqt_image_view` or `rviz2`.
- **&lt;cam&gt;/debug_image/compressed**: JPEG `sensor_msgs/CompressedImage` version of the debug image (enabled by default). It is much lighter over WiFi and lets `ros2 topic hz` measure the real frame rate, which the raw image topic is too large for on CPU-constrained boards.
- **&lt;cam&gt;/debug_bb_markers** / **&lt;cam&gt;/debug_kp_markers**: RViz `MarkerArray`s built from the debug plugin's input detection stream; they carry content when `detection3d` precedes debug in the chain (`bbox3d`/`keypoints3d` data).

Synchronized camera frames and the inter-plugin blackboard channels are in-process only — the topics above are the only ROS outputs.

### Services

There are no runtime services. Runtime enable/disable and class filtering were removed entirely: the old `/yolo/enable` and `/yolo/set_classes` services and their `detection.enable` / `detection.classes` startup replacements are gone, so inference always runs and every class is published.

### Parameters

Configuration is file-driven: `yolo.launch.py` declares `params_file` (default `config/yolo.yaml`) and `namespace` (default `yolo`) and passes the YAML params file plus any command-line overrides as `parameters=[params_file, overrides]` (with no topic remaps). The default is `config/yolo.yaml`; the per-task presets (`config/yolo_segment.yaml`, `yolo_pose.yaml`, `yolo_obb.yaml`, `yolo_classify.yaml`) and the VENTUNO Q Hexagon NPU preset (`config/yolo_qnn.yaml`) are selected with `params_file:=`. The YAML holds the node-level `cameras` list with the per-camera fields (`<cam>.rgb.topic`, ...), the ordered `plugins` chain, each plugin's `cameras` list and the plugin parameters grouped by instance and concern (`detection.model.*`, `detection.inference.*`, `tracking.track.*`, `detection3d.orientation.*`, ...); each instance also sets its `<instance>.plugin` class. In addition, **every scalar parameter can be overridden from the command line**; an argument left unset keeps the YAML value. Array parameters (the node `cameras` list and every plugin `cameras` list) and the per-camera blocks are **YAML-only** and cannot be passed as command-line overrides:

```bash
ros2 launch yolo_bringup yolo.launch.py model_path:=/path/model.onnx threshold:=0.5
ros2 launch yolo_bringup yolo.launch.py model_filename:=yolo26l-seg.onnx threshold:=0.6
```

Never write an empty sequence in the YAML (`cameras: []`, `plugins: []`): rcl's YAML parser produces no value for `[]` on Jazzy and the node fails to start. Omit the key instead; for a plugin's `cameras`, omitting it selects every defined camera (or list every camera explicitly).

The launch-argument names keep their historical flat aliases (for example `threshold` → `detection.inference.threshold`, `model_path` → `detection.model.path`, `tracker_type` → `tracking.tracker_type`, `with_reid` → `tracking.reid.enable`), so existing launch commands still work. The ROS parameter names themselves are namespaced: the old flat parameters (`detection.model_path`, `detection.threshold`, `tracking.track_high_thresh`, ...) are no longer declared. Because an empty value means "not provided", a non-empty YAML string cannot be overridden to empty from the CLI. The shared `provider:=` and `device:=` aliases target both `detection.provider`/`detection.device` and, with BoT-SORT, `tracking.reid.provider`/`tracking.reid.device`; use the YAML if the detector and ReID encoder need different values. Run `ros2 launch yolo_bringup yolo.launch.py --show-args` for the full list, and see the [parameter and alias reference](#parameter-and-alias-reference) for every key, default and CLI name.

Sections are keyed by the node's fully qualified name, so the key must include the launch namespace (default `yolo`):

```yaml
/yolo/yolo_node:
  ros__parameters:
    plugins: ["detection"]
    cameras: ["cam0"]
    cam0:
      rgb:
        topic: /camera/rgb/image_raw
    detection:
      plugin: yolo_ros/DetectionPlugin
      cameras: ["cam0"]
      model:
        type: auto
```

If you change `namespace:=`, update the matching config block names for any value you do not override on the command line. Every parameter is documented below, and the [parameter and alias reference](#parameter-and-alias-reference) lists the complete set with defaults and CLI names.

#### Plugin instances (`plugins`)

- **plugins**: Ordered list of plugin instance names loaded by the node; the order defines the data-flow chain. Each instance selects its pluginlib class with a `<instance>.plugin` parameter (the conventional instances map `detection` → `yolo_ros/DetectionPlugin`, `tracking` → `yolo_ros/TrackingPlugin`, `detection3d` → `yolo_ros/Detect3DPlugin` and `debug` → `yolo_ros/DebugPlugin`) and reads its parameters under `<instance>.*`. The first entry must be `yolo_ros/DetectionPlugin`, no plugin type may appear twice and `yolo_ros/DebugPlugin` must be last; any subset starting with detection is valid (`["detection"]`, `["detection", "debug"]`, `["detection", "detection3d", "debug"]`, ...). Camera selection follows the same chain: each plugin's `cameras` list must be a subset of the previous plugin's selection (the first plugin consumes the camera frames directly). A camera selected only downstream is a configure error — e.g. `plugin 'tracking': camera 'cam1' is not produced by plugin 'detection'` — and a camera listed twice in one instance is rejected as a duplicate.

#### Cameras (`cameras` and `<cam>.*`)

The node-level `cameras` parameter lists the camera names. Each name declares its own fields:

- **&lt;cam&gt;.rgb.topic** (required): RGB `sensor_msgs/Image` topic.
- **&lt;cam&gt;.depth.topic** / **&lt;cam&gt;.depth.info_topic**: optional depth image and its `sensor_msgs/CameraInfo`; they must be set together (setting only one is a configure error).
- **&lt;cam&gt;.rgb.reliability** / **&lt;cam&gt;.depth.reliability**: QoS reliability as an integer: `0`=system default, `1`=Reliable, any other value=Best Effort (defaults: `2`).

Camera names must be non-empty, unique and free of `.`, `:`, `/`, and no topic may be used by two cameras. Camera frames are synchronized and stay in-process, so these topics are the node's only image subscriptions (the only other subscription is the `tf2` listener used by `detection3d`).

#### Detection plugin (`detection.*`, `yolo_ros/DetectionPlugin`)

Parameters are grouped into general settings and per-provider settings.

##### General parameters

- **model.type** (CLI `model_type`): Pipeline to run (case-insensitive): `auto` (default), `yolo`/`detect`/`det`/`detection`, any value containing `segment`, any value containing `pose`/`keypoint` or exactly `kpt`, any value containing `obb`/`rotat`, or any value containing `class` or exactly `cls`/`clf`. `auto` infers the task from the model file name (`segment`/`-seg`/`_seg`, `pose`, `obb`, `cls`/`classify`) and falls back to detection. Use `segment` (not `seg`) as an explicit type; `-seg` is only a filename heuristic.
- **model.path** (CLI `model_path`): Path to the ONNX model (default: empty; the shipped configs set `model.repo`/`model.filename` instead).
- **model.repo** / **model.filename** / **model.force_download** / **model.cache_dir** (CLI `model_repo` / `model_filename` / `force_download` / `cache_dir`): Hugging Face Hub download, used instead of `model.path` when both `model.repo` and `model.filename` are set (defaults: empty / empty / `false` / `~/.cache/huggingface/hub`). If the download fails, the node logs the error and falls back to `model.path`. The shipped configs default to the `unileon-robotics/YOLO26-ONNX` mirror; clear `model.repo` to fall back to the local `model.path`. (The VENTUNO Q preset, `yolo_qnn.yaml`, downloads the mirror's `qnn/yolo26s_qnn.onnx` by default; pass `model_path:=` to use a local Ultralytics QNN export instead, since a set `model.repo` wins over `model.path` when the download succeeds.)
- **provider** (CLI `provider`): Execution provider: `auto` (default; CUDA → CPU), `tensorrt`/`trt` (TensorRT → CUDA → CPU), `cuda` (CUDA → CPU), `qnn` (Qualcomm Hexagon NPU via the ONNX Runtime QNN EP → CPU) or `cpu` (CPU only). The chain is filtered by the providers compiled into the ONNX Runtime build; a requested provider that is unavailable logs a warning and the next entry is used. An unknown value is treated as `auto`.
- **device** (CLI `device`): Device ordinal, e.g. `cuda:0`, `trt:1`, `qnn:0` or plain `1` (default: `cuda:0`). The prefix is a provider hint (`cuda:`, `tensorrt:`/`trt:`, `qnn:`, `cpu:`), but `provider` selects the execution provider; when the two disagree the provider wins and a warning is logged. The QNN EP ignores the ordinal and runs on the board's HTP.
- **cameras** (YAML-only): Cameras to run inference on; omit the list to select every defined camera (never write `cameras: []`). Detection is first in the chain, so it can select any defined camera. One camera uses the direct path, several run the dynamic-batch path.
- **inference.threshold** (CLI `threshold`): Detection confidence threshold (default: `0.7`).
- **inference.iou** (CLI `iou`): IoU threshold for the C++ NMS applied to raw-output exports (detection without a baked-NMS head, segmentation, pose and OBB); it has no effect on exports that bake NMS into the graph (default: `0.45`).
- **inference.max_det** (CLI `max_det`): Maximum number of detections per image (default: `300`); negative values are clamped to `0`.
- **batch.max_size** (CLI `max_batch_size`): Maximum batch size for the multi-camera path (default: `8`; clamped to at least `1`). The scheduler drains up to this many camera frames per inference; smaller batches are padded only when `batch.pad_to` is set. `provider: qnn` forces it to `1`, because the QNN EP runs static batch-1 models (a warning is logged).
- **batch.pad_to** (CLI `pad_to_batch`): Pad every dynamic-batch inference to this size (default: `0`, disabled; clamped to `[0, batch.max_size]`). The batch is padded by repeating its last image, and padded results are dropped. Use it when the camera batch size varies from cycle to cycle; it applies with any provider. Ignored for models with a fixed batch axis, which always use the batch size baked into the graph. See the TensorRT section for the engine-cache benefit.
- **inference.max_fps** (CLI `max_fps`): Cap the inference/publish rate in Hz; `0` = unlimited (default: `0`). With several cameras the cap applies across the detection loop, not per camera.
- **input.width** / **input.height** (CLI `img_width` / `img_height`): Network input size for models exported with a dynamic input (`dynamic=True`) (defaults: `640`/`480`). Models with a static input keep the size baked into the ONNX graph (a warning is logged when the parameters differ); non-positive values fail configure and values not divisible by 32 log a warning.
- **inference.top_k** (CLI `top_k`): Classification only — number of top classes to publish (default: `5`).

The provider-specific parameters below apply only when the matching provider is selected. `auto` tries CUDA and then CPU; TensorRT is used only when requested explicitly.

##### TensorRT provider parameters (`provider: tensorrt` or `trt`)

- **trt.fp16_enable** (CLI `trt_fp16_enable`): TensorRT FP16 precision (default: `true`). The flag is part of the engine cache name, so toggling it builds and caches a separate engine.
- **trt.engine_cache_enable** (CLI `trt_engine_cache_enable`): Persist built TensorRT engines (default: `true`). With a dynamic batch axis the engine is built on the first inference and can take several minutes.
- **trt.engine_cache_path** (CLI `trt_engine_cache_path`): TensorRT engine cache base directory; empty → `~/.cache/yolo_ros/trt_engines/<model>` (default: empty).
- **batch.pad_to** (CLI `pad_to_batch`): Recommended with a dynamic-batch export. ONNX Runtime 1.6's TensorRT provider learns the optimization profile from the batch sizes it observes and rebuilds the engine whenever a new size falls outside the saved `[min,max]` range. Padding every inference to a fixed size (e.g. `4` with four cameras) keeps that profile stable, so the cached engine is reused instead of rebuilt: no multi-minute stall when the first frame arrives alone or when the camera batch size changes. On a cold cache it also collapses the usual `[1,1]` then `[1,4]` sequence into a single `[4,4]` build. Same syntax and default as in General parameters.

##### CUDA provider parameters (`provider: cuda`)

- **cuda.graph_enable** (CLI `cuda_graph_enable`): Capture the fixed-shape model as a CUDA graph on the CUDA provider, cutting per-kernel launch overhead (default: `true`). Only applies to fixed-batch models with a float32 input; dynamic-batch exports keep the plain path, and legacy ONNX Runtime builds ignore it with a warning.

##### QNN provider parameters (`provider: qnn`)

The defaults target the VENTUNO Q, QCS8275 / HTP v75:

| Parameter                           | CLI argument                        | Default        | Description                                                                                                                                                                                                                                                                                                                                        |
| ----------------------------------- | ----------------------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `qnn.htp_performance_mode`          | `qnn_htp_performance_mode`          | `burst`        | HTP performance mode: `burst`, `balanced`, `sustained_high_performance`, `power_saver`, ... (see the QNN HTP documentation).                                                                                                                                                                                                                       |
| `qnn.htp_arch`                      | `qnn_htp_arch`                      | `75`           | Hexagon HTP architecture version; must match the board and the model's compile target (e.g. `73`, `79`, `81`).                                                                                                                                                                                                                                     |
| `qnn.soc_model`                     | `qnn_soc_model`                     | `82`           | QNN SoC model number (QCS8275); change it together with `qnn.htp_arch` when targeting another Qualcomm board.                                                                                                                                                                                                                                      |
| `qnn.backend_path`                  | `qnn_backend_path`                  | `libQnnHtp.so` | HTP backend library the EP loads, resolved from the loader path; empty lets ONNX Runtime resolve its own backend. Point it at `<sdk>/lib/<arch>/libQnnHtp.so` for a QAIRT SDK prefix.                                                                                                                                                              |
| `qnn.offload_graph_io_quantization` | `qnn_offload_graph_io_quantization` | `true`         | Offload graph I/O quant/dequant to the CPU EP, keeping graph inputs/outputs float32.                                                                                                                                                                                                                                                               |
| `qnn.enable_htp_fp16_precision`     | `qnn_enable_htp_fp16_precision`     | `false`        | Finalize float32 graphs at fp16 precision on the HTP, which is what lets non-quantized models run on the NPU; quantized models keep it off.                                                                                                                                                                                                        |
| `qnn.context_cache_enable`          | `qnn_context_cache_enable`          | `true`         | Compile a plain quantized (QDQ) model once on first use and cache the serialized QNN context; disable it for models that already embed a context binary (e.g. Ultralytics `*_qnn.onnx` exports).                                                                                                                                                   |
| `qnn.context_cache_path`            | `qnn_context_cache_path`            | empty          | Context cache base directory; empty → `~/.cache/yolo_ros/qnn_context/<model>-<fingerprint>.ctx.onnx` where `<model>` is the model file stem. The fingerprint covers the model's path/size/mtime, so updated or same-named models compile a fresh context; delete the directory after a QAIRT/OS upgrade or to reclaim space from old fingerprints. |

##### CPU provider parameters (`provider: cpu`)

- **cpu.n_threads** (CLI `n_threads`): CPU execution-provider threads; `-1` (or any value <= 0) = auto: performance cores on hybrid CPUs, otherwise physical cores (default: `-1`). Only used by the CPU provider; CUDA and TensorRT sessions run with a single ONNX Runtime intra-op thread.

#### Tracking plugin (`tracking.*`, `yolo_ros/TrackingPlugin`)

- **cameras** (YAML-only): Cameras to track; omit the list to select every defined camera (never write `cameras: []`). Every selected camera must also be selected by the previous plugin in the chain.
- **tracker_type** (CLI `tracker_type`): C++ implementation (case-insensitive): `bytetrack` (default) or `botsort` (XYWH Kalman filter + camera-motion compensation). The tracker knobs below live in this same instance block; only the selected tracker's knobs are declared, and an unknown `tracker_type` logs a warning and passes detections through.
- **track.high_thresh** (CLI `track_high_thresh`): First-stage association threshold (default: `0.25`).
- **track.low_thresh** (CLI `track_low_thresh`): Second-stage threshold for low-score matches (default: `0.1`).
- **track.new_thresh** (CLI `new_track_thresh`): Score above which a new track is started (default: `0.25`).
- **track.buffer** (CLI `track_buffer`): Frames a lost track is kept alive (default: `30`).
- **track.match_thresh** (CLI `match_thresh`): Association similarity threshold (IoU/cost) (default: `0.8`).
- **track.fuse_score** (CLI `fuse_score`): Fuse detection score with IoU cost for matching (default: `true`).
- **gmc.method** (CLI `gmc_method`, BoT-SORT only): Camera-motion method (case-insensitive): `none` (default), `sparseOptFlow`, `orb` or `ecc`; an unknown method disables compensation with a warning.
- **gmc.downscale** (CLI `gmc_downscale`, BoT-SORT only): Camera-motion downscale factor (default: `2`; values below `1` are clamped to `1`).
- **reid.enable** / **reid.model** / **reid.proximity_thresh** / **reid.appearance_thresh** (CLI `with_reid` / `reid_model` / `proximity_thresh` / `appearance_thresh`, BoT-SORT only): ReID appearance model and its matching thresholds (defaults: `false` / empty / `0.5` / `0.25`). Set `reid.enable: true` and point `reid.model` at an ONNX encoder (export recipes in the [model export guide](docs/models.md#reid-encoder-for-bot-sort-reid)); if the model is empty or fails to load, the tracker logs a warning and runs motion-only.
- **reid.provider** / **reid.device** (CLI `provider` / `device`, BoT-SORT + ReID only): ReID model execution provider and device (defaults: `auto` / `cuda:0`). These are the same CLI aliases as the detector's `provider` / `device`, so one launch argument sets both; use YAML for independent values.

#### 3D detection plugin (`detection3d.*`, `yolo_ros/Detect3DPlugin`)

- **cameras** (YAML-only): Cameras to lift to 3D; omit the list to select every defined camera (never write `cameras: []`), every selected camera must have the `depth.topic` + `depth.info_topic` pair, and it must also be selected by the previous plugin in the chain.
- **target_frame** (CLI `target_frame`): Frame to transform the 3D boxes and keypoints into; the transform is looked up from each camera frame with `tf2` (default: `base_link`).
- **depth.units_divisor** (CLI `depth_image_units_divisor`): Divisor applied to 16-bit depth images to convert them to meters (default: `1000`). 32-bit float depth images are already metric and ignore the divisor.
- **orientation.enable** (CLI `enable_orientation`): Estimate and publish the 3D box orientation (default: `false`).
- **orientation.min_seg_points** (CLI `min_seg_points_for_orientation`): Minimum valid depth points required per detection for orientation estimation (default: `20`).

#### Debug plugin (`debug.*`, `yolo_ros/DebugPlugin`)

- **cameras** (YAML-only): Cameras to render; omit the list to select every defined camera (never write `cameras: []`) and every selected camera must also be selected by the previous plugin. Debug is terminal, so the plugin must be last in the chain.
- **marker_lifetime** (CLI `marker_lifetime`): RViz marker lifetime in seconds (default: `0.5`; `0` keeps markers persistent, negative values are clamped to `0`).
- **compressed.enable** (CLI `debug_compressed_enable`): Also publish `<cam>/debug_image/compressed` as a JPEG `sensor_msgs/CompressedImage` (default: `true`). Encoding is skipped while the topic has no subscribers, so it costs no CPU when nobody listens.
- **compressed.quality** (CLI `debug_compressed_quality`): JPEG quality for the compressed debug image, clamped to `1`-`100` (default: `80`).

The plugin publishes `<cam>/debug_image` from the camera frame synchronized with the freshest non-3D upstream (the stream before `detection3d` when it is in the chain), so the image is not gated by the slower 3D stream. The RViz markers `<cam>/debug_bb_markers` and `<cam>/debug_kp_markers` are built from its chain input (the previous plugin's output) whenever those detections carry `bbox3d`/`keypoints3d` data (i.e. when `detection3d` precedes debug in the chain).

#### Parameter and alias reference

The tables below list every key with its type, declaration default and command-line name; YAML-only keys are listed without a CLI column. The shipped presets override some of these defaults (for example `config/yolo.yaml` sets `detection.provider: cuda`). Arguments left unset keep the YAML value (or the declaration default), and scalar values are converted to the declared type by the launch file; boolean arguments accept `true`/`1`/`yes`/`on` and `false`/`0`/`no`/`off`. `provider` and `device` are shared aliases: when the tracking instance selects BoT-SORT, `provider:=` and `device:=` also override `tracking.reid.provider` and `tracking.reid.device`, so use YAML when the detector and ReID encoder need different values.

**Launch arguments**

| Launch argument | Default            | Description                                                         |
| --------------- | ------------------ | ------------------------------------------------------------------- |
| `params_file`   | `config/yolo.yaml` | Pipeline YAML with the cameras, ordered plugins and all parameters. |
| `namespace`     | `yolo`             | Node namespace; the YAML section key is `/<namespace>/yolo_node`.   |

**Node, plugin-instance and camera parameters (YAML-only)**

| YAML key                  | Type       | Default                           | Description                                                                                            |
| ------------------------- | ---------- | --------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `plugins`                 | `string[]` | empty                             | Ordered plugin instance names; the order is the data-flow chain.                                       |
| `cameras`                 | `string[]` | empty                             | Camera names; each name declares its own `<cam>.*` block.                                              |
| `<instance>.plugin`       | `string`   | required                          | Pluginlib class of the instance (`yolo_ros/DetectionPlugin`, `yolo_ros/TrackingPlugin`, ...).          |
| `<instance>.cameras`      | `string[]` | every defined camera when omitted | Cameras the instance processes; must be a subset of the previous plugin's selection.                   |
| `<cam>.rgb.topic`         | `string`   | required                          | RGB `sensor_msgs/Image` topic.                                                                         |
| `<cam>.depth.topic`       | `string`   | empty                             | Depth image topic; must be set together with `<cam>.depth.info_topic`.                                 |
| `<cam>.depth.info_topic`  | `string`   | empty                             | `sensor_msgs/CameraInfo` for the depth image; must be set together with `<cam>.depth.topic`.           |
| `<cam>.rgb.reliability`   | `int`      | `2`                               | RGB QoS reliability: `0` = system default, `1` = Reliable, any other value = Best Effort.              |
| `<cam>.depth.reliability` | `int`      | `2`                               | Depth/CameraInfo QoS reliability: `0` = system default, `1` = Reliable, any other value = Best Effort. |

**Detection plugin (`detection.*`, `yolo_ros/DetectionPlugin`)**

| Parameter                                     | CLI argument                        | Type     | Default                    |
| --------------------------------------------- | ----------------------------------- | -------- | -------------------------- |
| `detection.model.type`                        | `model_type`                        | `string` | `auto`                     |
| `detection.model.path`                        | `model_path`                        | `string` | empty                      |
| `detection.model.repo`                        | `model_repo`                        | `string` | empty                      |
| `detection.model.filename`                    | `model_filename`                    | `string` | empty                      |
| `detection.model.cache_dir`                   | `cache_dir`                         | `string` | `~/.cache/huggingface/hub` |
| `detection.model.force_download`              | `force_download`                    | `bool`   | `false`                    |
| `detection.provider`                          | `provider`                          | `string` | `auto`                     |
| `detection.device`                            | `device`                            | `string` | `cuda:0`                   |
| `detection.inference.threshold`               | `threshold`                         | `double` | `0.7`                      |
| `detection.inference.iou`                     | `iou`                               | `double` | `0.45`                     |
| `detection.inference.max_det`                 | `max_det`                           | `int`    | `300`                      |
| `detection.inference.max_fps`                 | `max_fps`                           | `int`    | `0`                        |
| `detection.inference.top_k`                   | `top_k`                             | `int`    | `5`                        |
| `detection.input.width`                       | `img_width`                         | `int`    | `640`                      |
| `detection.input.height`                      | `img_height`                        | `int`    | `480`                      |
| `detection.batch.max_size`                    | `max_batch_size`                    | `int`    | `8`                        |
| `detection.batch.pad_to`                      | `pad_to_batch`                      | `int`    | `0`                        |
| `detection.cpu.n_threads`                     | `n_threads`                         | `int`    | `-1`                       |
| `detection.cuda.graph_enable`                 | `cuda_graph_enable`                 | `bool`   | `true`                     |
| `detection.trt.fp16_enable`                   | `trt_fp16_enable`                   | `bool`   | `true`                     |
| `detection.trt.engine_cache_enable`           | `trt_engine_cache_enable`           | `bool`   | `true`                     |
| `detection.trt.engine_cache_path`             | `trt_engine_cache_path`             | `string` | empty                      |
| `detection.qnn.htp_performance_mode`          | `qnn_htp_performance_mode`          | `string` | `burst`                    |
| `detection.qnn.htp_arch`                      | `qnn_htp_arch`                      | `string` | `75`                       |
| `detection.qnn.soc_model`                     | `qnn_soc_model`                     | `string` | `82`                       |
| `detection.qnn.backend_path`                  | `qnn_backend_path`                  | `string` | `libQnnHtp.so`             |
| `detection.qnn.offload_graph_io_quantization` | `qnn_offload_graph_io_quantization` | `bool`   | `true`                     |
| `detection.qnn.enable_htp_fp16_precision`     | `qnn_enable_htp_fp16_precision`     | `bool`   | `false`                    |
| `detection.qnn.context_cache_enable`          | `qnn_context_cache_enable`          | `bool`   | `true`                     |
| `detection.qnn.context_cache_path`            | `qnn_context_cache_path`            | `string` | empty                      |

**Tracking plugin (`tracking.*`, `yolo_ros/TrackingPlugin`)**

| Parameter                         | CLI argument        | Type     | Default     |
| --------------------------------- | ------------------- | -------- | ----------- |
| `tracking.tracker_type`           | `tracker_type`      | `string` | `bytetrack` |
| `tracking.track.high_thresh`      | `track_high_thresh` | `double` | `0.25`      |
| `tracking.track.low_thresh`       | `track_low_thresh`  | `double` | `0.1`       |
| `tracking.track.new_thresh`       | `new_track_thresh`  | `double` | `0.25`      |
| `tracking.track.buffer`           | `track_buffer`      | `int`    | `30`        |
| `tracking.track.match_thresh`     | `match_thresh`      | `double` | `0.8`       |
| `tracking.track.fuse_score`       | `fuse_score`        | `bool`   | `true`      |
| `tracking.gmc.method`             | `gmc_method`        | `string` | `none`      |
| `tracking.gmc.downscale`          | `gmc_downscale`     | `int`    | `2`         |
| `tracking.reid.enable`            | `with_reid`         | `bool`   | `false`     |
| `tracking.reid.model`             | `reid_model`        | `string` | empty       |
| `tracking.reid.proximity_thresh`  | `proximity_thresh`  | `double` | `0.5`       |
| `tracking.reid.appearance_thresh` | `appearance_thresh` | `double` | `0.25`      |
| `tracking.reid.provider`          | `provider` (shared) | `string` | `auto`      |
| `tracking.reid.device`            | `device` (shared)   | `string` | `cuda:0`    |

`gmc.*` and `reid.*` are only declared when `tracker_type: botsort`; ByteTrack declares only `tracker_type` plus `track.*`.

**3D detection plugin (`detection3d.*`, `yolo_ros/Detect3DPlugin`)**

| Parameter                                | CLI argument                     | Type     | Default     |
| ---------------------------------------- | -------------------------------- | -------- | ----------- |
| `detection3d.target_frame`               | `target_frame`                   | `string` | `base_link` |
| `detection3d.depth.units_divisor`        | `depth_image_units_divisor`      | `int`    | `1000`      |
| `detection3d.orientation.enable`         | `enable_orientation`             | `bool`   | `false`     |
| `detection3d.orientation.min_seg_points` | `min_seg_points_for_orientation` | `int`    | `20`        |

**Debug plugin (`debug.*`, `yolo_ros/DebugPlugin`)**

| Parameter                  | CLI argument               | Type     | Default |
| -------------------------- | -------------------------- | -------- | ------- |
| `debug.marker_lifetime`    | `marker_lifetime`          | `double` | `0.5`   |
| `debug.compressed.enable`  | `debug_compressed_enable`  | `bool`   | `true`  |
| `debug.compressed.quality` | `debug_compressed_quality` | `int`    | `80`    |

`detection.cameras`, `tracking.cameras`, `detection3d.cameras` and `debug.cameras` are YAML-only as well (the `cameras` row in the node table applies to every instance).

### Writing a plugin

Pipeline stages are **pluginlib** plugins. To add your own:

1. Derive from `yolo_ros::Plugin` and implement `declare_params`, `get_params`, `output_channel`, `setup`, `activate`, `deactivate` and `run` (the worker loop, which must poll its `stop` argument and return promptly). The two parameter methods receive the owning lifecycle node and the instance `prefix`, a runtime string ending in a dot (e.g. `"detection."`), so a plugin declares and reads every parameter under its `<instance>.*` namespace without hard-coding the instance name:

   ```cpp
   void declare_params(rclcpp_lifecycle::LifecycleNode &node,
                       const std::string &prefix) override {
     node.declare_parameter<double>(prefix + "inference.threshold", 0.7);
   }

   void get_params(const rclcpp_lifecycle::LifecycleNode &node,
                   const std::string &prefix) override {
     node.get_parameter(prefix + "inference.threshold", this->threshold_);
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

The whole repository is licensed under the **MIT License**. See the root
[`LICENSE`](./LICENSE) file; the same license applies to every ROS 2 package in
this repository (`yolo_ros`, `yolo_msgs`, `yolo_bringup` and
`yolo_hfhub_vendor`).

Third-party notices are installed with the applicable packages
(`yolo_ros/THIRD_PARTY_NOTICES.md` and
`yolo_hfhub_vendor/THIRD_PARTY_NOTICES.md`).

The C++ pipeline adapts behavior from the original `yolo_ros` Python nodes; those contributions were authorized by their copyright holder for release in the MIT-licensed C++ pipeline (see `THIRD_PARTY_NOTICES.md`).

Model weights and exported ONNX files are separate artifacts and remain subject to their respective licenses; the MIT license for the C++ pipeline does not relicense them. In particular, Ultralytics models are AGPL-3.0 — see the [Models](#models) section.
