# Usage

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

Each task ships a preset params file that pins `detection.model.type` and task-specific tuning — `config/yolo_segment.yaml`, `yolo_pose.yaml`, `yolo_obb.yaml`, `yolo_classify.yaml` — plus the VENTUNO Q Hexagon NPU preset (`config/yolo_qnn.yaml`, see [QNN](installation.md#qnn-hexagon-npu-ventuno-q)). Pass one with `params_file:=`:

```shell
ros2 launch yolo_bringup yolo.launch.py params_file:=$(ros2 pkg prefix yolo_bringup)/share/yolo_bringup/config/yolo_segment.yaml
```

Runnable launch commands for each task (detection, segmentation, pose, OBB and classification) are collected in [Demos](demos.md).

## Architecture

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

## 3D Detection

A camera gets a depth stream by setting both `depth.topic` and `depth.info_topic` in its block; `CameraStreams` then synchronizes its RGB, depth and `CameraInfo` into one `CameraFrame`. Adding `detection3d` (class `yolo_ros/Detect3DPlugin`) to the `plugins` chain synchronizes that frame with the upstream 2D detections and publishes `<cam>/detections_3d`. Cameras selected by `detection3d` must have depth and must also be selected by the plugin whose detections it lifts (tracking, or detection when tracking is omitted), and `detection3d` must come after that plugin. The shipped `config/yolo.yaml` already includes `detection3d` over a depth camera; for RGB-only setups remove the depth fields and the `detection3d` entry.

For segmentation the depth ROI is driven by the mask polygon; for pose the 2D keypoints are back-projected to 3D (`<cam>/debug_kp_markers`). The 3D plugin can also estimate the orientation of each box (an oriented bounding box fit by PCA to a strided depth sample) when `orientation.enable` is set. When enabled, `detections_3d` carries a non-identity quaternion in each box's `center.orientation` and the box `size` is expressed along the object's own axes.

## Multi-camera pipelines

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

`DetectionPlugin` runs directly on the single-camera case and batches (up to `batch.max_size`) when it selects several cameras. A dynamic-batch ONNX export gives true batching (the mirror's `dynamic/` files are); a fixed-batch model still works by chunking to the batch axis baked into the graph, so a stock batch-1 export simply processes one image per inference. Outputs stay camera-prefixed (`cam0/detections`, `cam1/detections`, ...). See the [multi-camera pipelines guide](pipelines.md) for the full example and the batching-scaling numbers.

With TensorRT, ONNX Runtime 1.6 only reuses a cached engine when the input batch size is inside the optimization profile saved with it. The scheduler produces variable batch sizes (the first frame often arrives alone), so the engine can be rebuilt for several minutes on the first inference. Set `batch.pad_to` (CLI `pad_to_batch:=`) to a stable size (usually the camera count, e.g. `4`) to keep the profile valid and the cached engine reusable.

## Topics

All output topics are published under the launch namespace (default `yolo`) and are **camera-prefixed**: a camera named `cam0` publishes under `/<namespace>/cam0/`. Whether a topic exists depends on the `plugins` chain and on the camera being in each plugin's `cameras` list:

- **&lt;cam&gt;/detections**: Objects detected by YOLO using the camera's RGB images. Each object contains a bounding box and a class name, plus a mask or a list of keypoints for segmentation/pose models.
- **&lt;cam&gt;/tracking**: Objects detected and tracked by ByteTrack (or BoT-SORT with `tracking.tracker_type: botsort`). Each object is assigned a stable tracking ID.
- **&lt;cam&gt;/detections_3d**: 3D objects detected by `detection3d`. YOLO results are used to crop the depth image and create 3D bounding boxes and keypoints.
- **&lt;cam&gt;/debug_image**: Debug image showing the detected and tracked objects. It can be visualized with `rqt_image_view` or `rviz2`.
- **&lt;cam&gt;/debug_image/compressed**: JPEG `sensor_msgs/CompressedImage` version of the debug image (enabled by default). It is much lighter over WiFi and lets `ros2 topic hz` measure the real frame rate, which the raw image topic is too large for on CPU-constrained boards.
- **&lt;cam&gt;/debug_bb_markers** / **&lt;cam&gt;/debug_kp_markers**: RViz `MarkerArray`s built from the debug plugin's input detection stream; they carry content when `detection3d` precedes debug in the chain (`bbox3d`/`keypoints3d` data).

Synchronized camera frames and the inter-plugin blackboard channels are in-process only — the topics above are the only ROS outputs.

## Services

There are no runtime services. Runtime enable/disable and class filtering were removed entirely: the old `/yolo/enable` and `/yolo/set_classes` services and their `detection.enable` / `detection.classes` startup replacements are gone, so inference always runs and every class is published.
