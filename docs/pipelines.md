# Multi-camera pipelines

One `yolo_node`, one YAML file, several cameras. Cameras are declared once on the node; every plugin lists the cameras it processes (a subset of the previous plugin's selection), and the `plugins` order defines the data-flow chain. `DetectionPlugin` batches every camera it selects through one shared model, while tracking, 3D and debug can run on different camera subsets.

```shell
ros2 launch yolo_bringup yolo.launch.py params_file:=/path/multi_camera.yaml
```

## One YAML

The whole pipeline lives in a single params file: the node-level `cameras` list with a block per camera, then the ordered `plugins` chain and each plugin's parameters. A two-camera example:

```yaml
/yolo/yolo_node:
  ros__parameters:
    plugins: ["detection", "tracking", "detection3d", "debug"]
    cameras: ["cam0", "cam1"]
    cam0:
      rgb_topic: /cam0/color/image_raw
      depth_topic: /cam0/depth/image_raw # optional pair, set together
      depth_info_topic: /cam0/depth/camera_info
      image_reliability: 2
      depth_reliability: 2
    cam1:
      rgb_topic: /cam1/color/image_raw
    detection:
      plugin: yolo_ros/DetectionPlugin
      cameras: ["cam0", "cam1"] # omit to select all cameras
      model_repo: unileon-robotics/YOLO26-ONNX
      model_filename: dynamic/yolo26m.onnx # dynamic-batch export
      max_batch_size: 8
    tracking:
      plugin: yolo_ros/TrackingPlugin
      cameras: ["cam0", "cam1"]
      tracker_type: bytetrack
      track_high_thresh: 0.25
      track_low_thresh: 0.1
      new_track_thresh: 0.25
      track_buffer: 30
      match_thresh: 0.8
      fuse_score: true
    detection3d:
      plugin: yolo_ros/Detect3DPlugin
      cameras: ["cam0"] # every camera here must have depth
      target_frame: camera_link
    debug:
      plugin: yolo_ros/DebugPlugin
      cameras: ["cam0"]
      marker_lifetime: 0.5
```

Rules:

- `cameras` is a plain name list; each name must be non-empty and contain none of `.`, `:`, `/`; names are unique.
- `rgb_topic` is required per camera. `depth_topic` and `depth_info_topic` are optional but must be set together; `image_reliability` / `depth_reliability` are optional ints (`0`=system default, `1`=Reliable, `2`=Best Effort, default `2`).
- No topic may be used by two different cameras.
- A plugin's `cameras` list selects a subset of the defined cameras. Omit the list (or list every camera) to select every defined camera; never write an empty list (`cameras: []`, `plugins: []`) — see the Jazzy note below. Entries must reference defined cameras, must not repeat a name inside one instance, and `detection3d` cameras must have depth.
- A plugin can only select cameras the previous plugin in the chain also selects: the first plugin consumes the camera frames directly, so every camera listed by plugin _i_ must also be listed by plugin _i−1_. A camera selected only downstream is a configure error: `plugin '<instance>': camera '<cam>' is not produced by plugin '<previous instance>'`.
- The `plugins` order is the chain: the first entry must be `yolo_ros/DetectionPlugin`, no plugin type may appear twice and `yolo_ros/DebugPlugin` must be last. Valid subsets include `["detection"]`, `["detection", "tracking"]`, `["detection", "detection3d", "debug"]`, ...

> **Jazzy quirk**: an empty sequence in the YAML params file (`cameras: []`, `plugins: []`) is not a valid way to say "all cameras" or "no plugins": rcl's YAML parser produces no value for `[]` and the node fails to start. Omit the key instead; for a plugin's `cameras`, omitting it means every defined camera (or list every camera explicitly).

Every scalar parameter can still be overridden from the command line (arrays and per-camera blocks stay YAML-only); see the [README](../README.md#parameters).

## Chain semantics

For plugin _i_, the input of camera `cam` is the channel `output_channel(cam)` produced by plugin _i−1_; for the first plugin it is the camera's internal `CameraFrame` channel. A plugin can therefore only select cameras that plugin _i−1_ also selects — there is no upstream channel for the others, and configure fails with `... camera '<cam>' is not produced by plugin '<previous instance>'` if one is listed. `CameraStreams` synchronizes each camera's RGB (and depth + `CameraInfo` when configured) and publishes the frames in-process, so camera images never travel over DDS. Each plugin runs on its own worker thread and every output channel is exposed as a ROS topic by the reduced `TopicRegistry`.

| Plugin                     | Consumes (per camera)                            | Produces (per camera)                       |
| -------------------------- | ------------------------------------------------ | ------------------------------------------- |
| `yolo_ros/DetectionPlugin` | `CameraFrame`                                    | `<cam>/detections`                          |
| `yolo_ros/TrackingPlugin`  | `CameraFrame` + previous detections              | `<cam>/tracking`                            |
| `yolo_ros/Detect3DPlugin`  | `CameraFrame` (with depth) + previous detections | `<cam>/detections_3d`                       |
| `yolo_ros/DebugPlugin`     | `CameraFrame` + previous detections              | `<cam>/debug_image` + the two marker arrays |

The 3D plugin must come after the detections it should lift (tracking, or detection when tracking is omitted); debug is terminal and draws whatever the previous plugin published, adding RViz markers when those detections carry `bbox3d`/`keypoints3d` data (i.e. when `detection3d` precedes it).

## Per-camera outputs

Each camera's outputs live under `/<namespace>/<cam>/`:

- **detections**: the raw detector stream (batched when several cameras are selected).
- **tracking**: tracked objects with stable IDs (tracking plugin in the chain).
- **detections_3d**: 3D boxes/keypoints (3D plugin in the chain and the camera has depth).
- **debug_image**, **debug_bb_markers**, **debug_kp_markers**: annotated image and RViz marker arrays (debug plugin in the chain).

Because the names derive from the camera, adding a camera needs no extra wiring: declare it in `cameras`, give it a block, and list it in the plugins that should process it — keeping each plugin's list a subset of the previous plugin's.

## Batching scaling

Batching is a **throughput** optimization, not a latency one: per-image latency grows roughly linearly with the batch, while aggregate throughput improves because the GPU work is shared. Measured engine-only (`Model::detect_batch`, no queue/DDS/node overhead) with `yolo26n-dyn.onnx` on the 640×480 frames from the benchmark harness ([methodology](benchmark.md)), on an RTX 3060 (12 GB), median of 50 runs after warm-up; `img/s` is aggregate throughput:

|              cameras (batch) |                 CUDA fp32 |             TensorRT fp16 |
| ---------------------------: | ------------------------: | ------------------------: |
|                            1 |        6.4 ms · 156 img/s |        4.3 ms · 234 img/s |
|                            2 |       11.2 ms · 178 img/s |        7.2 ms · 278 img/s |
|                            3 |       16.3 ms · 184 img/s |       10.9 ms · 275 img/s |
|                            4 |       21.6 ms · 186 img/s |       13.9 ms · 287 img/s |
| 4 × unbatched (same session) | 24.9 ms total · 161 img/s | 16.9 ms total · 237 img/s |

Batching 4 cameras buys ~15-20 % more aggregate throughput than four unbatched runs; the execution provider is the larger lever (~+55 % for TensorRT fp16 over CUDA fp32). The dynamic-batch export itself costs ~0-8 % per image versus the static batch-1 model. These are saturated-batch numbers - with real cameras the latest-frame queue drops stale frames, so partial batches are cheaper and per-camera latency stays bounded. See the [model export guide](models.md#dynamic-batch-export-multi-camera-pipelines) for the dynamic-batch export.
