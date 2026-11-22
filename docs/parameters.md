# Parameters

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

### Plugin instances (`plugins`)

- **plugins**: Ordered list of plugin instance names loaded by the node; the order defines the data-flow chain. Each instance selects its pluginlib class with a `<instance>.plugin` parameter (the conventional instances map `detection` → `yolo_ros/DetectionPlugin`, `tracking` → `yolo_ros/TrackingPlugin`, `detection3d` → `yolo_ros/Detect3DPlugin` and `debug` → `yolo_ros/DebugPlugin`) and reads its parameters under `<instance>.*`. The first entry must be `yolo_ros/DetectionPlugin`, no plugin type may appear twice and `yolo_ros/DebugPlugin` must be last; any subset starting with detection is valid (`["detection"]`, `["detection", "debug"]`, `["detection", "detection3d", "debug"]`, ...). Camera selection follows the same chain: each plugin's `cameras` list must be a subset of the previous plugin's selection (the first plugin consumes the camera frames directly). A camera selected only downstream is a configure error — e.g. `plugin 'tracking': camera 'cam1' is not produced by plugin 'detection'` — and a camera listed twice in one instance is rejected as a duplicate.

### Cameras (`cameras` and `<cam>.*`)

The node-level `cameras` parameter lists the camera names. Each name declares its own fields:

- **&lt;cam&gt;.rgb.topic** (required): RGB `sensor_msgs/Image` topic.
- **&lt;cam&gt;.depth.topic** / **&lt;cam&gt;.depth.info_topic**: optional depth image and its `sensor_msgs/CameraInfo`; they must be set together (setting only one is a configure error).
- **&lt;cam&gt;.rgb.reliability** / **&lt;cam&gt;.depth.reliability**: QoS reliability as an integer: `0`=system default, `1`=Reliable, any other value=Best Effort (defaults: `2`).

Camera names must be non-empty, unique and free of `.`, `:`, `/`, and no topic may be used by two cameras. Camera frames are synchronized and stay in-process, so these topics are the node's only image subscriptions (the only other subscription is the `tf2` listener used by `detection3d`).

### Detection plugin (`detection.*`, `yolo_ros/DetectionPlugin`)

Parameters are grouped into general settings and per-provider settings.

#### General parameters

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

#### TensorRT provider parameters (`provider: tensorrt` or `trt`)

- **trt.fp16_enable** (CLI `trt_fp16_enable`): TensorRT FP16 precision (default: `true`). The flag is part of the engine cache name, so toggling it builds and caches a separate engine.
- **trt.engine_cache_enable** (CLI `trt_engine_cache_enable`): Persist built TensorRT engines (default: `true`). With a dynamic batch axis the engine is built on the first inference and can take several minutes.
- **trt.engine_cache_path** (CLI `trt_engine_cache_path`): TensorRT engine cache base directory; empty → `~/.cache/yolo_ros/trt_engines/<model>` (default: empty).
- **batch.pad_to** (CLI `pad_to_batch`): Recommended with a dynamic-batch export. ONNX Runtime 1.6's TensorRT provider learns the optimization profile from the batch sizes it observes and rebuilds the engine whenever a new size falls outside the saved `[min,max]` range. Padding every inference to a fixed size (e.g. `4` with four cameras) keeps that profile stable, so the cached engine is reused instead of rebuilt: no multi-minute stall when the first frame arrives alone or when the camera batch size changes. On a cold cache it also collapses the usual `[1,1]` then `[1,4]` sequence into a single `[4,4]` build. Same syntax and default as in General parameters.

#### CUDA provider parameters (`provider: cuda`)

- **cuda.graph_enable** (CLI `cuda_graph_enable`): Capture the fixed-shape model as a CUDA graph on the CUDA provider, cutting per-kernel launch overhead (default: `true`). Only applies to fixed-batch models with a float32 input; dynamic-batch exports keep the plain path, and legacy ONNX Runtime builds ignore it with a warning.

#### QNN provider parameters (`provider: qnn`)

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

#### CPU provider parameters (`provider: cpu`)

- **cpu.n_threads** (CLI `n_threads`): CPU execution-provider threads; `-1` (or any value <= 0) = auto: performance cores on hybrid CPUs, otherwise physical cores (default: `-1`). Only used by the CPU provider; CUDA and TensorRT sessions run with a single ONNX Runtime intra-op thread.

### Tracking plugin (`tracking.*`, `yolo_ros/TrackingPlugin`)

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
- **reid.enable** / **reid.model** / **reid.proximity_thresh** / **reid.appearance_thresh** (CLI `with_reid` / `reid_model` / `proximity_thresh` / `appearance_thresh`, BoT-SORT only): ReID appearance model and its matching thresholds (defaults: `false` / empty / `0.5` / `0.25`). Set `reid.enable: true` and point `reid.model` at an ONNX encoder (export recipes in the [model export guide](models.md#reid-encoder-for-bot-sort-reid)); if the model is empty or fails to load, the tracker logs a warning and runs motion-only.
- **reid.provider** / **reid.device** (CLI `provider` / `device`, BoT-SORT + ReID only): ReID model execution provider and device (defaults: `auto` / `cuda:0`). These are the same CLI aliases as the detector's `provider` / `device`, so one launch argument sets both; use YAML for independent values.

### 3D detection plugin (`detection3d.*`, `yolo_ros/Detect3DPlugin`)

- **cameras** (YAML-only): Cameras to lift to 3D; omit the list to select every defined camera (never write `cameras: []`), every selected camera must have the `depth.topic` + `depth.info_topic` pair, and it must also be selected by the previous plugin in the chain.
- **target_frame** (CLI `target_frame`): Frame to transform the 3D boxes and keypoints into; the transform is looked up from each camera frame with `tf2` (default: `base_link`).
- **depth.units_divisor** (CLI `depth_image_units_divisor`): Divisor applied to 16-bit depth images to convert them to meters (default: `1000`). 32-bit float depth images are already metric and ignore the divisor.
- **orientation.enable** (CLI `enable_orientation`): Estimate and publish the 3D box orientation (default: `false`).
- **orientation.min_seg_points** (CLI `min_seg_points_for_orientation`): Minimum valid depth points required per detection for orientation estimation (default: `20`).

### Debug plugin (`debug.*`, `yolo_ros/DebugPlugin`)

- **cameras** (YAML-only): Cameras to render; omit the list to select every defined camera (never write `cameras: []`) and every selected camera must also be selected by the previous plugin. Debug is terminal, so the plugin must be last in the chain.
- **marker_lifetime** (CLI `marker_lifetime`): RViz marker lifetime in seconds (default: `0.5`; `0` keeps markers persistent, negative values are clamped to `0`).
- **compressed.enable** (CLI `debug_compressed_enable`): Also publish `<cam>/debug_image/compressed` as a JPEG `sensor_msgs/CompressedImage` (default: `true`). Encoding is skipped while the topic has no subscribers, so it costs no CPU when nobody listens.
- **compressed.quality** (CLI `debug_compressed_quality`): JPEG quality for the compressed debug image, clamped to `1`-`100` (default: `80`).

The plugin publishes `<cam>/debug_image` from the camera frame synchronized with the freshest non-3D upstream (the stream before `detection3d` when it is in the chain), so the image is not gated by the slower 3D stream. The RViz markers `<cam>/debug_bb_markers` and `<cam>/debug_kp_markers` are built from its chain input (the previous plugin's output) whenever those detections carry `bbox3d`/`keypoints3d` data (i.e. when `detection3d` precedes debug in the chain).

### Parameter and alias reference

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
