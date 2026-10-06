# Multi-camera pipelines

One shared detector batched across several cameras, with per-camera tracking / 3D / debug stages, driven by `config/pipelines.yaml`.

```shell
ros2 launch yolo_bringup yolo_pipelines.launch.py
```

A single `yolo_batch_node` subscribes to every camera's image topic and infers them in batches (up to `max_batch_size`); each camera then gets its own nodes under a `/<namespace>/<camera>` namespace: a `tracking_node` when `tracking` is on, a `detect_3d_node` when the camera has a `depth` block, and a `debug_node` when `debug` is on.

`pipelines.yaml` has three parts:

- **`namespace`**: launch namespace for the shared detector (default `yolo`).
- **`model`**: one shared detector block, with the same parameters as the single-camera `yolo_node` plus `max_batch_size`. It must be a **dynamic-batch** ONNX export (`dynamic=True`) — see the [model export guide](models.md). The YOLO26 mirror ships the whole set under `dynamic/`, so the default config just downloads `dynamic/yolo26m.onnx`. A plain batch-1 export still runs, but is processed one image at a time.
- **`cameras`**: a list of camera entries, each with a unique `name` and an `image_topic`, plus its own `tracker` (a `config/trackers/<name>.yaml` selector, e.g. `botsort`; defaults to the top-level `tracker`), `tracking` and `debug` flags, and an optional `depth` block that starts the 3D node (`image_topic` / `info_topic` / `target_frame` / `units_divisor` / `enable_orientation`).

```yaml
namespace: yolo
tracker: bytetrack               # default for cameras that don't set one

model:                           # shared detector (must be a dynamic-batch export)
  model_repo: unileon-robotics/YOLO26-ONNX
  model_filename: dynamic/yolo26m.onnx
  max_batch_size: 8

cameras:
  - name: front
    image_topic: /camera/rgb/image_raw
    tracker: botsort
    tracking: true
    debug: true
    depth:
      image_topic: /camera/depth/image_raw
      info_topic: /camera/depth/camera_info
      target_frame: camera_link
      units_divisor: 1000

  - name: aux
    image_topic: /aux/image_raw
    tracking: false              # no tracker for this camera
    debug: false
    # no depth block -> no 3D stage
```

Every camera's output lives under `/<ns>/<cam>/`: **detections** (the raw batched detector stream), **tracking** (when `tracking` is on), **detections_3d** (when the camera has a `depth` block) and **debug_image**. The launch takes a single `pipeline_file:=` argument to point at a different config (default `config/pipelines.yaml`).

## Batching scaling

Batching is a **throughput** optimization, not a latency one: per-image latency grows roughly linearly with the batch, while aggregate throughput improves because the GPU work is shared. Measured engine-only (`Model::detect_batch`, no queue/DDS/node overhead) with `yolo26n-dyn.onnx` on the 640×480 frames from the benchmark harness ([methodology](benchmark.md)), on an RTX 3060 (12 GB), median of 50 runs after warm-up; `img/s` is aggregate throughput:

| cameras (batch) | CUDA fp32 | TensorRT fp16 |
|--:|--:|--:|
| 1 | 6.4 ms · 156 img/s | 4.3 ms · 234 img/s |
| 2 | 11.2 ms · 178 img/s | 7.2 ms · 278 img/s |
| 3 | 16.3 ms · 184 img/s | 10.9 ms · 275 img/s |
| 4 | 21.6 ms · 186 img/s | 13.9 ms · 287 img/s |
| 4 × unbatched (same session) | 24.9 ms total · 161 img/s | 16.9 ms total · 237 img/s |

Batching 4 cameras buys ~15-20 % more aggregate throughput than four unbatched runs; the execution provider is the larger lever (~+55 % for TensorRT fp16 over CUDA fp32). The dynamic-batch export itself costs ~0-8 % per image versus the static batch-1 model. These are saturated-batch numbers - with real cameras the latest-frame queue drops stale frames, so partial batches are cheaper and per-camera latency stays bounded.
