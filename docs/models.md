# Model export

See the [README](../README.md#models) for the model list and the licensing note. This guide covers exporting a checkpoint and downloading models from the Hugging Face Hub.

The C++ pipeline runs any Ultralytics-exported **ONNX** model whose output matches one of the YOLO layouts below. The compatible model families are:

- [YOLOv3](https://docs.ultralytics.com/models/yolov3/) (`yolov3u`, Ultralytics' updated anchor-free head)
- [YOLOv5](https://docs.ultralytics.com/models/yolov5/) (`yolov5u`)
- [YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- [YOLOv9](https://docs.ultralytics.com/models/yolov9/)
- [YOLOv10](https://docs.ultralytics.com/models/yolov10/)
- [YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [YOLO12](https://docs.ultralytics.com/models/yolo12/)
- [YOLO26](https://docs.ultralytics.com/models/yolo26/)

These families are verified end-to-end with this C++ node. Export `yolov3u`/`yolov5u`, not `yolov3`/`yolov5` — Ultralytics only ships the updated heads for those generations. YOLOv4, YOLOv6 and YOLOv7 appear in the Ultralytics docs but have no downloadable weights, so they cannot be exported; YOLO-World and YOLOE depend on open-vocabulary text prompts, which this C++ pipeline does not implement.

Models are exported for one task — **detection** (`detect`), **instance segmentation** (`segment`), **human pose** (`pose`), **oriented bounding box** (`obb`) or **image classification** (`classify`) — depending on the tasks the family's checkpoint provides (not every family offers every task). Export a `.pt` checkpoint to ONNX with the `ultralytics` package — either with `uv` (no environment needed) or a plain `pip install`:

```shell
# uv one-liner (fetches ultralytics + ONNX deps on the fly)
uv run --with ultralytics --with onnx --with onnxruntime --with onnxslim \
  yolo export model=yolo26m.pt format=onnx imgsz=640 opset=12

# classic: pip install ultralytics, then
yolo export model=yolo26m.pt format=onnx imgsz=640 opset=12
```

Notes:

- The exported `.onnx` is self-contained: Ultralytics writes the class vocabulary into the ONNX graph metadata (`names` key), which `Model::load_class_names()` reads at startup. The `coco.names` fallback is only used when a model has no metadata. The engine also honors an `input_color` metadata value (`rgb`/`bgr`) when the graph carries one; otherwise the input is treated as RGB (what Ultralytics exports expect).
- Keep the exported file outside the repository and pass `model_path:=<path>` on every launch.
- `model_type` selects the pipeline (`YOLO`/`Detect`, `Segment`, `Pose`, `OBB`, `Classify`, case-insensitive). `auto` (the default) falls back to a filename heuristic: a path containing `segment`/`-seg`/`_seg` selects segmentation (Ultralytics names its exports `*-seg.onnx`), `pose` selects pose, `obb` selects OBB, `cls`/`classify` selects classification, otherwise detection.
- The input size is fixed by the ONNX tensor (logged at startup), so the Python-only knobs `imgsz_height` / `imgsz_width` and `half` / `augment` / `agnostic_nms` / `retina_masks` were removed from the C++ node and its configs: the pipeline runs FP32 with no test-time augmentation, and NMS is either baked into the graph (end-to-end exports) or applied by the C++ postprocessor with `iou`.
- OBB models have no baked-NMS export (rotated NMS cannot be exported into the graph), so the C++ postprocessor performs its own per-class rotated NMS and `iou` re-tunes it. The rotation angle is published in `BoundingBox2D.center.theta` (radians) with `size` holding the rotated `w`/`h`.
- Classification exports bake the softmax into the graph (`output0` is `[1, N]` probabilities), so the postprocessor does **not** re-apply it. Top-`top_k` classes are published as detections with an **empty** bbox (image-level labels have no spatial extent).

## Dynamic-batch export (multi-camera pipelines)

The [`unileon-robotics/YOLO26-ONNX`](https://huggingface.co/unileon-robotics/YOLO26-ONNX)
mirror already ships this set under `dynamic/` —
`dynamic/yolo26{size}{-task}.onnx` for every size (`n`/`s`/`m`/`l`/`x`) and
task (`""`/`-seg`/`-pose`/`-obb`/`-cls`), exported batch-only dynamic. Point
`model_filename` at one and skip the export:

```yaml
/yolo/yolo_node:
  ros__parameters:
    cameras: ["cam0"]
    cam0:
      rgb_topic: /camera/rgb/image_raw
    detection:
      plugin: yolo_ros/DetectionPlugin
      cameras: ["cam0"]
      model_repo: unileon-robotics/YOLO26-ONNX
      model_filename: dynamic/yolo26m.onnx
```

To export your own instead, Ultralytics' ONNX exporter is batch-1 by default.
For a multi-camera pipeline (a `DetectionPlugin` whose `cameras` list selects
several cameras), re-export with `dynamic=True`:

```bash
cp yolo26n.pt /tmp/yolo26n-dyn.pt
uv run --with ultralytics --with onnx --with onnxruntime --with onnxslim \
  yolo export model=/tmp/yolo26n-dyn.pt format=onnx dynamic=True
mv /tmp/yolo26n-dyn.onnx ~/models/
```

`dynamic=True` makes the input `['batch', 3, 'height', 'width']` — the batch
**and** H/W axes. `yolo_ros` pins 640×640 for dynamic graphs (all shipped
models train at that size). If you need a batch-only-dynamic graph (so
`input_image_shape` is read from the file), zero the H/W dims first — this is
exactly the form the mirror's `dynamic/` files ship:

```python
import onnx
m = onnx.load("/tmp/yolo26n-dyn.onnx")
for inp in m.graph.input:
    for d in (2, 3):
        inp.type.tensor_type.shape.dim[d].ClearField("dim_param")
        inp.type.tensor_type.shape.dim[d].dim_value = 640
onnx.save(m, "/tmp/yolo26n-batch.onnx")
```

## Download a model from the Hugging Face Hub

Instead of a local path, the node can fetch the model from the Hub at startup via the `yolo_hfhub_vendor` package. Set `model_repo` + `model_filename` in the matching `config/yolo*.yaml` section (or on the command line); the `model_path` is then ignored. The file is cached under `~/.cache/huggingface/hub` by default (override with the optional `cache_dir` param) and reused unless `force_download: true`:

```yaml
/yolo/yolo_node:
  ros__parameters:
    cameras: ["cam0"]
    cam0:
      rgb_topic: /camera/rgb/image_raw
    detection:
      plugin: yolo_ros/DetectionPlugin
      cameras: ["cam0"]
      model_repo: unileon-robotics/YOLO26-ONNX # HF repo id
      model_filename: yolo26s.onnx # file inside that repo
      force_download: false
```

Or on the command line: `ros2 launch yolo_bringup yolo.launch.py model_repo:=unileon-robotics/YOLO26-ONNX model_filename:=yolo26s.onnx`. The Hub client and its libcurl dependency are pulled in by `yolo_hfhub_vendor` (installed with the `rosdep install` step in the [README](../README.md#installation)); this path is only used when `model_repo`/`model_filename` are set. The node logs the model source as `[huggingface]` (with repo/filename) or `[local]` (with the path) so the two are easy to tell apart.

The shipped `config/yolo*.yaml` files already default to the [`unileon-robotics/YOLO26-ONNX`](https://huggingface.co/unileon-robotics/YOLO26-ONNX) mirror — ONNX exports of the [`Ultralytics/YOLO26`](https://huggingface.co/Ultralytics/YOLO26) checkpoints (25 files: `yolo26{n,s,m,l,x}` for detect / `-seg` / `-pose` / `-obb` / `-cls`, exported with `imgsz=640 opset=12`). The same 25 models are also available batch-only dynamic under `dynamic/`, used by multi-camera pipelines whose detector `cameras` list selects several cameras. The first launch per model downloads it (so it needs network access) and caches it; clear `model_repo` to fall back to the local `model_path`, or pass `model_path:=<path>` for a local file.

## ReID encoder for BoT-SORT-ReID

The BoT-SORT-ReID tracker (`tracking.tracker_type: botsort` with
`tracking.with_reid: true`) needs a second ONNX model
that turns a person crop into an appearance embedding. The C++ encoder
(`engine::ReIDEncoder`) is model-agnostic; it only requires this contract:

- **Input:** `[N, 3, H, W]`, RGB, float32 in **[0, 255]** (no `/255`, no
  mean/std in C++).
- **Output:** one embedding per box (`[N, D]` or `[N, D, 1, 1]`).
- Normalization and L2 normalization should be **baked into the graph**; the
  C++ side only resizes crops to `HxW`, converts BGR→RGB, and re-normalizes.

Input `HxW` is read from the graph, so any size works. The embedding dimension
is read from the **runtime** output tensor, so a symbolic output dim is
tolerated; the input `HxW` must still be static.

### OSNet (default suggestion, MIT)

```python
import torch
from torchreid.models import build_model
from torchreid.utils import load_pretrained_weights

class Export(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, x):  # x: RGB float32 in [0, 255]
        # No F.normalize tail: some exporters turn it into a symbolic output
        # dim; the C++ side L2-normalizes the embeddings anyway.
        return self.net((x - self.mean) / self.std)

net = build_model("osnet_x0_25", num_classes=1000, pretrained=False)
load_pretrained_weights(net, "<msmt17_combineall osnet_x0_25 .pth>")
model = Export(net).eval()
torch.onnx.export(
    model, torch.zeros(1, 3, 256, 128), "osnet_x0_25_reid.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}, opset_version=12)
```

Weights: `kaiyangzhou/osnet` on the Hugging Face Hub (MIT). Then set
`tracking.reid_model` in the pipeline YAML.

### FastReID SBS-S50 (Apache-2.0 code, MIT weights via BoT-SORT)

FastReID's `preprocess_image` already normalizes RGB `[0, 255]` input
(`PIXEL_MEAN`/`PIXEL_STD` are scaled by 255), matching the C++ contract. Use the
fork vendored by [BoT-SORT](https://github.com/NirAharon/BoT-SORT) (`fast_reid/`)
— it ships the MOT17 config, which the upstream JDAI repo does not — with the
`mot17_sbs_S50.pth` weights. Export at the model's own `INPUT.SIZE_TEST`:
**384×128** for MOT17 sbs_S50 (not 256×128):

```python
import sys; sys.path.insert(0, "<BoT-SORT>")  # so `fast_reid.fastreid` imports
import torch, torch.nn.functional as F
from fast_reid.fastreid.config import get_cfg
from fast_reid.fastreid.modeling.meta_arch import build_model
from fast_reid.fastreid.utils.checkpoint import Checkpointer

cfg = get_cfg()
cfg.merge_from_file("<BoT-SORT>/fast_reid/configs/MOT17/sbs_S50.yml")
cfg.MODEL.WEIGHTS = "mot17_sbs_S50.pth"
cfg.MODEL.BACKBONE.PRETRAIN = False  # don't fetch the ImageNet backbone
cfg.MODEL.DEVICE = "cpu"
cfg.freeze()
net = build_model(cfg)
Checkpointer(net).load(cfg.MODEL.WEIGHTS)
net.eval()

class Export(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):  # x: RGB float32 in [0, 255]
        out = self.net(x)
        if isinstance(out, dict):  # some forks return {"features": ...}
            out = out["features"]
        return F.normalize(out)

h, w = cfg.INPUT.SIZE_TEST
torch.onnx.export(
    Export(net), torch.zeros(1, 3, h, w), "sbs_S50_reid.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=12)
```

The `dynamic_axes` entry is **required**: a static batch-1 export makes
`session.Run` throw as soon as 2+ boxes are batched, which the tracker catches
and silently degrades to motion-only. Use the weights BoT-SORT releases
(`mot17_sbs_S50.pth`, MIT), not FastReID's own model zoo (no explicit weight
license).
