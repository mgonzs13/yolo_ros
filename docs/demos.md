# Demos

## Object Detection

Standard behavior including ByteTrack (or BoT-SORT) object tracking.

```shell
ros2 launch yolo_bringup yolo.launch.py
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1gTQt6soSIq1g2QmK7locHDiZ-8MqVl2w)](https://drive.google.com/file/d/1gTQt6soSIq1g2QmK7locHDiZ-8MqVl2w/view?usp=sharing)

## Instance Segmentation

Instance masks are the borders of the detected objects, not all the pixels inside the masks.

```shell
ros2 launch yolo_bringup yolo.launch.py model_filename:=yolo26l-seg.onnx
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1dwArjDLSNkuOGIB0nSzZR6ABIOCJhAFq)](https://drive.google.com/file/d/1dwArjDLSNkuOGIB0nSzZR6ABIOCJhAFq/view?usp=sharing)

## Human Pose

Visible persons are detected along with their skeleton keypoints.

```shell
ros2 launch yolo_bringup yolo.launch.py model_filename:=yolo26m-pose.onnx
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1pRy9lLSXiFEVFpcbesMCzmTMEoUXGWgr)](https://drive.google.com/file/d/1pRy9lLSXiFEVFpcbesMCzmTMEoUXGWgr/view?usp=sharing)

## Oriented Bounding Box

Rotated boxes are estimated for oriented objects.

```shell
ros2 launch yolo_bringup yolo.launch.py model_filename:=yolo26m-obb.onnx
```

<!-- Demo recording pending: add ./docs/media/demo_obb.gif to display it here.
<p align="center">
  <img src="./docs/media/demo_obb.gif" alt="Oriented bounding box demo" width="100%" />
</p>
-->

## Image Classification

Image-level ImageNet-1k labels are published as detections with an empty bbox, so the `yolo_classify.yaml` preset chains only detection + debug (no tracker to run).

```shell
ros2 launch yolo_bringup yolo.launch.py params_file:=$(ros2 pkg prefix yolo_bringup)/share/yolo_bringup/config/yolo_classify.yaml
```

<!-- Demo recording pending: add ./docs/media/demo_classify.gif to display it here.
<p align="center">
  <img src="./docs/media/demo_classify.gif" alt="Image classification demo" width="100%" />
</p>
-->

## 3D Object Detection

The 3D bounding boxes are calculated by filtering the depth image data from an RGB-D camera using the 2D bounding box. The default `config/yolo.yaml` chains `detection3d` over a camera with depth, so `<cam>/detections_3d` is published as soon as the camera streams depth.

```shell
ros2 launch yolo_bringup yolo.launch.py
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1ZcN_u9RB9_JKq37mdtpzXx3b44tlU-pr)](https://drive.google.com/file/d/1ZcN_u9RB9_JKq37mdtpzXx3b44tlU-pr/view?usp=sharing)

## 3D Object Detection (Using Instance Segmentation Masks)

The depth image data is filtered using the instance mask polygon. Only objects with a 3D bounding box are visualized in the 2D image (the preset keeps the default `detection3d` chain over a depth camera).

```shell
ros2 launch yolo_bringup yolo.launch.py model_filename:=yolo26l-seg.onnx
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1wVZgi5GLkAYxv3GmTxX5z-vB8RQdwqLP)](https://drive.google.com/file/d/1wVZgi5GLkAYxv3GmTxX5z-vB8RQdwqLP/view?usp=sharing)

## 3D Human Pose

Each keypoint is back-projected to 3D using the depth image. Only objects with a 3D bounding box are visualized in the 2D image (the preset keeps the default `detection3d` chain over a depth camera).

```shell
ros2 launch yolo_bringup yolo.launch.py model_filename:=yolo26m-pose.onnx
```

[![](https://drive.google.com/thumbnail?authuser=0&sz=w1280&id=1j4VjCAsOCx_mtM2KFPOLkpJogM0t227r)](https://drive.google.com/file/d/1j4VjCAsOCx_mtM2KFPOLkpJogM0t227r/view?usp=sharing)
