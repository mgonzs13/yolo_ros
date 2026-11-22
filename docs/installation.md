# Installation

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

## CPU (x64 and aarch64)

```shell
colcon build --symlink-install
source install/setup.bash
```

`yolo_ros` links the ROS 2 `onnxruntime_vendor` CPU runtime and the node runs on the CPU execution provider — no CUDA or cuDNN required.

## Custom ONNX Runtime (CUDA, TensorRT, QNN)

By default `yolo_ros` uses the CPU ONNX Runtime from the released ROS 2 `onnxruntime_vendor` package. For CUDA/TensorRT/QNN, build or obtain an ONNX Runtime prefix and pass it to CMake with `-DONNXRUNTIME_ROOT`:

```shell
# CUDA + TensorRT on x86_64/Jetson
src/yolo_ros/scripts/build_ort_from_source.sh 1.28.0 --ep cuda --cuda-arch 87

# QNN on the Ventuno Q
src/yolo_ros/scripts/build_ort_from_source.sh 1.28.0 --ep qnn --qnn-home /usr

# Build yolo_ros against the resulting prefix (default: src/yolo_ros/ort-1.28.0)
colcon build --symlink-install --packages-up-to yolo_ros \
    --cmake-args -DONNXRUNTIME_ROOT=$PWD/src/yolo_ros/ort-1.28.0
```

The prefix must contain `include/onnxruntime_cxx_api.h` and either `lib/libonnxruntime.so` or `lib64/libonnxruntime.so`. Select the provider with `provider: cuda`, `provider: tensorrt` or `provider: qnn`.

The CUDA/cuDNN/TensorRT versions must match the ONNX Runtime build. See the [build guide](build.md) for the version tables, source-build knobs and troubleshooting.

## QNN (Hexagon NPU, VENTUNO Q)

The QNN execution provider is **not in the default ONNX Runtime**, so it always needs a custom source build. On the VENTUNO Q: install the board's QAIRT packages, build ONNX Runtime with `--ep qnn` and pass the prefix to CMake — details, version notes and troubleshooting are in the [QNN build guide](build.md#qnn-hexagon-npu-onnx-runtime-build):

```shell
sudo apt install qairt-libs qairt-dsp-binaries qairt-headers

src/yolo_ros/scripts/build_ort_from_source.sh 1.28.0 --ep qnn --qnn-home /usr

colcon build --symlink-install --packages-up-to yolo_ros \
    --cmake-args -DONNXRUNTIME_ROOT=$PWD/src/yolo_ros/ort-1.28.0

ros2 launch yolo_bringup yolo.launch.py \
    params_file:=$(ros2 pkg prefix yolo_bringup)/share/yolo_bringup/config/yolo_qnn.yaml
```

Then export a QNN model with Ultralytics (`model.export(format="qnn", name="iq-8275")`) and launch the Hexagon NPU preset (`config/yolo_qnn.yaml`) — see the [QNN model guide](models.md#qualcomm-qnn-hexagon-npu).

Launch from the workspace root so relative source paths resolve.
