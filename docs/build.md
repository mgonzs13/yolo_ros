# Build

[Installation](../README.md#installation) covers CPU everywhere and CUDA/TensorRT on x64. Use this guide when you need a different ONNX Runtime build or CUDA/TensorRT on aarch64.

## Custom ONNX Runtime

`yolo_onnxruntime_vendor/scripts/build_ort_from_source.sh` builds ONNX Runtime from source on x86_64 or aarch64 and packages it in the flat `lib/` + `include/` layout the vendor expects. Select the execution provider with `--ep` (`cpu`, or `cuda`, which also builds TensorRT), then point the colcon build at the resulting prefix with `-DONNXRUNTIME_ROOT=<prefix>` — it takes precedence over the prebuilt download. The node still selects CPU/CUDA/TensorRT at run time via the `provider` parameter (see [Parameters](../README.md#parameters)).

```shell
# CPU build (x86_64 or aarch64); the prefix defaults to <package>/ort-<version>
yolo_onnxruntime_vendor/scripts/build_ort_from_source.sh 1.20.0 --ep cpu

colcon build --symlink-install --cmake-args \
    -DONNXRUNTIME_ROOT=$PWD/src/yolov8_ros/yolo_onnxruntime_vendor/ort-1.20.0
```

A source build is heavy (tens of minutes and several GB of RAM). It reuses a previous build of the same version/EP/architecture unless you pass `--rebuild`, and it accepts an existing source tree or tarball for offline use.

## Source build knobs

`build_ort_from_source.sh <ort_version> [output_dir]` accepts these flags:

- **`--ep cpu|cuda`** — execution provider to build (default `cpu`); `cuda` also builds TensorRT, matching the node's provider chain.
- **`--cuda-arch NN`** — CUDA architecture for `--ep cuda` (Jetson Xavier `72`, Orin `87`); auto-detected with `nvidia-smi` on x86_64.
- **`--rebuild`** — ignore a previous build and build again.
- **`--dry-run`** — print the assembled `build.sh` command and exit.

Environment knobs (the CLI flags win over `ORT_EP` / `ORT_CUDA_ARCH`):

- **`ORT_SOURCE_DIR`** — an existing ONNX Runtime source tree (contains `build.sh`); skips the clone.
- **`ORT_SOURCE_TARBALL`** — a tarball whose top level contains `onnxruntime/`.
- **`ORT_BUILD_DIR`** — work dir for the source tree and build (default `<package>/ort-<version>-build`).
- **`ORT_PARALLEL`** — concurrent compile jobs for `build.sh` (`--parallel N`); unset = all cores. Lower it if a CUDA build exhausts RAM.
- **`ORT_CMAKE_EXTRA_DEFINES`** — extra space-separated `--cmake_extra_defines`. `--ep cuda` defaults to `onnxruntime_USE_FLASH_ATTENTION=OFF onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF`: the detection/segmentation/pose/OBB pipelines don't use the CUDA attention kernels, and dropping them roughly halves the CUDA provider build. Override to re-enable (e.g. for attention-based models such as YOLOv12).
- **`ORT_OPS_CONFIG`** — reduced-ops config passed to `--include_ops_by_config` (default `<source>/reduced_ops.config`; set it empty for the full kernel set).
- **`ORT_DISABLE_UNUSED_OPS`** (default `1`) / **`ORT_DISABLE_CONTRIB_OPS`** (default `0`) — extra kernel pruning. `ORT_DISABLE_CONTRIB_OPS=1` is **incompatible with `--ep cuda`**: the TensorRT execution provider needs contrib ops, so ORT's configure fails.
- **`ORT_MIN_CMAKE`** — override the CMake version check (ONNX Runtime 1.16+ needs CMake ≥ 3.26).
- **`CUDA_HOME` / `CUDNN_HOME` / `TENSORRT_HOME`** — `--ep cuda` only; each is a prefix with `include/` and `lib/`.

On x86_64 with CUDA + TensorRT the two prefix defaults usually need overriding: `CUDNN_HOME` (the runtime `libcudnn9-cuda-12` package ships no headers — add `libcudnn9-dev-cuda-12`) and `TENSORRT_HOME` (TensorRT lives under the CUDA toolkit, e.g. `/usr/local/cuda-12.6/targets/x86_64-linux`). CMake ≥ 3.26 must be on `PATH`:

```shell
ORT_SOURCE_DIR=<onnxruntime-source> \
CUDNN_HOME=<prefix-with-cudnn-headers> \
TENSORRT_HOME=/usr/local/cuda-12.6/targets/x86_64-linux \
ORT_CUDA_ARCH=86 \
    yolo_onnxruntime_vendor/scripts/build_ort_from_source.sh 1.20.2 --ep cuda
```

The fixed `build.sh` recipe is `--config Release --update --build --build_shared_lib --skip_tests --skip_submodule_sync --parallel`; the source tree and ORT build dir are reused across runs, so an interrupted build resumes incrementally.

## CUDA / TensorRT on aarch64 (Jetson, offline build)

GPU inference on an aarch64 Jetson (Xavier, Orin) needs an ONNX Runtime built with CUDA + TensorRT — the prebuilt GPU tarball is x64-only. `yolo_onnxruntime_vendor/scripts/` ships two tools for that, each documented in its own file header:

- **`build_ort_from_source.sh`** — runs on the robot and builds ONNX Runtime from source into a flat `lib/` + `include/` prefix: `build_ort_from_source.sh <version> --ep cuda --cuda-arch <NN>` (Xavier `72`, Orin `87`). It reuses a previous build of the same version/EP/architecture unless you pass `--rebuild`.
- **`prepare_offline_bundle.sh`** — runs on an internet-connected host and produces a self-contained tarball for a robot with no network: the ONNX Runtime source tree with its submodules, the mirrored CMake dependency archives, the ONNX models, a bundled CMake (ONNX Runtime 1.16+ needs CMake ≥ 3.26, while JetPack 6 / Ubuntu 22.04 ship 3.22) and both scripts. It prints the exact copy-paste sequence for the robot when it finishes.

Extract the bundle **outside** the colcon workspace — it carries `COLCON_IGNORE` markers so colcon does not treat the ONNX Runtime tree as a package. The robot build also passes `-DFETCHCONTENT_SOURCE_DIR_YOLO_HFHUB=<bundle>/huggingface-hub-cpp` so `yolo_hfhub_vendor` does not fetch `huggingface-hub-cpp` from the network.

Link the built ONNX Runtime into the colcon build with `-DONNXRUNTIME_ROOT` (it overrides the prebuilt download):

```shell
colcon build --symlink-install --cmake-args \
    -DONNXRUNTIME_ROOT=$PWD/src/yolov8_ros/yolo_onnxruntime_vendor/ort-1.20.0 \
    -DFETCHCONTENT_SOURCE_DIR_YOLO_HFHUB=$HOME/offline-bundle-1.20.0/huggingface-hub-cpp
```

`prepare_offline_bundle.sh` prints the exact paths for the bundle it produced.

## Fast C++-only rebuild loop

Rebuild just the package (select `yolo_msgs` first if the messages changed — it must build before `yolo_ros`, and keep `-DONNX_GPU=ON` for a GPU build):

```shell
colcon build --symlink-install --cmake-args -DONNX_GPU=ON --packages-select yolo_msgs
colcon build --symlink-install --cmake-args -DONNX_GPU=ON --packages-select yolo_ros
```
