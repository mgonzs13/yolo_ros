# Build

[Installation](../README.md#installation) covers CPU everywhere and CUDA/TensorRT on x64. Use this guide when you need a different ONNX Runtime build or CUDA/TensorRT on aarch64.

## Prebuilt GPU selection

With `-DONNX_GPU=ON` the vendor package detects the CUDA major version — `CUDA_VERSION` in the environment (NVIDIA container images export it), `$CUDA_HOME/version.json`, `/usr/local/cuda*/version.json`, then `nvcc` — and downloads the matching prebuilt:

| CUDA major | ONNX Runtime | Tarball suffix | cuDNN |
| ---------- | ------------ | -------------- | ----- |
| 10 (10.2)  | 1.6.0        | `-gpu`         | 8     |
| 11         | 1.18.0       | `-gpu`         | 8     |
| 12         | 1.20.0       | `-gpu`         | 9     |
| 13         | 1.28.0       | `-gpu_cuda13`  | 9     |

The suffix is not uniform across ONNX Runtime releases (`-gpu`, `-cuda12`, `-gpu-cuda12`, `-gpu_cuda12`, `-gpu_cuda13`), so the table above is curated per CUDA major. CUDA minor-version compatibility is forward-only within a major, and cuDNN 8 and 9 are not interchangeable: install the exact CUDA runtime series and cuDNN major shown.

Overrides (all as `colcon build --cmake-args`):

- **`ONNX_CUDA_MAJOR=10|11|12|13`** — skip detection and pick the row explicitly.
- **`ONNXRUNTIME_VERSION=...`** — override the ONNX Runtime version (keep `ONNX_GPU_SUFFIX` consistent, or pass a full URL).
- **`ONNX_GPU_SUFFIX=...`** — override the tarball suffix for versions outside the table.
- **`ONNXRUNTIME_URL=...`** — full tarball URL; skips version/suffix selection entirely.
- **`ONNXRUNTIME_ROOT=...`** — a locally built ONNX Runtime prefix (takes precedence over any download).

### cuDNN runtime

The ONNX Runtime CUDA execution provider needs the cuDNN **major** matching the selected release at run time — see the table above — from a directory on the loader path: ORT 1.18/1.20 link the versioned `libcudnn.so.8`/`libcudnn.so.9` directly, while ORT 1.28 resolves cuDNN on first use (versioned `libcudnn.so.9` or unversioned `libcudnn.so`). The cuDNN **version** must also support the GPU architecture: cuDNN 9.11+ binaries no longer ship pre-Turing kernels (compute capability < 7.5; their precompiled engines start at sm_75), so a GTX 1060 fails on the first convolution with `CUDNN_STATUS_EXECUTION_FAILED` when a 9.11+ cuDNN is installed. 9.10.x is the newest usable release for those GPUs, and CUDA 13 does not target CC < 7.5 at all.

| CUDA major | GPU                   | apt package (Ubuntu)                                          |
| ---------- | --------------------- | ------------------------------------------------------------- |
| 11         | any                   | `libcudnn8-dev` (cuDNN 8)                                     |
| 12         | pre-Turing (CC < 7.5) | `libcudnn9-{cuda-12,dev-cuda-12,headers-cuda-12}` at `9.10.x` |
| 12         | Turing+ (CC >= 7.5)   | `libcudnn9-dev-cuda-12` (latest 9.x)                          |
| 13         | pre-Turing            | not supported; use CUDA 12 + cuDNN 9.10.x or CUDA 11 + 8      |
| 13         | Turing+ (CC >= 7.5)   | `libcudnn9-dev-cuda-13` (latest 9.x)                          |

The `-dev` packages are the reliable choice: besides the headers they install the unversioned `libcudnn.so` symlink, while the runtime-only packages ship just the versioned `libcudnn.so.9`; `-dev` therefore covers both the direct link of 1.18/1.20 and the runtime lookup of 1.28.

The helper script detects the CUDA major (or takes `--cuda-major`), the GPU compute capability (`--gpu-arch NN` with or without the dot, `$GPU_ARCH`, or `nvidia-smi`, taking the lowest capability across GPUs) and the Ubuntu release (override with `--ubuntu 2004|2204|2404`, e.g. inside containers), installs the matching packages, adds the NVIDIA repository when apt does not know them, pins and `apt-mark hold`s the cuDNN 9.10.x packages for pre-Turing GPUs on CUDA 12, runs `ldconfig` and verifies the result. `--tensorrt` also installs TensorRT (skipped on pre-Turing GPUs, see below) and `--dry-run` previews without touching the system. When no GPU is visible (build host, container) it installs the latest cuDNN and warns; pass `--gpu-arch` when provisioning for a known GPU.

```shell
sudo yolo_onnxruntime_vendor/scripts/install_gpu_deps.sh
sudo yolo_onnxruntime_vendor/scripts/install_gpu_deps.sh --cuda-major 12
sudo yolo_onnxruntime_vendor/scripts/install_gpu_deps.sh --cuda-major 12 --gpu-arch 61
sudo yolo_onnxruntime_vendor/scripts/install_gpu_deps.sh --tensorrt
```

Equivalent manual installs:

```shell
# CUDA 11 (Ubuntu 20.04 / 22.04: ORT 1.18.0, cuDNN 8)
sudo apt-get install libcudnn8-dev
# CUDA 12 (Ubuntu 20.04 / 22.04 / 24.04: ORT 1.20.0, cuDNN 9)
sudo apt-get install libcudnn9-dev-cuda-12
# CUDA 12 on a pre-Turing GPU (Pascal/Volta, CC < 7.5): pin the newest compatible
# cuDNN (9.10.2.21-1 is what Ubuntu 22.04 serves; check yours with
# `apt-cache madison libcudnn9-cuda-12 | grep 9.10`; the helper script resolves it).
# --allow-downgrades is required when a newer cuDNN (e.g. 9.27) is already installed
sudo apt-get install --allow-downgrades libcudnn9-cuda-12=9.10.2.21-1 \
                     libcudnn9-dev-cuda-12=9.10.2.21-1 \
                     libcudnn9-headers-cuda-12=9.10.2.21-1
sudo apt-mark hold libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-headers-cuda-12
# Undo the hold with: sudo apt-mark unhold libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-headers-cuda-12
# CUDA 13 (Ubuntu 22.04 / 24.04: ORT 1.28.0, cuDNN 9; Turing+ only)
sudo apt-get install libcudnn9-dev-cuda-13
```

For the manual path, if `apt` does not know `libcudnn9-*` (for example a machine set up from a local `cuda-repo-*-local` archive), add the NVIDIA network repository first, using the URL for the Ubuntu release (`ubuntu2004`, `ubuntu2204`, `ubuntu2404`):

```shell
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install libcudnn9-dev-cuda-13
sudo ldconfig
ldconfig -p | grep cudnn   # libcudnn.so.9 and libcudnn.so
```

Without apt, extract the cuDNN 9 tarball for the right CUDA major, register it and add the symlink if the archive lacks it:

```shell
sudo sh -c 'echo /opt/cudnn/lib > /etc/ld.so.conf.d/cudnn.conf'
sudo ldconfig
sudo ln -sf /opt/cudnn/lib/libcudnn.so.9 /opt/cudnn/lib/libcudnn.so
```

Symptoms of a missing/mismatched cuDNN: the node log falls back to `Using execution provider: cpu`, or the CUDA provider fails on the first convolution with
`cuDNN is unavailable or disabled for CUDA Execution Provider: dlopen failed for libcudnn.so`.
A cuDNN that no longer supports the GPU architecture fails differently: loading succeeds and the first `Conv` node fails with `CUDNN_STATUS_EXECUTION_FAILED` (e.g. cuDNN 9.11+ on a GTX 1060), which is fixed by pinning 9.10.x as above.
TensorRT is unrelated to this error (it is only used by `provider: tensorrt`).

### CUDA 10 (legacy)

CUDA 10.2 maps to ONNX Runtime 1.6.0, the last upstream release with CUDA 10.2
support. x86_64 uses the prebuilt `onnxruntime-linux-x64-gpu-1.6.0.tgz`; on
aarch64 (JetPack 4) build it from source:

```shell
yolo_onnxruntime_vendor/scripts/build_ort_from_source.sh 1.6.0 --ep cuda --cuda-arch 72
# Nano: 53, TX2: 62, Xavier: 72
```

ONNX Runtime 1.7-1.11 are CUDA 11 builds, so keep `ONNXRUNTIME_VERSION=1.6.0`
for CUDA 10: overriding it with a newer release on a CUDA 10 host produces a
library that cannot load.

`install_gpu_deps.sh` does not cover CUDA 10 — JetPack provides the CUDA/cuDNN 8
userland.

The 1.6.0 GPU tarball hard-links the CUDA 10.2 / cuDNN 8 / cuBLAS 10 userland,
so it can only be linked and run on a real CUDA 10.2 host. To compile-test the
legacy code path on a modern host, configure a clean build directory with
`-DONNXRUNTIME_VERSION=1.6.0 -DONNX_GPU=OFF`: with `ONNX_GPU=ON` cached or
copy-pasted, the vendor still selects the CUDA 10.2-linked GPU tarball and the
link fails.

The legacy path has hard limits: only the CUDA execution provider is available
(TensorRT and its fp16/engine-cache options are not), CUDA Graph is disabled
(`cuda_graph_enable` is ignored with a warning), cuDNN 8 is required, and models
must be exported with `opset <= 13` (the mirror's `opset=12` exports are the
tested combination). ROS 2 Galactic (Ubuntu 20.04) is the supported distribution
for this stack; JetPack 4 ships Ubuntu 18.04 by default, so a community 20.04
rootfs (or equivalent) is required to run Galactic.

Manual check on a CUDA 10.2 device: set `provider: cuda`, confirm the startup
log prints `Using execution provider: cuda`, and run `yolo.launch.py` end-to-end
with an opset-12 model.

### TensorRT 10 runtime

The ONNX Runtime TensorRT EP is built against TensorRT 10 for all the releases the vendor selects (`libnvinfer.so.10` / `libnvonnxparser.so.10`):

| ORT              | TensorRT |
| ---------------- | -------- |
| 1.18.0 (CUDA 11) | 10.0     |
| 1.20.0 (CUDA 12) | 10.4     |
| 1.28.0 (CUDA 13) | 10.x     |

Install the versioned runtime packages, pinning the build tagged for the CUDA major when several are offered (`apt-cache madison libnvinfer10`); the helper script does this automatically:

```shell
# CUDA 13 example; substitute cuda12/cuda11 for the other majors
sudo apt-get install libnvinfer10=10.16.1.11-1+cuda13.2 \
                     libnvonnxparsers10=10.16.1.11-1+cuda13.2
sudo ldconfig
ldconfig -p | grep libnvinfer.so.10
```

TensorRT 10 requires compute capability 7.5+ (Turing or newer); the helper script skips it on older GPUs, where the CUDA provider is the GPU option.

Or use the script: `sudo yolo_onnxruntime_vendor/scripts/install_gpu_deps.sh --tensorrt`. Do **not** install the `tensorrt` meta package: it may point at a newer major (TensorRT 11), which does not satisfy the `libnvinfer.so.10` soname. A missing runtime logs `Failed to load library .../libonnxruntime_providers_tensorrt.so ... libnvinfer.so.10: cannot open shared object file` and `provider: tensorrt` falls back to CUDA.

CUDA 10 is the legacy path (ONNX Runtime 1.6.0, CUDA EP only) and keeps its x86_64 prebuilt tarball; CUDA 11 has no C++ GPU tarball after ONNX Runtime 1.18, CUDA 13 starts at 1.28, and the prebuilt GPU tarballs are x64-only. For aarch64 (JetPack), a custom ONNX Runtime, or offline installs, use the source build below.

## Custom ONNX Runtime

`yolo_onnxruntime_vendor/scripts/build_ort_from_source.sh` builds ONNX Runtime from source on x86_64 or aarch64 and packages it in the flat `lib/` + `include/` layout the vendor expects. Select the execution provider with `--ep` (`cpu`, or `cuda`, which also builds TensorRT for ONNX Runtime >= 1.12; older releases build CUDA only), then point the colcon build at the resulting prefix with `-DONNXRUNTIME_ROOT=<prefix>` — it takes precedence over the prebuilt download. The node still selects CPU/CUDA/TensorRT at run time via the `provider` parameter (see [Parameters](../README.md#parameters)).

```shell
# CPU build (x86_64 or aarch64); the prefix defaults to <package>/ort-<version>
yolo_onnxruntime_vendor/scripts/build_ort_from_source.sh 1.20.0 --ep cpu

colcon build --symlink-install --cmake-args \
    -DONNXRUNTIME_ROOT=$PWD/src/yolo_ros/yolo_onnxruntime_vendor/ort-1.20.0
```

A source build is heavy (tens of minutes and several GB of RAM). It reuses a previous build of the same version/EP/architecture (the reuse stamp also covers the CUDA architecture and extra CMake defines) unless you pass `--rebuild`, and it accepts an existing source tree or tarball for offline use.

## Source build knobs

`build_ort_from_source.sh <ort_version> [output_dir]` accepts these flags:

- **`--ep cpu|cuda`** — execution provider to build (default `cpu`); `cuda` also builds TensorRT for ONNX Runtime >= 1.12 (older releases build CUDA only), matching the node's provider chain.
- **`--cuda-arch NN`** — CUDA architecture(s) for `--ep cuda` (Jetson Xavier `72`, Orin `87`; semicolon-separated for several, e.g. `"72;87"` (quote it)); auto-detected with `nvidia-smi` when available.
- **`--rebuild`** — ignore a previous build and build again.
- **`--dry-run`** — print the assembled `build.sh` command and exit.

Environment knobs (the CLI flags win over `ORT_EP` / `ORT_CUDA_ARCH`):

- **`ORT_SOURCE_DIR`** — an existing ONNX Runtime source tree (contains `build.sh`); skips the clone.
- **`ORT_SOURCE_TARBALL`** — a tarball whose top level contains `onnxruntime/`.
- **`ORT_BUILD_DIR`** — work dir for the source tree and build (default `<package>/ort-<version>-build`).
- **`ORT_PARALLEL`** — concurrent compile jobs for `build.sh` (`--parallel N`); unset = all cores. Lower it if a CUDA build exhausts RAM.
- **`ORT_CMAKE_EXTRA_DEFINES`** — extra space-separated `--cmake_extra_defines`. `--ep cuda` on ONNX Runtime >= 1.11 defaults to `onnxruntime_USE_FLASH_ATTENTION=OFF onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF`: the detection/segmentation/pose/OBB pipelines don't use the CUDA attention kernels, and dropping them roughly halves the CUDA provider build. Override to re-enable (e.g. for attention-based models such as YOLOv12).
- **`ORT_OPS_CONFIG`** — reduced-ops config passed to `--include_ops_by_config` (default `<source>/reduced_ops.config`; set it empty for the full kernel set).
- **`ORT_DISABLE_UNUSED_OPS`** (default `1`) / **`ORT_DISABLE_CONTRIB_OPS`** (default `0`) — extra kernel pruning. `ORT_DISABLE_CONTRIB_OPS=1` is **incompatible with `--ep cuda`**: the TensorRT execution provider needs contrib ops, so the script refuses the combination up front.
- **`ORT_MIN_CMAKE`** — override the CMake version check (ONNX Runtime 1.16+ needs CMake ≥ 3.26; `--ep cuda` on ORT < 1.16 needs ≥ 3.18).
- **`CUDA_HOME` / `CUDNN_HOME` / `TENSORRT_HOME`** — `--ep cuda` only. `CUDA_HOME` is a toolkit prefix (`include/` + `lib/`); the `CUDNN_HOME`/`TENSORRT_HOME` defaults resolve to the system library directories (`/usr/lib/<arch>-linux-gnu`), which carry no headers — set them explicitly when ORT's configure needs the development files. `TENSORRT_HOME` is only used for ONNX Runtime >= 1.12.

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

- **`build_ort_from_source.sh`** — runs on the robot and builds ONNX Runtime from source into a flat `lib/` + `include/` prefix: `build_ort_from_source.sh <version> --ep cuda --cuda-arch <NN>` (Xavier `72`, Orin `87`). For CUDA 10 / JetPack 4 use `build_ort_from_source.sh 1.6.0 --ep cuda --cuda-arch 53|62|72` (Nano/TX2/Xavier): that legacy build yields a CUDA-EP-only ONNX Runtime. It reuses a previous build of the same version/EP/architecture (the reuse stamp also covers the CUDA architecture and extra CMake defines) unless you pass `--rebuild`.
- **`prepare_offline_bundle.sh`** — runs on an internet-connected host and produces a self-contained tarball for a robot with no network: the ONNX Runtime source tree with its submodules, the mirrored CMake dependency archives, the ONNX models, a bundled CMake (ONNX Runtime 1.16+ needs CMake ≥ 3.26, while JetPack 6 / Ubuntu 22.04 ship 3.22) and all the helper scripts (the CMake dependency mirror is only used by ONNX Runtime 1.16+; older releases skip it). It prints the exact copy-paste sequence for the robot when it finishes.

Extract the bundle **outside** the colcon workspace — it carries `COLCON_IGNORE` markers so colcon does not treat the ONNX Runtime tree as a package. The robot build also passes `-DFETCHCONTENT_SOURCE_DIR_YOLO_HFHUB=<bundle>/huggingface-hub-cpp` so `yolo_hfhub_vendor` does not fetch `huggingface-hub-cpp` from the network.

Link the built ONNX Runtime into the colcon build with `-DONNXRUNTIME_ROOT` (it overrides the prebuilt download):

```shell
colcon build --symlink-install --cmake-args \
    -DONNXRUNTIME_ROOT=$PWD/src/yolo_ros/yolo_onnxruntime_vendor/ort-1.20.0 \
    -DFETCHCONTENT_SOURCE_DIR_YOLO_HFHUB=$HOME/offline-bundle-1.20.0/huggingface-hub-cpp
```

`prepare_offline_bundle.sh` prints the exact paths for the bundle it produced.

## Fast C++-only rebuild loop

Rebuild just the package (select `yolo_msgs` first if the messages changed — it must build before `yolo_ros`, and keep `-DONNX_GPU=ON` for a GPU build):

```shell
colcon build --symlink-install --cmake-args -DONNX_GPU=ON --packages-select yolo_msgs
colcon build --symlink-install --cmake-args -DONNX_GPU=ON --packages-select yolo_ros
```
