# Docker (CPU and CUDA / TensorRT)

See the [README](../README.md) for the short project overview. This guide covers the GPU image and the NVIDIA runtime in detail.

Two Dockerfiles are provided:

- `Dockerfile` — CPU image (plain `colcon build`; uses the ROS 2 `onnxruntime_vendor` CPU runtime).
- `Dockerfile.gpu` — GPU image built on `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04`. It expects a flat GPU ONNX Runtime prefix in `./onnxruntime` in the build context and builds `yolo_ros` with `-DONNXRUNTIME_ROOT=/opt/onnxruntime`. Only jammy-based ROS 2 distros (`humble`, `iron`) are supported; overriding `CUDA_VERSION` also requires a matching `TENSORRT_VERSION` build argument.

## CPU image

```shell
docker build -t yolo_ros .
```

## GPU image (CUDA / TensorRT)

The GPU build compiles against a custom GPU ONNX Runtime prefix. Put the prefix in the build context before building:

```shell
cp -a ~/onnxruntime-1.20.0-gpu ./onnxruntime
docker build -f Dockerfile.gpu -t yolo_ros:gpu .
```

The Dockerfile copies it to `/opt/onnxruntime` and configures `yolo_ros` with `-DONNXRUNTIME_ROOT=/opt/onnxruntime`; the TensorRT 10 runtime libraries are installed for the provider at run time.

The GPU image targets CUDA 12 on x86_64 only; CUDA 10 / JetPack 4 is not containerized (see [build.md](build.md#cuda-10-legacy)).

Build it with:

```shell
docker build -f Dockerfile.gpu -t yolo_ros:gpu .
```

Run it with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) passing `--gpus all`. Persist the TensorRT engine cache so engines are not rebuilt on every run:

```shell
docker run -it --rm \
  --gpus all \
  -v "$HOME/models:/models:ro" \
  -v "$HOME/.cache/yolo_ros/container:/root/.cache/yolo_ros" \
  yolo_ros:gpu
```

Note the cache is mounted under `~/.cache/yolo_ros/container`, not `~/.cache/yolo_ros` directly: a TensorRT engine plan file is only readable by the exact TensorRT version that built it. The image pins TensorRT 10.13.2 to match the development host, but if the host ever builds engines with a different version, sharing the cache would make the TensorRT provider fail to deserialize them and fall back to CUDA (a `failed to initialize` warning is logged). A dedicated container cache avoids that entirely (or keep both sides on the same TensorRT version).

`config/yolo.yaml` sets `provider: cuda` (CUDA -> CPU) and the `device` parameter defaults to `cuda:0`. Select a backend with `provider:=tensorrt` (TensorRT -> CUDA -> CPU) or `provider:=cuda`, and check the startup log for the line `Using execution provider: tensorrt` (or `cuda`). If it reports `cpu`, the GPU libraries are not visible to the container — make sure you are using the GPU image and passed `--gpus all`.

### Pinned versions

The image fixes the run-time stack so a build is reproducible and its TensorRT engine cache is portable:

| Component         | Version                                                 |
| :---------------- | :------------------------------------------------------ |
| CUDA (base image) | 12.6.3 (`nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04`) |
| cuDNN             | 9 (from the base image)                                 |
| TensorRT          | 10.13.2 (`libnvinfer10=10.13.2.6-1+cuda12.9`, held)     |
| ONNX Runtime      | 1.20.0 (`onnxruntime-linux-x64-gpu-1.20.0.tgz`)         |
| ROS 2             | Humble (`ros-humble-ros-core`)                          |

TensorRT is pinned to 10.13.2 to match the development host, and the three `libnvinfer*` packages are held with `apt-mark` so the image's `apt upgrade` does not bump them to the newest `+cuda13.x` build. The `+cuda12.9` TensorRT build runs on the CUDA 12.6 base through CUDA minor-version compatibility — the same combination used on the host (CUDA 12.6 + TensorRT 10.13.2).

## NVIDIA runtime notes

The legacy `nvidia-docker` wrapper is deprecated; the supported mechanism is the NVIDIA Container Toolkit. Installing it registers an `nvidia` runtime and installs a hook that injects the host driver and GPU devices. You do **not** need CUDA installed in the image beyond its user-space libraries (the GPU base image above provides them) — the host kernel driver is injected at run time.

Prefer the fine-grained `--gpus all` (or `--gpus '"device=0"'`) flag over `--runtime=nvidia`; both work, but `--gpus` does not require the runtime to be the daemon default:

```shell
# equivalent to the run command above, using the named runtime explicitly.
# The CUDA base image already sets NVIDIA_VISIBLE_DEVICES=all and
# NVIDIA_DRIVER_CAPABILITIES=compute,utility, so no -e flags are needed.
docker run -it --rm --runtime=nvidia yolo_ros:gpu
```

If you want GPU access without passing a flag, set the runtime as the daemon default in `/etc/docker/daemon.json` and restart Docker:

```json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "args": [],
      "path": "nvidia-container-runtime"
    }
  }
}
```

This makes every container on the host use the NVIDIA runtime, so only do it on a dedicated GPU machine. Verify the setup with `docker run --rm --gpus all nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 nvidia-smi`.
