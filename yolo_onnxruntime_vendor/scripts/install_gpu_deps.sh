#!/usr/bin/env bash
# Copyright (c) 2026 Miguel Ángel González Santamarta
# SPDX-License-Identifier: MIT
#
# Install the CUDA runtime libraries (cuDNN, optional TensorRT 10) required by
# the ONNX Runtime GPU build that yolo_onnxruntime_vendor downloads.
#
# The CUDA major selects ONNX Runtime and the cuDNN major (see docs/build.md):
#   CUDA 11 -> ONNX Runtime 1.18.0 + cuDNN 8
#   CUDA 12 -> ONNX Runtime 1.20.0 + cuDNN 9
#   CUDA 13 -> ONNX Runtime 1.28.0 + cuDNN 9
#
# cuDNN 9.11+ binaries no longer ship pre-Turing kernels (compute capability
# < 7.5): the precompiled engines start at sm_75 and fail with
# CUDNN_STATUS_EXECUTION_FAILED on a GTX 1060. On CUDA 12 this script pins the
# newest 9.10.x for such GPUs and holds the packages; CUDA 13 cannot target
# them at all. TensorRT 10 requires 7.5+ and is skipped on older GPUs.
#
# Usage:
#   sudo scripts/install_gpu_deps.sh [--cuda-major 11|12|13] [--gpu-arch NN]
#                                    [--tensorrt] [--ubuntu 2004|2204|2404]
#                                    [--dry-run]
#
# The CUDA major is auto-detected from CUDA_VERSION, $CUDA_HOME/version.json,
# /usr/local/cuda*/version.json, then nvcc; the GPU architecture from
# --gpu-arch, $GPU_ARCH, then nvidia-smi (minimum across all GPUs). The Ubuntu
# release is read from /etc/os-release unless --ubuntu is passed (useful
# inside containers).

set -euo pipefail

usage() {
  cat <<'EOF'
Install the CUDA runtime libraries required by the ONNX Runtime GPU build.

Usage:
  sudo scripts/install_gpu_deps.sh [--cuda-major 11|12|13] [--gpu-arch NN]
                                   [--tensorrt] [--ubuntu 2004|2204|2404]
                                   [--dry-run]

Options:
  --cuda-major M   CUDA major to install for (auto-detected when omitted)
  --gpu-arch NN    GPU compute capability, with or without the dot
                   (61, 6.1, 75, 86); auto-detected with nvidia-smi
                   when omitted
  --tensorrt       also install the TensorRT 10 runtime for provider: tensorrt
                   (requires compute capability 7.5+)
  --ubuntu R       override the Ubuntu release detected from /etc/os-release
  --dry-run        print what would be done without touching the system
  -h, --help       show this help
EOF
}

CUDA_MAJOR=""
GPU_ARCH="${GPU_ARCH:-}"
UBUNTU=""
WITH_TENSORRT=0
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --cuda-major)
      if [ -z "${2:-}" ]; then
        echo "error: --cuda-major needs a value (11, 12 or 13)" >&2
        usage >&2
        exit 2
      fi
      CUDA_MAJOR="$2"
      shift 2
      ;;
    --gpu-arch)
      if [ -z "${2:-}" ]; then
        echo "error: --gpu-arch needs a value (e.g. 61, 6.1, 75, 86)" >&2
        usage >&2
        exit 2
      fi
      GPU_ARCH="$2"
      shift 2
      ;;
    --ubuntu)
      if [ -z "${2:-}" ]; then
        echo "error: --ubuntu needs a value (2004, 2204 or 2404)" >&2
        usage >&2
        exit 2
      fi
      UBUNTU="$2"
      shift 2
      ;;
    --tensorrt)
      WITH_TENSORRT=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# First number of the "version" field in a CUDA version.json file. Returns 0
# with empty output when the file is missing or unparseable so a bad candidate
# cannot abort the detection under `set -e`.
cuda_major_from_json() {
  grep -oE '"version"[[:space:]]*:[[:space:]]*"[0-9]+' "$1" 2>/dev/null |
    grep -oE '[0-9]+$' | head -1 || true
}

detect_cuda_major() {
  if [ -n "${CUDA_VERSION:-}" ]; then
    echo "${CUDA_VERSION%%.*}"
    return 0
  fi
  if [ -n "${CUDA_HOME:-}" ] && [ -f "${CUDA_HOME}/version.json" ]; then
    local value
    value="$(cuda_major_from_json "${CUDA_HOME}/version.json")"
    if [ -n "$value" ]; then
      echo "$value"
      return 0
    fi
  fi
  local json value
  for json in /usr/local/cuda*/version.json; do
    [ -f "$json" ] || continue
    value="$(cuda_major_from_json "$json")"
    if [ -n "$value" ]; then
      echo "$value"
      return 0
    fi
  done
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | grep -oE 'release [0-9]+' | grep -oE '[0-9]+' |
      head -1
    return 0
  fi
  return 0
}

# Normalize a compute capability to digits only: "6.1" and "61" -> "61".
normalize_gpu_arch() {
  local arch="${1//./}"
  if [[ ! "$arch" =~ ^[0-9]{2,3}$ ]]; then
    return 1
  fi
  echo "$arch"
}

# Compute capability of the slowest GPU on the host, digits only; empty when
# no GPU is visible. Honours --gpu-arch / GPU_ARCH when set.
detect_gpu_arch() {
  if [ -n "${GPU_ARCH:-}" ]; then
    normalize_gpu_arch "$GPU_ARCH"
    return $?
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local caps cap min=""
    caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)"
    while IFS= read -r cap; do
      cap="$(normalize_gpu_arch "$cap" 2>/dev/null)" || continue
      if [ -z "$min" ] || [ "$cap" -lt "$min" ]; then
        min="$cap"
      fi
    done <<< "${caps}"
    if [ -n "$min" ]; then
      echo "$min"
      return 0
    fi
  fi
  return 1
}

# Newest cuDNN 9.10.x available in apt. 9.11+ precompiled engines ship no
# pre-Turing (sm_61) kernels, so 9.10.x is the newest usable release for them.
resolve_pre_turing_cudnn_version() {
  apt-cache madison libcudnn9-cuda-12 2>/dev/null |
    awk -F'|' \
      '$2 ~ /^[[:space:]]*9[.]10[.]/ && !found { gsub(/[[:space:]]/, "", $2); print $2; found=1 }' ||
    true
}

detect_ubuntu() {
  local version_id=""
  if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    version_id="${VERSION_ID:-}"
  fi
  case "$version_id" in
    20.04) echo "2004" ;;
    22.04) echo "2204" ;;
    24.04) echo "2404" ;;
    *) echo "" ;;
  esac
}

if [ -n "$GPU_ARCH" ]; then
  if ! GPU_ARCH="$(normalize_gpu_arch "$GPU_ARCH")"; then
    echo "error: invalid --gpu-arch value (expected a compute capability like 61 or 6.1)" >&2
    exit 2
  fi
fi

if [ -z "$CUDA_MAJOR" ]; then
  CUDA_MAJOR="$(detect_cuda_major)"
fi
if [ -z "$UBUNTU" ]; then
  UBUNTU="$(detect_ubuntu)"
fi

if [ -z "$CUDA_MAJOR" ]; then
  echo "Could not detect the CUDA major; pass --cuda-major 11|12|13." >&2
  exit 1
fi
if [ -z "$UBUNTU" ]; then
  echo "Could not detect the Ubuntu release; pass --ubuntu 2004|2204|2404." >&2
  exit 1
fi
case "$UBUNTU" in
  2004 | 2204 | 2404) ;;
  *)
    echo "Unsupported Ubuntu release '${UBUNTU}' (supported: 2004, 2204, 2404)." >&2
    exit 1
    ;;
esac

if [ -z "$GPU_ARCH" ]; then
  if GPU_ARCH="$(detect_gpu_arch)"; then
    :
  else
    GPU_ARCH=""
  fi
fi

PRE_TURING=0
if [ -n "$GPU_ARCH" ] && [ "$GPU_ARCH" -lt 75 ]; then
  PRE_TURING=1
fi

CUDNN_PKGS=()
CUDNN_PIN=0

case "${CUDA_MAJOR}:${UBUNTU}" in
  11:2004 | 11:2204)
    CUDNN_PKGS=("libcudnn8-dev")
    ;;
  12:2004 | 12:2204 | 12:2404)
    if [ "$PRE_TURING" -eq 1 ]; then
      CUDNN_PKGS=("libcudnn9-cuda-12" "libcudnn9-dev-cuda-12" "libcudnn9-headers-cuda-12")
      CUDNN_PIN=1
    else
      CUDNN_PKGS=("libcudnn9-dev-cuda-12")
    fi
    ;;
  13:2204 | 13:2404)
    if [ "$PRE_TURING" -eq 1 ]; then
      echo "error: CUDA 13 dropped pre-Turing GPUs (compute capability < 7.5)." >&2
      echo "Use CUDA 12 (cuDNN 9.10.x) or CUDA 11 (cuDNN 8) instead; see docs/build.md." >&2
      exit 1
    fi
    CUDNN_PKGS=("libcudnn9-dev-cuda-13")
    ;;
  11:2404)
    echo "CUDA 11 has no NVIDIA repo package on Ubuntu 24.04. Use a source-built ONNX Runtime (scripts/build_ort_from_source.sh) instead." >&2
    exit 1
    ;;
  13:2004)
    echo "CUDA 13 has no NVIDIA repo package on Ubuntu 20.04. Use Ubuntu 22.04/24.04 or a source-built ONNX Runtime instead." >&2
    exit 1
    ;;
  *)
    echo "Unsupported CUDA major '${CUDA_MAJOR}' (supported: 11, 12, 13)." >&2
    exit 1
    ;;
esac

case "$CUDA_MAJOR" in
  11) ORT_VERSION="1.18.0" ;;
  12) ORT_VERSION="1.20.0" ;;
  13) ORT_VERSION="1.28.0" ;;
esac

# TensorRT runtime for the ONNX Runtime TensorRT EP. The supported ONNX Runtime
# builds load the TensorRT 10 soname (libnvinfer.so.10 / libnvonnxparser.so.10);
# the "tensorrt" meta package may point at a newer major, so use the versioned
# packages and prefer the build tagged for the detected CUDA major.
# TensorRT build newest first that matches the CUDA major. Like
# resolve_pre_turing_cudnn_version, the awk never exits early so pipefail does not
# turn a successful lookup into SIGPIPE status 141.
trt_pinned_version() {
  apt-cache madison libnvinfer10 2>/dev/null |
    awk -v pat="+cuda${CUDA_MAJOR}." 'index($3, pat) && !found { print $3; found=1 }' ||
    true
}

if [ "$WITH_TENSORRT" -eq 1 ] && [ "$PRE_TURING" -eq 1 ]; then
  echo "warning: TensorRT 10 requires compute capability 7.5+; skipping TensorRT for GPU cc ${GPU_ARCH}." >&2
  WITH_TENSORRT=0
fi

TAG="ubuntu${UBUNTU}"
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${TAG}/x86_64/cuda-keyring_1.1-1_all.deb"

if [ -n "$GPU_ARCH" ]; then
  gpu_desc="GPU cc ${GPU_ARCH}"
  if [ "$PRE_TURING" -eq 1 ]; then
    gpu_desc="${gpu_desc} (pre-Turing)"
  fi
else
  gpu_desc="GPU not detected"
fi

if [ "$CUDNN_PIN" -eq 1 ]; then
  cudnn_desc="$(IFS=,; echo "${CUDNN_PKGS[*]}") pinned to 9.10.x"
else
  cudnn_desc="${CUDNN_PKGS[0]}"
fi

echo "CUDA ${CUDA_MAJOR} on ${TAG}: ${cudnn_desc} (ONNX Runtime ${ORT_VERSION})"
echo "  ${gpu_desc}"
if [ -z "$GPU_ARCH" ]; then
  echo "warning: no GPU detected; installing the latest cuDNN. Pass --gpu-arch NN when targeting a known GPU." >&2
fi
if [ "$WITH_TENSORRT" -eq 1 ]; then
  echo "TensorRT 10 runtime requested (libnvinfer10, libnvonnxparsers10)"
  echo "warning: do not install the 'tensorrt' meta package (now TensorRT 11); it does not provide libnvinfer.so.10" >&2
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "--- dry run:"
  echo "  install NVIDIA cuda-keyring for ${TAG} if ${CUDNN_PKGS[0]} is unknown"
  if [ "$CUDNN_PIN" -eq 1 ]; then
    cudnn_pin_version="$(resolve_pre_turing_cudnn_version)"
    if [ -n "$cudnn_pin_version" ]; then
      cudnn_specs=()
      for pkg in "${CUDNN_PKGS[@]}"; do
        cudnn_specs+=("${pkg}=${cudnn_pin_version}")
      done
      echo "  apt-get install -y --allow-downgrades --allow-change-held-packages ${cudnn_specs[*]}"
    else
      echo "  apt-get install -y libcudnn9-{cuda-12,dev-cuda-12,headers-cuda-12}=9.10.* (newest available, resolved after apt-get update)"
    fi
    echo "  apt-mark hold ${CUDNN_PKGS[*]}"
  else
    echo "  apt-get install -y ${CUDNN_PKGS[*]}"
  fi
  if [ "$WITH_TENSORRT" -eq 1 ]; then
    trt_version="$(trt_pinned_version)"
    if [ -n "$trt_version" ]; then
      echo "  apt-get install -y libnvinfer10=${trt_version} libnvonnxparsers10=${trt_version}"
    else
      echo "  warning: no libnvinfer10 build tagged +cuda${CUDA_MAJOR}. in apt; would install the unversioned packages (may pull a different CUDA runtime)"
    fi
  fi
  echo "  ldconfig && ldconfig -p | grep cudnn"
  exit 0
fi

if [ "$(id -u)" -ne 0 ]; then
  echo "Root is required; re-running with sudo ..."
  reexec=(--cuda-major "$CUDA_MAJOR" --ubuntu "$UBUNTU")
  if [ -n "$GPU_ARCH" ]; then
    reexec+=(--gpu-arch "$GPU_ARCH")
  fi
  if [ "$WITH_TENSORRT" -eq 1 ]; then
    reexec+=(--tensorrt)
  fi
  exec sudo "$0" "${reexec[@]}"
fi

need_keyring=0
if ! apt-cache show "${CUDNN_PKGS[0]}" >/dev/null 2>&1; then
  need_keyring=1
fi
if [ "$WITH_TENSORRT" -eq 1 ] &&
  ! apt-cache show libnvinfer10 >/dev/null 2>&1; then
  need_keyring=1
fi

# Download with whichever HTTP client is available.
fetch_url() {
  local url="$1" dest="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$dest" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$dest" "$url"
  else
    echo "error: neither curl nor wget is available to download ${url}" >&2
    return 1
  fi
}

if [ "$need_keyring" -eq 1 ]; then
  echo "Adding the NVIDIA CUDA repository for ${TAG} ..."
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "${tmp_dir}"' EXIT
  fetch_url "$KEYRING_URL" "${tmp_dir}/cuda-keyring.deb"
  dpkg -i "${tmp_dir}/cuda-keyring.deb"
  apt-get update
fi

export DEBIAN_FRONTEND=noninteractive
if [ "$CUDNN_PIN" -eq 1 ]; then
  CUDNN_PIN_VERSION="$(resolve_pre_turing_cudnn_version)"
  if [ -z "$CUDNN_PIN_VERSION" ]; then
    echo "error: no cuDNN 9.10.x found in apt for CUDA 12." >&2
    echo "Run 'sudo apt-get update' and retry, install it manually, or rebuild ONNX Runtime for CUDA 11 (cuDNN 8); see docs/build.md." >&2
    exit 1
  fi
  cudnn_specs=()
  for pkg in "${CUDNN_PKGS[@]}"; do
    cudnn_specs+=("${pkg}=${CUDNN_PIN_VERSION}")
  done
  echo "Pinning pre-Turing cuDNN: ${cudnn_specs[*]}"
  apt-get install -y --allow-downgrades --allow-change-held-packages "${cudnn_specs[@]}"
  apt-mark hold "${CUDNN_PKGS[@]}"
else
  apt-get install -y "${CUDNN_PKGS[@]}"
fi
if [ "$WITH_TENSORRT" -eq 1 ]; then
  trt_version="$(trt_pinned_version)"
  if [ -n "$trt_version" ]; then
    apt-get install -y "libnvinfer10=${trt_version}" \
      "libnvonnxparsers10=${trt_version}" || {
      echo "warning: pinned TensorRT install failed; retrying with the unversioned packages" >&2
      apt-get install -y libnvinfer10 libnvonnxparsers10
    }
  else
    echo "warning: no libnvinfer10 build tagged +cuda${CUDA_MAJOR}. in apt; installing the unversioned packages (may pull a different CUDA runtime)" >&2
    apt-get install -y libnvinfer10 libnvonnxparsers10
  fi
fi
ldconfig

echo "--- installed:"
dpkg-query -W -f='${Package} ${Version}\n' "${CUDNN_PKGS[@]}" || true
if [ "$CUDNN_PIN" -eq 1 ]; then
  echo "Held ${CUDNN_PKGS[*]} to keep pre-Turing support; unhold with:"
  echo "  sudo apt-mark unhold ${CUDNN_PKGS[*]}"
fi
if [ "$WITH_TENSORRT" -eq 1 ]; then
  dpkg-query -W -f='${Package} ${Version}\n' libnvinfer10 \
    libnvonnxparsers10 || true
fi

echo "--- loader check:"
if ! ldconfig -p | grep -i cudnn; then
  echo "cuDNN is not visible to the dynamic loader" >&2
  exit 1
fi
if [ "$WITH_TENSORRT" -eq 1 ]; then
  if ! ldconfig -p | grep -i "libnvinfer.so.10"; then
    echo "TensorRT is not visible to the dynamic loader" >&2
    exit 1
  fi
fi
