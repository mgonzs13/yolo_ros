#!/usr/bin/env bash
# Copyright (c) 2026 Miguel Ángel González Santamarta
# SPDX-License-Identifier: MIT
#
# Install the CUDA runtime libraries (cuDNN, optional TensorRT) required by the
# ONNX Runtime GPU build that yolo_onnxruntime_vendor downloads.
#
# The CUDA major selects ONNX Runtime and cuDNN (see docs/build.md):
#   CUDA 11 -> ONNX Runtime 1.18.0 + cuDNN 8
#   CUDA 12 -> ONNX Runtime 1.20.0 + cuDNN 9
#   CUDA 13 -> ONNX Runtime 1.28.0 + cuDNN 9
#
# Usage:
#   sudo scripts/install_gpu_deps.sh [--cuda-major 11|12|13] [--tensorrt]
#                                    [--ubuntu 2004|2204|2404] [--dry-run]
#
# The CUDA major is auto-detected from CUDA_VERSION, $CUDA_HOME/version.json,
# /usr/local/cuda*/version.json, then nvcc. The Ubuntu release is read from
# /etc/os-release unless --ubuntu is passed (useful inside containers).

set -euo pipefail

usage() {
  cat <<'EOF'
Install the CUDA runtime libraries required by the ONNX Runtime GPU build.

Usage:
  sudo scripts/install_gpu_deps.sh [--cuda-major 11|12|13] [--tensorrt]
                                   [--ubuntu 2004|2204|2404] [--dry-run]

Options:
  --cuda-major M   CUDA major to install for (auto-detected when omitted)
  --tensorrt       also install the TensorRT runtime (provider: tensorrt)
  --ubuntu R       override the Ubuntu release detected from /etc/os-release
  --dry-run        print what would be done without touching the system
  -h, --help       show this help
EOF
}

CUDA_MAJOR=""
UBUNTU=""
WITH_TENSORRT=0
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --cuda-major)
      CUDA_MAJOR="${2:-}"
      shift 2
      ;;
    --ubuntu)
      UBUNTU="${2:-}"
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

# First number of the "version" field in a CUDA version.json file.
cuda_major_from_json() {
  grep -oE '"version"[[:space:]]*:[[:space:]]*"[0-9]+' "$1" 2>/dev/null |
    grep -oE '[0-9]+$' | head -1
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

case "${CUDA_MAJOR}:${UBUNTU}" in
  11:2004 | 11:2204) CUDNN_PKG="libcudnn8-dev" ;;
  12:2004 | 12:2204 | 12:2404) CUDNN_PKG="libcudnn9-dev-cuda-12" ;;
  13:2204 | 13:2404) CUDNN_PKG="libcudnn9-dev-cuda-13" ;;
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

TAG="ubuntu${UBUNTU}"
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${TAG}/x86_64/cuda-keyring_1.1-1_all.deb"

echo "CUDA ${CUDA_MAJOR} on ${TAG}: ${CUDNN_PKG} (ONNX Runtime ${ORT_VERSION})"
if [ "$WITH_TENSORRT" -eq 1 ]; then
  echo "TensorRT runtime requested"
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "--- dry run:"
  echo "  install NVIDIA cuda-keyring for ${TAG} if ${CUDNN_PKG} is unknown"
  echo "  apt-get install -y ${CUDNN_PKG}"
  if [ "$WITH_TENSORRT" -eq 1 ]; then
    echo "  apt-get install -y tensorrt"
  fi
  echo "  ldconfig && ldconfig -p | grep cudnn"
  exit 0
fi

if [ "$(id -u)" -ne 0 ]; then
  echo "Root is required; re-running with sudo ..."
  reexec=(--cuda-major "$CUDA_MAJOR" --ubuntu "$UBUNTU")
  if [ "$WITH_TENSORRT" -eq 1 ]; then
    reexec+=(--tensorrt)
  fi
  exec sudo "$0" "${reexec[@]}"
fi

if ! apt-cache show "$CUDNN_PKG" >/dev/null 2>&1; then
  echo "Adding the NVIDIA CUDA repository for ${TAG} ..."
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "${tmp_dir}"' EXIT
  curl -fsSL -o "${tmp_dir}/cuda-keyring.deb" "$KEYRING_URL"
  dpkg -i "${tmp_dir}/cuda-keyring.deb"
  apt-get update
fi

export DEBIAN_FRONTEND=noninteractive
apt-get install -y "$CUDNN_PKG"
if [ "$WITH_TENSORRT" -eq 1 ]; then
  apt-get install -y tensorrt
fi
ldconfig

echo "--- installed:"
dpkg-query -W -f='${Package} ${Version}\n' "$CUDNN_PKG" || true
if [ "$WITH_TENSORRT" -eq 1 ]; then
  dpkg-query -W -f='${Package} ${Version}\n' tensorrt || true
fi

echo "--- loader check:"
if ! ldconfig -p | grep -i cudnn; then
  echo "cuDNN is not visible to the dynamic loader" >&2
  exit 1
fi
