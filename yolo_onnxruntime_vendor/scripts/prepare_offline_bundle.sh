#!/usr/bin/env bash
# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT
#
# Run on an INTERNET-CONNECTED host to produce a self-contained bundle for
# building ONNX Runtime <ort_version> (aarch64, CUDA + TensorRT) on the robot.
#
# The bundle contains:
#   - onnxruntime/     the v<ort_version> source tree with all git submodules
#                      (git metadata stripped)
#   - onnxruntime/mirror/  pre-downloaded CMake dependency archives (>= 1.16)
#   - models/          the opset-12 ONNX models for the pipeline
#
# Usage: scripts/prepare_offline_bundle.sh <ort_version> [output_dir]
#   ort_version  ONNX Runtime release, e.g. 1.6.0 or 1.20.2
#   output_dir   defaults to <package>/offline-bundle-<ort_version>; the
#                tarball is <output_dir>.tar.gz
#
# For versions that resolve dependencies through CMake (>= 1.16, driven by
# cmake/deps.txt) the script also mirrors every archive into
# <onnxruntime>/mirror/, which ONNX Runtime prefers over the network, so those
# versions build offline too.
#
# Env overrides:
#   MODELS_DIR  directory holding the .onnx files (default ~/models)
#   MODELS      space-separated model file names to include
#   MIRROR_SKIP space-separated cmake/deps.txt names to not mirror (defaults to
#               platform-specific archives not needed for a native aarch64 build)
set -euo pipefail

usage() {
  echo "usage: $(basename "$0") <ort_version> [output_dir]" >&2
}

if [[ $# -lt 1 || -z "${1:-}" ]]; then
  echo "error: ONNX Runtime version is required" >&2
  usage
  exit 2
fi

ORT_VERSION="${1#v}"
if [[ ! "${ORT_VERSION}" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
  echo "error: '${ORT_VERSION}' does not look like a version (e.g. 1.20.2)" >&2
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${2:-${SCRIPT_DIR}/../offline-bundle-${ORT_VERSION}}"
MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
MODELS="${MODELS:-yolo26m.onnx}"
ORT_GIT_URL="https://github.com/microsoft/onnxruntime"
HFHUB_GIT_URL="https://github.com/agonzc34/huggingface-hub-cpp"
HFHUB_TAG="1.1.4"
# Kitware prebuilt aarch64 CMake bundled for the robot (needs only glibc 2.17).
CMAKE_VERSION="${CMAKE_VERSION:-3.26.6}"

# The colcon workspace root (repo is <workspace>/src/yolov8_ros/).
WORKSPACE="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

for tool in curl git tar; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "error: ${tool} not found" >&2
    exit 1
  }
done

echo "==> Cloning ONNX Runtime v${ORT_VERSION} with submodules (shallow)..."
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"
git clone --depth 1 --shallow-submodules --recursive \
  -b "v${ORT_VERSION}" "${ORT_GIT_URL}" "${OUT_DIR}/onnxruntime"

echo "==> Verifying git submodules are populated..."
missing="$(git -C "${OUT_DIR}/onnxruntime" submodule status --recursive \
  | awk '$1 ~ /^-/ {print $2}')"
if [[ -n "${missing}" ]]; then
  echo "error: uninitialized submodules: ${missing}" >&2
  exit 1
fi

# ONNX Runtime >= 1.16 resolves most dependencies from cmake/deps.txt through
# CMake FetchContent at configure time. Before hitting the network it looks for
# a pre-downloaded copy under <onnxruntime>/mirror/<url-without-https://>, so
# mirroring each archive here lets the robot build fully offline.
DEPS_FILE="${OUT_DIR}/onnxruntime/cmake/deps.txt"
if [[ -f "${DEPS_FILE}" ]]; then
  MIRROR_SKIP="${MIRROR_SKIP:-protoc_win64 protoc_win32 protoc_mac_universal protoc_linux_x86 protoc_linux_x64 directx_headers composable_kernel dawn coremltools microsoft_wil}"
  drifted=()
  echo "==> Mirroring cmake/deps.txt archives into onnxruntime/mirror/..."
  while IFS=';' read -r name url sha; do
    [[ -z "${name}" || "${name}" == \#* || -z "${url}" ]] && continue
    [[ "${url}" == https://* ]] || continue
    if [[ " ${MIRROR_SKIP} " == *" ${name} "* ]]; then
      echo "    skip ${name} (not needed for a native aarch64 build)"
      continue
    fi
    dest="${OUT_DIR}/onnxruntime/mirror/${url#https://}"
    mkdir -p "$(dirname "${dest}")"
    echo "    fetch ${name}"
    curl -fsSL -o "${dest}" "${url}"
    actual="$(sha1sum "${dest}" | awk '{print tolower($1)}')"
    expected="$(printf '%s' "${sha}" | tr 'A-Z' 'a-z')"
    if [[ "${actual}" != "${expected}" ]]; then
      # GitLab regenerates archives non-deterministically, so a few pinned
      # hashes in cmake/deps.txt no longer match (eigen, kleidiai). The bytes
      # still come from the official URL over HTTPS; re-pin the bundled copy
      # and record the drift. Set STRICT_SHA=1 to abort instead.
      if [[ -n "${STRICT_SHA:-}" ]]; then
        echo "error: SHA1 mismatch for ${name}: expected ${expected}, got ${actual}" >&2
        exit 1
      fi
      echo "warning: SHA1 drift for ${name} (expected ${expected:0:12}...," \
        "got ${actual:0:12}...); re-pinning bundled deps.txt" >&2
      drifted+=("${name}|${expected}|${actual}|${url}")
    fi
  done < "${DEPS_FILE}"

  if [[ ${#drifted[@]} -gt 0 ]]; then
    drift_log="${OUT_DIR}/onnxruntime/mirror/SHA1_DRIFT.txt"
    {
      echo "# Archives whose SHA1 no longer matches cmake/deps.txt because the"
      echo "# server re-zipped them. The bundled deps.txt was re-pinned to the"
      echo "# actual downloaded hashes below (all fetched from the official URLs)."
      echo "#"
      printf '# %-22s %-42s %-42s %s\n' name expected actual url
    } > "${drift_log}"
    for entry in "${drifted[@]}"; do
      IFS='|' read -r name expected actual url <<< "${entry}"
      printf '  %-22s %-42s %-42s %s\n' \
        "${name}" "${expected}" "${actual}" "${url}" >> "${drift_log}"
      sed -i "s|^\(${name};[^;]*;\)[^;]*|\1${actual}|" "${DEPS_FILE}"
    done
    echo "    re-pinned ${#drifted[@]} drifted hash(es); see" \
      "onnxruntime/mirror/SHA1_DRIFT.txt"
  fi
fi

echo "==> Stripping git metadata (offline builds call no git)..."
# git submodule foreach aborts once .git is missing in a nested submodule (e.g.
# emsdk), so remove every .git file/dir directly instead.
while IFS= read -r -d '' gitmeta; do
  rm -rf "${gitmeta}"
done < <(find "${OUT_DIR}/onnxruntime" -name .git -print0)

echo "==> Collecting opset-12 models from ${MODELS_DIR}..."
mkdir -p "${OUT_DIR}/models"
found=0
for model in ${MODELS}; do
  if [[ -f "${MODELS_DIR}/${model}" ]]; then
    cp -v "${MODELS_DIR}/${model}" "${OUT_DIR}/models/"
    found=$((found + 1))
  else
    echo "warning: ${MODELS_DIR}/${model} not found; skipping" >&2
  fi
done
if [[ "${found}" -eq 0 ]]; then
  echo "error: no models found in ${MODELS_DIR}" >&2
  exit 1
fi

# --- Reduced-operator-kernel build config ------------------------------------
# The robot builds ONNX Runtime with only the kernels the bundled models need
# (--include_ops_by_config). Generate that list here: the host has the models,
# the ORT source and internet; the robot has none of the Python tooling. Failure
# is non-fatal -- the robot then does a full kernel build.
REDUCED_OPS_CONFIG="${OUT_DIR}/onnxruntime/reduced_ops.config"
PY_RUN=()
if [[ -n "${ONNX_PY:-}" ]]; then
  # ONNX_PY may be a command with arguments, e.g. "uv run --with onnx python3".
  read -r -a PY_RUN <<< "${ONNX_PY}"
elif command -v python3 >/dev/null 2>&1 && python3 -c 'import onnx' >/dev/null 2>&1; then
  PY_RUN=(python3)
elif command -v uv >/dev/null 2>&1 && uv run --with onnx python3 -c 'import onnx' >/dev/null 2>&1; then
  PY_RUN=(uv run --with onnx python3)
fi

REDUCED_OPS_NOTE=""
if [[ ${#PY_RUN[@]} -eq 0 ]]; then
  echo "warning: no Python with 'onnx' found; skipping the reduced-ops config" >&2
  echo "         set ONNX_PY or install onnx to shrink the onnxruntime build" >&2
  REDUCED_OPS_NOTE="has NO reduced-ops config (the robot build is a full build)"
elif "${PY_RUN[@]}" \
       "${OUT_DIR}/onnxruntime/tools/python/create_reduced_build_config.py" \
       -f ONNX "${OUT_DIR}/models" "${REDUCED_OPS_CONFIG}" \
     && [[ -s "${REDUCED_OPS_CONFIG}" ]] \
     && grep -q '^ai\.onnx;' "${REDUCED_OPS_CONFIG}"; then
  op_count="$(awk -F';' '/^[^#]/{n=split($3,a,","); c+=n} END{print c+0}' "${REDUCED_OPS_CONFIG}")"
  echo "    reduced-ops config: ${REDUCED_OPS_CONFIG} (${op_count} ops)"
  REDUCED_OPS_NOTE="ships onnxruntime/reduced_ops.config (picked up automatically)"
else
  echo "warning: reduced-ops config generation failed; continuing without it" >&2
  rm -f "${REDUCED_OPS_CONFIG}"
  REDUCED_OPS_NOTE="has NO reduced-ops config (the robot build is a full build)"
fi

# ONNX Runtime 1.16+ requires CMake >= 3.26, but JetPack 6 / Ubuntu 22.04 ship
# 3.22 and the robot is offline. Bundle Kitware's prebuilt aarch64 CMake so the
# robot can put it on PATH.
CMAKE_DIRNAME="cmake-${CMAKE_VERSION}-linux-aarch64"
echo "==> Bundling CMake ${CMAKE_VERSION} (aarch64)..."
mkdir -p "${OUT_DIR}/tools"
CMAKE_TGZ="${OUT_DIR}/tools/${CMAKE_DIRNAME}.tar.gz"
curl -fsSL -o "${CMAKE_TGZ}" \
  "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_DIRNAME}.tar.gz"
CMAKE_SHA_FILE="$(mktemp)"
if curl -fsSL -o "${CMAKE_SHA_FILE}" \
  "https://cmake.org/files/v${CMAKE_VERSION%.*}/cmake-${CMAKE_VERSION}-SHA-256.txt"; then
  expected="$(grep " ${CMAKE_DIRNAME}.tar.gz\$" "${CMAKE_SHA_FILE}" | awk '{print $1}')"
  actual="$(sha256sum "${CMAKE_TGZ}" | awk '{print $1}')"
  if [[ -z "${expected}" || "${expected}" != "${actual}" ]]; then
    echo "error: CMake tarball sha256 mismatch (expected '${expected}', got '${actual}')" >&2
    rm -f "${CMAKE_SHA_FILE}" "${CMAKE_TGZ}"
    exit 1
  fi
  echo "    sha256 verified (${actual:0:12}...)"
else
  echo "warning: could not fetch the cmake.org checksum; recording local sha256" >&2
  sha256sum "${CMAKE_TGZ}" >> "${OUT_DIR}/tools/SHA256SUMS"
fi
rm -f "${CMAKE_SHA_FILE}"
tar -xzf "${CMAKE_TGZ}" -C "${OUT_DIR}/tools"
rm -f "${CMAKE_TGZ}"

# Carry the robot-side scripts inside the bundle too, so the bundle does not
# depend on the robot's workspace checkout having the latest version.
echo "==> Bundling vendor scripts..."
mkdir -p "${OUT_DIR}/tools/scripts"
cp -v "${SCRIPT_DIR}"/*.sh "${OUT_DIR}/tools/scripts/"

# yolo_hfhub_vendor pulls huggingface-hub-cpp with CMake FetchContent at
# configure time, which needs network. Bundle the source and let the robot build
# pass -DFETCHCONTENT_SOURCE_DIR_YOLO_HFHUB=<this dir>.
echo "==> Bundling huggingface-hub-cpp (${HFHUB_TAG})..."
git clone --depth 1 -b "${HFHUB_TAG}" "${HFHUB_GIT_URL}" \
  "${OUT_DIR}/huggingface-hub-cpp"
rm -rf "${OUT_DIR}/huggingface-hub-cpp/.git"

# The ONNX Runtime source tree and the fetched library contain setup.py /
# CMakeLists.txt, which colcon would otherwise treat as workspace packages if the
# bundle is extracted inside a colcon workspace. Mark them so colcon skips them.
echo "==> Marking the bundle with COLCON_IGNORE..."
touch "${OUT_DIR}/COLCON_IGNORE" \
  "${OUT_DIR}/onnxruntime/COLCON_IGNORE" \
  "${OUT_DIR}/huggingface-hub-cpp/COLCON_IGNORE"

BUNDLE="${OUT_DIR}.tar.gz"
echo "==> Creating ${BUNDLE}..."
tar -C "$(dirname "${OUT_DIR}")" -czf "${BUNDLE}" "$(basename "${OUT_DIR}")"

SIZE="$(du -h "${BUNDLE}" | cut -f1)"
BUNDLE_NAME="$(basename "${OUT_DIR}")"
cat <<EOF

Bundle ready: ${BUNDLE} (${SIZE})
This bundle ${REDUCED_OPS_NOTE}.

1. Copy the workspace source and the bundle to the robot (replace <user>@<robot>):
     rsync -a --exclude build --exclude install --exclude log \\
         "${WORKSPACE}/" <user>@<robot>:~/yr_ws/
     scp "${BUNDLE}" <user>@<robot>:~/

2. On the robot (offline). Extract the bundle OUTSIDE the colcon workspace:
     tar xzf ~/$(basename "${BUNDLE}") -C ~       # -> ~/${BUNDLE_NAME}
     cd ~/yr_ws
     source /opt/ros/<distro>/setup.bash
     # ONNX Runtime 1.16+ needs CMake >= 3.26 (JetPack 6 / Ubuntu 22.04 ship 3.22).
     # Use the CMake bundled in this archive; python3-numpy is also needed.
     export PATH=~/${BUNDLE_NAME}/tools/${CMAKE_DIRNAME}/bin:\$PATH
     # The build script ships inside the bundle; write the ONNX Runtime prefix
     # into the workspace vendor package so the colcon arg below finds it.
     ORT_SOURCE_DIR=~/${BUNDLE_NAME}/onnxruntime \\
         ~/${BUNDLE_NAME}/tools/scripts/build_ort_from_source.sh ${ORT_VERSION} \\
         "\$HOME/yr_ws/src/yolov8_ros/yolo_onnxruntime_vendor/ort-${ORT_VERSION}" \\
         --ep cuda --cuda-arch 87
     # --base-paths src + the bundle's COLCON_IGNORE keep colcon away from the
     # ONNX Runtime source tree. The ORT source tree's mirror/ makes its CMake
     # dependency fetches local, so no FETCHCONTENT_FULLY_DISCONNECTED here (it
     # would skip extracting those local archives).
     colcon build --symlink-install --base-paths src --cmake-args \\
         -DONNXRUNTIME_ROOT=\$(pwd)/src/yolov8_ros/yolo_onnxruntime_vendor/ort-${ORT_VERSION} \\
         -DFETCHCONTENT_SOURCE_DIR_YOLO_HFHUB=\$HOME/${BUNDLE_NAME}/huggingface-hub-cpp
     source install/setup.bash
     ros2 launch yolo_bringup yolo.launch.py \\
         model:=\$HOME/${BUNDLE_NAME}/models/yolo26m.onnx
EOF
