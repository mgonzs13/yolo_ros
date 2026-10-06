#!/usr/bin/env bash
# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT
#
# Build ONNX Runtime (any version) from source, for the CPU or CUDA (+TensorRT)
# execution provider, and package it in the flat layout consumed by
# yolo_onnxruntime_vendor (point that package at the result with
# -DONNXRUNTIME_ROOT=<output_dir>).
#
# Runs on x86_64 and aarch64. It can work fully offline when given a source tree
# prepared by scripts/prepare_offline_bundle.sh (see that script for the
# host-side workflow).
#
# Usage: scripts/build_ort_from_source.sh <ort_version> [output_dir] [options]
#   ort_version   ONNX Runtime release to build, e.g. 1.6.0 or 1.20.2
#                 (the git tag is v<ort_version>).
#   output_dir    defaults to <package>/ort-<ort_version>
#
# Options:
#   --ep <cpu|cuda>   execution provider to build (default: cpu). "cuda" also
#                     builds the TensorRT provider, matching the node's
#                     TensorRT -> CUDA -> CPU provider chain.
#   --cuda-arch <NN>  CUDA architecture for --ep cuda (Jetson Xavier 72,
#                     Orin 87). Auto-detected with nvidia-smi on x86_64.
#   --rebuild         ignore any previous build and build again.
#   --dry-run         print the ONNX Runtime build command and exit.
#   -h, --help        show this help.
#
# ORT_EP and ORT_CUDA_ARCH are honoured as fallbacks for --ep / --cuda-arch;
# ORT_CUDA_ARCH is only read when the EP is cuda.
#
# Source acquisition (first match wins):
#   ORT_SOURCE_DIR=<path>     an existing onnxruntime source tree (contains build.sh)
#   ORT_SOURCE_TARBALL=<path> a tarball whose top level contains onnxruntime/
#   ${ORT_BUILD_DIR}/onnxruntime   a previously extracted/cloned tree
#   otherwise                 git clone --recursive (requires internet)
#
# Env overrides: CUDA_HOME, CUDNN_HOME, TENSORRT_HOME, ORT_BUILD_DIR,
#   ORT_PARALLEL (concurrent compile jobs; unset = all cores),
#   ORT_CMAKE_EXTRA_DEFINES (space-separated extra --cmake_extra_defines; for
#   --ep cuda it defaults to "onnxruntime_USE_FLASH_ATTENTION=OFF
#   onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF" since the detection family
#   doesn't use the CUDA attention kernels; override to re-enable),
#   ORT_OPS_CONFIG (default: <source>/reduced_ops.config; empty disables the
#   reduced build), ORT_DISABLE_UNUSED_OPS (default 1), ORT_DISABLE_CONTRIB_OPS
#   (default 0; 1 is incompatible with --ep cuda, whose TensorRT EP needs
#   contrib ops)
set -euo pipefail

usage() {
  echo "usage: $(basename "$0") <ort_version> [output_dir] [--ep cpu|cuda] [--cuda-arch NN] [--rebuild] [--dry-run]" >&2
}

ORIG_ARGS=("$@")
EP="${ORT_EP:-cpu}"
CUDA_ARCH="${ORT_CUDA_ARCH:-}"
REBUILD=0
DRY_RUN=0
positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ep)
      [[ $# -ge 2 ]] || { echo "error: --ep needs a value" >&2; exit 2; }
      EP="$2"; shift 2 ;;
    --cuda-arch)
      [[ $# -ge 2 ]] || { echo "error: --cuda-arch needs a value" >&2; exit 2; }
      CUDA_ARCH="$2"; shift 2 ;;
    --rebuild) REBUILD=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; while [[ $# -gt 0 ]]; do positional+=("$1"); shift; done ;;
    -*) echo "error: unknown option '$1'" >&2; usage; exit 2 ;;
    *) positional+=("$1"); shift ;;
  esac
done

ORT_VERSION="${positional[0]:-}"
if [[ -z "${ORT_VERSION}" ]]; then
  echo "error: ONNX Runtime version is required" >&2
  usage
  exit 2
fi
ORT_VERSION="${ORT_VERSION#v}"
if [[ ! "${ORT_VERSION}" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
  echo "error: '${ORT_VERSION}' does not look like a version (e.g. 1.20.2)" >&2
  usage
  exit 2
fi

case "${EP}" in
  cpu|cuda) ;;
  *) echo "error: unsupported --ep '${EP}' (supported: cpu, cuda)" >&2; exit 2 ;;
esac

# CUDA_ARCH is only meaningful for the cuda EP; ignore it otherwise so a stale
# ORT_CUDA_ARCH does not perturb the reuse stamp.
if [[ "${EP}" != "cuda" ]]; then
  CUDA_ARCH=""
fi

# --- Default CMake defines ----------------------------------------------------
# A --ep cuda build disables the CUDA attention kernels by default: the
# detection / segmentation / pose / OBB pipelines don't use them, and skipping
# them roughly halves the CUDA provider build. Override ORT_CMAKE_EXTRA_DEFINES
# to re-enable them (e.g. attention-based models such as YOLOv12).
if [[ "${EP}" == "cuda" && -z "${ORT_CMAKE_EXTRA_DEFINES:-}" ]]; then
  ORT_CMAKE_EXTRA_DEFINES="onnxruntime_USE_FLASH_ATTENTION=OFF onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${positional[1]:-${SCRIPT_DIR}/../ort-${ORT_VERSION}}"
WORK_DIR="${ORT_BUILD_DIR:-${SCRIPT_DIR}/../ort-${ORT_VERSION}-build}"

ARCH="$(uname -m)"
case "${ARCH}" in
  x86_64) DEB_ARCH="x86_64" ;;
  aarch64) DEB_ARCH="aarch64" ;;
  *)
    echo "error: unsupported architecture '${ARCH}' (supported: x86_64, aarch64)" >&2
    exit 1
    ;;
esac

for tool in cmake python3; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "error: ${tool} not found" >&2
    exit 1
  }
done

# --- Reuse a previous build ---------------------------------------------------
# A completed build writes a stamp beside the prefix; when it matches the
# requested version/EP/arch, skip the (multi-minute) rebuild. --rebuild forces it.
STAMP="${OUT_DIR}/.ort-build-info"
if [[ ${REBUILD} -eq 0 && -f "${OUT_DIR}/lib/libonnxruntime.so" && -f "${STAMP}" ]] \
   && grep -qx "version=${ORT_VERSION}" "${STAMP}" \
   && grep -qx "ep=${EP}" "${STAMP}" \
   && grep -qx "arch=${ARCH}" "${STAMP}" \
   && grep -qx "cuda_arch=${CUDA_ARCH}" "${STAMP}" \
   && grep -qx "extra_defines=${ORT_CMAKE_EXTRA_DEFINES:-}" "${STAMP}"; then
  echo "==> Reusing existing ONNX Runtime ${ORT_VERSION} build (${EP}) at ${OUT_DIR}"
  echo "    (pass --rebuild to force a rebuild)"
  exit 0
fi

# --- CUDA toolkit (only for the cuda EP) --------------------------------------
CUDAHOSTCXX=""
if [[ "${EP}" == "cuda" ]]; then
  # JetPack commonly has no /usr/local/cuda symlink; fall back to the newest
  # /usr/local/cuda-<version> when CUDA_HOME is not given.
  if [[ -z "${CUDA_HOME:-}" ]]; then
    if [[ -d /usr/local/cuda ]]; then
      CUDA_HOME=/usr/local/cuda
    else
      CUDA_HOME="$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)"
    fi
  fi
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  CUDNN_HOME="${CUDNN_HOME:-/usr/lib/${DEB_ARCH}-linux-gnu}"
  TRT_HOME="${TENSORRT_HOME:-/usr/lib/${DEB_ARCH}-linux-gnu}"
fi

# ONNX Runtime 1.16+ needs CMake >= 3.26 (JetPack/Ubuntu 22.04 ship 3.22, which
# is too old). Older ONNX Runtime builds accept much older CMake.
version_ge() { [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" == "$1" ]]; }
min_cmake=3.13
if version_ge "${ORT_VERSION}" 1.16; then
  min_cmake=3.26
fi
MIN_CMAKE="${ORT_MIN_CMAKE:-${min_cmake}}"
CMAKE_VER="$(cmake --version | sed -n 's/^cmake version //p')"
if ! version_ge "${CMAKE_VER}" "${MIN_CMAKE}"; then
  echo "error: ONNX Runtime ${ORT_VERSION} needs CMake >= ${MIN_CMAKE}, found ${CMAKE_VER}" >&2
  echo "       If you used the offline bundle, add its CMake to PATH first:" >&2
  echo "         export PATH=~/offline-bundle-*/tools/cmake-*-linux-aarch64/bin:\$PATH" >&2
  echo "       otherwise install a newer CMake (e.g. https://apt.kitware.com/)," >&2
  echo "       or set ORT_MIN_CMAKE to override this check." >&2
  exit 1
fi
echo "==> CMake ${CMAKE_VER} (>= ${MIN_CMAKE} required)"

if [[ "${EP}" == "cuda" ]]; then
  [[ -d "${CUDA_HOME}" ]] || {
    echo "error: CUDA toolkit not found (CUDA_HOME='${CUDA_HOME}')" >&2
    echo "       found: $(ls -d /usr/local/cuda* 2>/dev/null | tr '\n' ' ' || true)" >&2
    echo "       set CUDA_HOME to the toolkit root, e.g. /usr/local/cuda-12.6" >&2
    exit 1
  }

  # --- CUDA host-compiler compatibility ---------------------------------------
  # nvcc only supports host compilers up to a version that depends on the CUDA
  # release (CUDA 10.2 -> gcc <= 8). Distros with a newer default gcc (9+ on
  # Ubuntu 20.04) make nvcc fail with "unsupported GNU version". Auto-select an
  # older g++ when needed; override with CUDAHOSTCXX to skip the detection.
  NVCC="${CUDA_HOME}/bin/nvcc"
  [[ -x "${NVCC}" ]] || {
    echo "error: nvcc not found at ${NVCC}" >&2
    exit 1
  }
  CUDA_MAJOR="$("${NVCC}" --version | sed -n 's/.*release \([0-9]\+\).*/\1/p')"
  case "${CUDA_MAJOR}" in
    9) CUDA_MAX_GCC=7 ;;
    10) CUDA_MAX_GCC=8 ;;
    11) CUDA_MAX_GCC=10 ;;
    *) CUDA_MAX_GCC=99 ;;
  esac
  DEFAULT_GCC_MAJOR="$(g++ -dumpversion | cut -d. -f1)"
  if [[ -z "${CUDAHOSTCXX:-}" && "${DEFAULT_GCC_MAJOR}" -gt "${CUDA_MAX_GCC}" ]]; then
    host_cxx=""
    host_ver=""
    for ver in "${CUDA_MAX_GCC}" "$((CUDA_MAX_GCC - 1))" 7 6; do
      if command -v "g++-${ver}" >/dev/null 2>&1; then
        host_cxx="$(command -v "g++-${ver}")"
        host_ver="${ver}"
        break
      fi
    done
    if [[ -z "${host_cxx}" ]]; then
      echo "error: CUDA ${CUDA_MAJOR} supports host gcc <= ${CUDA_MAX_GCC}, but the" >&2
      echo "       default is gcc ${DEFAULT_GCC_MAJOR}. Install a compatible one, e.g.:" >&2
      echo "         sudo apt install gcc-${CUDA_MAX_GCC} g++-${CUDA_MAX_GCC}" >&2
      echo "       or export CUDAHOSTCXX=/usr/bin/g++-${CUDA_MAX_GCC}" >&2
      exit 1
    fi
    export CXX="${host_cxx}"
    export CUDAHOSTCXX="${host_cxx}"
    if host_cc="$(command -v "gcc-${host_ver}" 2>/dev/null)"; then
      export CC="${host_cc}"
    fi
    echo "==> Default g++ ${DEFAULT_GCC_MAJOR} is too new for CUDA ${CUDA_MAJOR};" \
      "using g++-${host_ver} (${host_cxx})"
  fi
  CUDAHOSTCXX="${CUDAHOSTCXX:-$(command -v g++)}"
  echo "==> CUDA host compiler: ${CUDAHOSTCXX} (CC=${CC:-$(command -v gcc)}, CXX=${CXX:-$(command -v g++)})"

  if [[ -z "${CUDA_ARCH}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
      | head -1 | tr -d '. ' || true)"
  fi
  if [[ -z "${CUDA_ARCH}" ]]; then
    echo "error: --cuda-arch (or ORT_CUDA_ARCH) is required for --ep cuda" >&2
    echo "       (e.g. 72 for Xavier, 87 for Orin); auto-detection failed." >&2
    exit 2
  fi
  echo "==> CUDA architecture: ${CUDA_ARCH}"
fi

mkdir -p "${WORK_DIR}"
SRC_DIR="${WORK_DIR}/onnxruntime"

if [[ -n "${ORT_SOURCE_TARBALL:-}" ]]; then
  echo "==> Extracting source from ${ORT_SOURCE_TARBALL}"
  rm -rf "${SRC_DIR}"
  tar -xf "${ORT_SOURCE_TARBALL}" -C "${WORK_DIR}"
elif [[ -n "${ORT_SOURCE_DIR:-}" ]]; then
  SRC_DIR="${ORT_SOURCE_DIR}"
elif [[ -d "${SRC_DIR}" ]]; then
  echo "==> Reusing existing source tree at ${SRC_DIR}"
else
  command -v git >/dev/null 2>&1 || {
    echo "error: no local ONNX Runtime source and git is unavailable." >&2
    echo "       Run prepare_offline_bundle.sh on an internet host first." >&2
    exit 1
  }
  echo "==> Cloning ONNX Runtime v${ORT_VERSION} (requires internet)"
  # --recursive is mandatory: the GitHub source tarball has no submodules.
  git clone --recursive -b "v${ORT_VERSION}" \
    https://github.com/microsoft/onnxruntime "${SRC_DIR}"
fi

[[ -f "${SRC_DIR}/build.sh" ]] || {
  echo "error: no build.sh under '${SRC_DIR}'. Expected onnxruntime/ at the" >&2
  echo "       source root (e.g. ORT_SOURCE_DIR=<bundle>/onnxruntime)." >&2
  exit 1
}

if [[ "${EP}" == "cuda" ]]; then
  echo "==> Building ONNX Runtime ${ORT_VERSION} (cuda: CUDA ${CUDA_HOME}," \
    "TRT ${TRT_HOME}, arch ${CUDA_ARCH})"
else
  echo "==> Building ONNX Runtime ${ORT_VERSION} (cpu)"
fi
if [[ -d "${SRC_DIR}/build" ]]; then
  echo "    note: ${SRC_DIR}/build already exists; if a previous attempt used a" \
    "different compiler, remove it first (rm -rf ${SRC_DIR}/build)."
fi

# --- Optional build reduction ------------------------------------------------
# Passing an unknown flag aborts build.py, and older releases (e.g. 1.20.0)
# lack --disable_generation_ops, so probe each flag in the source tree first.
supports_build_flag() {
  grep -q -- "$1" "${SRC_DIR}/tools/ci_build/build.py"
}

# Empty ORT_OPS_CONFIG disables the reduced build; unset uses the bundled config.
reduced_ops_config="${ORT_OPS_CONFIG-${SRC_DIR}/reduced_ops.config}"
if [[ -n "${reduced_ops_config}" && ! -f "${reduced_ops_config}" ]]; then
  echo "warning: reduced-ops config '${reduced_ops_config}' not found;" \
    "building without --include_ops_by_config" >&2
  reduced_ops_config=""
fi

extra_build_args=()
skipped_build_args=()
if [[ -n "${reduced_ops_config}" ]]; then
  extra_build_args+=(--include_ops_by_config "${reduced_ops_config}")
fi
if [[ "${ORT_DISABLE_UNUSED_OPS:-1}" == "1" ]]; then
  for flag in --disable_ml_ops --disable_generation_ops; do
    if supports_build_flag "${flag}"; then
      extra_build_args+=("${flag}")
    else
      skipped_build_args+=("${flag}")
    fi
  done
fi
if [[ "${ORT_DISABLE_CONTRIB_OPS:-0}" == "1" ]]; then
  if supports_build_flag --disable_contrib_ops; then
    extra_build_args+=(--disable_contrib_ops)
  else
    skipped_build_args+=(--disable_contrib_ops)
  fi
fi

# --- Assemble the build.sh command -------------------------------------------
build_cmd=(
  ./build.sh --config Release --update --build --build_shared_lib --skip_tests
  --skip_submodule_sync
)
# Cap the compile jobs with ORT_PARALLEL (unset = build.sh's own default, all
# cores). CUDA provider builds are memory-hungry, so a lower cap avoids OOM.
if [[ -n "${ORT_PARALLEL:-}" ]]; then
  build_cmd+=(--parallel "${ORT_PARALLEL}")
else
  build_cmd+=(--parallel)
fi
if [[ "${EP}" == "cuda" ]]; then
  build_cmd+=(
    --use_cuda --cuda_home "${CUDA_HOME}" --cudnn_home "${CUDNN_HOME}"
    --use_tensorrt --tensorrt_home "${TRT_HOME}"
    --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
    --cmake_extra_defines "CMAKE_CUDA_HOST_COMPILER=${CUDAHOSTCXX}"
  )
fi
build_cmd+=(--cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF)
if [[ -n "${ORT_CMAKE_EXTRA_DEFINES:-}" ]]; then
  # shellcheck disable=SC2206
  read -r -a _ort_extra_defines <<< "${ORT_CMAKE_EXTRA_DEFINES}"
  build_cmd+=(--cmake_extra_defines "${_ort_extra_defines[@]}")
fi
build_cmd+=("${extra_build_args[@]}")

echo "==> Build reduction:"
if [[ -n "${reduced_ops_config}" ]]; then
  echo "    --include_ops_by_config ${reduced_ops_config}"
else
  echo "    reduced-ops config: none (full kernel set)"
fi
if [[ ${#skipped_build_args[@]} -gt 0 ]]; then
  echo "    skipped (unsupported by this ORT version): ${skipped_build_args[*]}"
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "==> Dry run; would run in ${SRC_DIR}:"
  printf '    %q' "${build_cmd[@]}"
  printf '\n'
  exit 0
fi

pushd "${SRC_DIR}"
# --skip_submodule_sync keeps the build offline (prepare_offline_bundle.sh has
# already populated every submodule). --update is what runs the operator
# reduction when --include_ops_by_config is set.
if ! "${build_cmd[@]}"; then
  popd >/dev/null
  echo "error: ONNX Runtime build failed." >&2
  echo "       To check whether the reduction caused it, retry with:" >&2
  echo "         ORT_OPS_CONFIG= ORT_DISABLE_UNUSED_OPS=0 $0 ${ORIG_ARGS[*]}" >&2
  exit 1
fi
popd

BUILD_DIR="${SRC_DIR}/build/Linux/Release"

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}/lib" "${OUT_DIR}/include"

# Copy every produced library. The provider set differs across versions: 1.6
# links CUDA into the TensorRT provider, while >= 1.12 ships a separate
# libonnxruntime_providers_cuda.so. Globbing keeps this version-agnostic.
shopt -s nullglob
built_libs=("${BUILD_DIR}"/libonnxruntime*.so*)
shopt -u nullglob
if [[ ${#built_libs[@]} -eq 0 ]]; then
  echo "error: no libonnxruntime*.so* found in ${BUILD_DIR}" >&2
  exit 1
fi
cp -av "${built_libs[@]}" "${OUT_DIR}/lib/"

# Header names are stable across releases but their location under include/ is
# not, so look each one up by basename (flat layout expected by the vendor).
copy_header() {
  local name="$1" required="$2" src
  src="$(find "${SRC_DIR}/include" -name "${name}" -print -quit)"
  if [[ -n "${src}" ]]; then
    cp "${src}" "${OUT_DIR}/include/"
  elif [[ "${required}" == "required" ]]; then
    echo "error: required header ${name} not found under ${SRC_DIR}/include" >&2
    exit 1
  else
    echo "warning: header ${name} not found; skipping" >&2
  fi
}

copy_header onnxruntime_c_api.h required
copy_header onnxruntime_cxx_api.h required
copy_header onnxruntime_cxx_inline.h required
copy_header onnxruntime_float16.h optional
copy_header onnxruntime_session_options_config_keys.h optional
copy_header onnxruntime_run_options_config_keys.h optional
copy_header provider_options.h optional
copy_header cpu_provider_factory.h optional
if [[ "${EP}" == "cuda" ]]; then
  copy_header cuda_provider_factory.h optional
  copy_header tensorrt_provider_factory.h optional
fi

for notice in LICENSE ThirdPartyNotices.txt; do
  if [[ -f "${SRC_DIR}/${notice}" ]]; then
    cp "${SRC_DIR}/${notice}" "${OUT_DIR}/"
  else
    echo "warning: ${notice} not found in the source tree; skipping" >&2
  fi
done

# Record what was built so a later run can skip the rebuild.
{
  echo "version=${ORT_VERSION}"
  echo "ep=${EP}"
  echo "arch=${ARCH}"
  echo "cuda_arch=${CUDA_ARCH}"
  echo "extra_defines=${ORT_CMAKE_EXTRA_DEFINES:-}"
} > "${STAMP}"

echo "ONNX Runtime ${ORT_VERSION} installed to ${OUT_DIR}"
echo "Consume it with: colcon build --cmake-args -DONNXRUNTIME_ROOT=${OUT_DIR}"
