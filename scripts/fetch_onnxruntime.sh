#!/usr/bin/env bash
# Copyright (C) 2026 Miguel Ángel González Santamarta
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Fetch the released ROS 2 onnxruntime_vendor CPU runtime and unpack it into a
# flat ONNXRUNTIME_ROOT prefix. This is for distros that do not ship
# onnxruntime_vendor themselves (Foxy, Galactic, Iron); the package is a
# self-contained ONNX Runtime binary and only needs libc/libstdc++.
#
# Usage: scripts/fetch_onnxruntime.sh [output_dir]
#   output_dir defaults to <repo>/ort-1.24.3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-${SCRIPT_DIR}/../ort-1.24.3}"
BASE_URL="http://packages.ros.org/ros2/ubuntu"
PACKAGE="ros-humble-onnxruntime-vendor"
CODENAME="jammy"

for tool in curl gzip awk dpkg-deb sha256sum; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "error: ${tool} is required to fetch ONNX Runtime" >&2
        exit 1
    fi
done

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

curl -fsSL "${BASE_URL}/dists/${CODENAME}/main/binary-amd64/Packages.gz" \
    -o "${TMP_DIR}/Packages.gz"
ENTRY="$(
    gzip -dc "${TMP_DIR}/Packages.gz" |
        awk -v pkg="${PACKAGE}" 'BEGIN { RS = ""; FS = "\n" } $1 == "Package: " pkg && !found { print; found = 1 }'
)"

if [[ -z "${ENTRY}" ]]; then
    echo "error: ${PACKAGE} was not found in the ${CODENAME} package index" >&2
    exit 1
fi

FILENAME="$(printf '%s\n' "${ENTRY}" | awk -F': ' '/^Filename: / { print $2; exit }')"
SHA256="$(printf '%s\n' "${ENTRY}" | awk -F': ' '/^SHA256: / { print $2; exit }')"
if [[ -z "${FILENAME}" || -z "${SHA256}" ]]; then
    echo "error: could not parse ${PACKAGE} from the package index" >&2
    exit 1
fi

echo "Downloading ${PACKAGE} from ${BASE_URL}/${FILENAME}" >&2
curl -fsSL "${BASE_URL}/${FILENAME}" -o "${TMP_DIR}/onnxruntime-vendor.deb"
printf '%s  %s\n' "${SHA256}" "${TMP_DIR}/onnxruntime-vendor.deb" | sha256sum -c - >/dev/null

dpkg-deb -x "${TMP_DIR}/onnxruntime-vendor.deb" "${TMP_DIR}/unpacked"
PREFIX="${TMP_DIR}/unpacked/opt/ros/humble/opt/onnxruntime_vendor"
if [[ ! -f "${PREFIX}/include/onnxruntime_cxx_api.h" || ! -e "${PREFIX}/lib/libonnxruntime.so" ]]; then
    echo "error: the downloaded package does not contain the expected ONNX Runtime layout" >&2
    exit 1
fi

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"
cp -a "${PREFIX}/." "${OUT_DIR}/"

echo "ONNX Runtime prefix ready at ${OUT_DIR}"
