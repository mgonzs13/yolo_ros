#!/usr/bin/env python3
# Copyright (c) 2026 Miguel Ángel González Santamarta
# SPDX-License-Identifier: MIT

"""Compare FP32 and W8A8 QNN outputs on the board (accuracy gate).

Runs on the VENTUNO Q with onnxruntime-qnn installed:
  pip install onnxruntime-qnn
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from compile_qnn_models import collect_images, letterbox, to_tensor


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b) / denominator)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def verdict(min_cosine: float, threshold: float) -> bool:
    return min_cosine >= threshold


def merge_manifest(path: Path, model_name: str, update: dict) -> bool:
    """Merge an accuracy update into the compile manifest entry.

    Returns True when the entry was updated, False when the model was not
    found. A missing manifest is treated as empty; the write is atomic.
    """
    path = Path(path)
    data = json.loads(path.read_text()) if path.is_file() else {"models": []}
    updated = False
    for entry in data.get("models", []):
        if entry.get("model") == model_name:
            if not isinstance(entry.get("accuracy"), dict):
                entry["accuracy"] = {}
            entry["accuracy"].update(update)
            updated = True
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)
    return updated


def resolve_input_color(requested: str, metadata: dict) -> bool:
    """True when the graph expects RGB; requested wins over metadata."""
    if requested == "rgb":
        return True
    if requested == "bgr":
        return False
    return str(metadata.get("input_color", "rgb")).lower() != "bgr"


def min_cosine_arg(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("min-cosine must be in [0, 1]")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--qnn", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--min-cosine", type=min_cosine_arg, default=0.99)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--input-color", choices=("auto", "rgb", "bgr"), default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "error: onnxruntime is not installed; install onnxruntime-qnn "
            "(pip install onnxruntime-qnn) or use an ONNX Runtime build with "
            "the QNN execution provider",
            file=sys.stderr,
        )
        return 2

    try:
        images = collect_images(args.images, args.limit)
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        fp32_session = ort.InferenceSession(
            str(args.fp32), providers=["CPUExecutionProvider"]
        )
    except Exception as error:
        print(
            f"error: could not load {args.fp32} (CPUExecutionProvider): {error}",
            file=sys.stderr,
        )
        return 2
    try:
        qnn_session = ort.InferenceSession(
            str(args.qnn),
            providers=["QNNExecutionProvider"],
            provider_options=[{"backend_path": "libQnnHtp.so"}],
        )
    except Exception as error:
        print(
            f"error: could not load {args.qnn} (QNNExecutionProvider): {error}",
            file=sys.stderr,
        )
        return 2

    if "QNNExecutionProvider" not in qnn_session.get_providers():
        print(
            "error: the QNN execution provider is not available; install "
            "onnxruntime-qnn (pip install onnxruntime-qnn) or use an ONNX "
            "Runtime build with --use_qnn and ensure libQnnHtp.so is on the "
            "loader path",
            file=sys.stderr,
        )
        return 2

    rgb = resolve_input_color(
        args.input_color, fp32_session.get_modelmeta().custom_metadata_map
    )

    fp32_inputs = fp32_session.get_inputs()
    qnn_inputs = qnn_session.get_inputs()
    fp32_outputs = fp32_session.get_outputs()
    qnn_outputs = qnn_session.get_outputs()
    if not fp32_inputs or not qnn_inputs:
        print(
            f"error: {args.fp32} and {args.qnn} must each declare an input",
            file=sys.stderr,
        )
        return 2
    if len(fp32_inputs) != len(qnn_inputs) or len(fp32_outputs) != len(qnn_outputs):
        print(
            f"error: input/output counts differ: {args.fp32} has "
            f"{len(fp32_inputs)} input(s)/{len(fp32_outputs)} output(s), "
            f"{args.qnn} has {len(qnn_inputs)} input(s)/{len(qnn_outputs)} output(s)",
            file=sys.stderr,
        )
        return 2

    shape = fp32_inputs[0].shape
    if (
        len(shape) != 4
        or not isinstance(shape[2], int)
        or not isinstance(shape[3], int)
        or shape[2] <= 0
        or shape[3] <= 0
    ):
        print(
            f"error: {args.fp32} needs a static NCHW input, got shape {shape}",
            file=sys.stderr,
        )
        return 2
    height, width = shape[2], shape[3]
    fp32_name = fp32_inputs[0].name
    qnn_name = qnn_inputs[0].name

    worst = 1.0
    worst_max_abs = 0.0
    compared = 0
    qnn_ms: list[float] = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            continue
        tensor = to_tensor(letterbox(image, width, height), rgb=rgb)
        reference = fp32_session.run(None, {fp32_name: tensor})
        start = time.perf_counter()
        quantized = qnn_session.run(None, {qnn_name: tensor})
        qnn_ms.append((time.perf_counter() - start) * 1000.0)
        for index, (ref, qnt) in enumerate(zip(reference, quantized)):
            similarity = cosine(ref, qnt)
            if not np.isfinite(similarity) or similarity < worst:
                worst = similarity
            difference = max_abs_diff(ref, qnt)
            worst_max_abs = max(worst_max_abs, difference)
            compared += 1
            print(
                f"{path.name} output[{index}]: cosine={similarity:.6f} "
                f"max_abs={difference:.6f}"
            )

    if compared == 0:
        print("error: no outputs were compared", file=sys.stderr)
        return 2

    mean_qnn_ms = sum(qnn_ms) / len(qnn_ms)
    print(f"qnn latency: {mean_qnn_ms:.3f} ms/image")

    ok = verdict(worst, args.min_cosine)
    print(
        f"worst cosine={worst:.6f} (threshold {args.min_cosine}): "
        f"{'PASS' if ok else 'FAIL'}"
    )

    if args.manifest is not None and args.model_name:

        def _finite(value: float):
            return float(value) if np.isfinite(value) else None

        recorded = merge_manifest(
            args.manifest,
            args.model_name,
            {
                "worst_cosine": _finite(worst),
                "worst_max_abs": _finite(worst_max_abs),
                "mean_qnn_ms": _finite(mean_qnn_ms),
            },
        )
        if not recorded:
            print(
                f"warning: {args.model_name} not found in {args.manifest}; "
                "accuracy not recorded",
                file=sys.stderr,
            )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
