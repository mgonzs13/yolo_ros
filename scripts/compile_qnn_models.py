#!/usr/bin/env python3
# Copyright (c) 2026 Miguel Ángel González Santamarta
# SPDX-License-Identifier: MIT

"""Compile static YOLO ONNX models into W8A8 QNN artifacts for QCS8275.

Pipeline per model: AI Hub INT8 quantization of a static ONNX export, QNN
context-binary compilation for the target device, and a precompiled QNN ONNX
wrapping of that context for the ONNX Runtime QNN execution provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

TASKS = {
    "detect": "",
    "segment": "-seg",
    "pose": "-pose",
    "obb": "-obb",
    "classify": "-cls",
}
SIZES = ("n", "s", "m", "l", "x")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_DEVICE_NAME = "Dragonwing IQ-8275"
DEFAULT_SOC = "qcs8275"


def model_filename(size: str, task: str) -> str:
    if size not in SIZES:
        raise ValueError(f"unknown size {size!r} (expected one of {SIZES})")
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r} (expected one of {sorted(TASKS)})")
    return f"yolo26{size}{TASKS[task]}.onnx"


def collect_images(directory: Path, limit: int) -> list[Path]:
    """Return up to `limit` image paths under `directory`, evenly sampled.

    `limit=0` means "no limit" and negative values raise `ValueError`.
    """
    if limit < 0:
        raise ValueError("limit must be >= 0")
    images = sorted(
        path
        for path in Path(directory).rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"no calibration images under {directory}")
    if limit and limit < len(images):
        if limit == 1:
            images = [images[0]]
        else:
            step = (len(images) - 1) / (limit - 1)
            images = [images[round(i * step)] for i in range(limit)]
    return images


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize `image` with letterbox padding to `width` x `height`."""
    h, w = image.shape[:2]
    ratio = min(width / w, height / h)
    new_w, new_h = max(1, int(w * ratio)), max(1, int(h * ratio))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    top = (height - new_h) // 2
    left = (width - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def to_tensor(image: np.ndarray, rgb: bool) -> np.ndarray:
    """Convert a HWC uint8 image to a NCHW float32 tensor in [0, 1]."""
    if rgb:
        image = image[:, :, ::-1]
    tensor = image.astype(np.float32) / 255.0
    return np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]


def select_device_name(names: list[str], soc: str) -> str:
    """Return the AI Hub device whose name matches the SoC identifier."""
    compact = soc.lower().replace("-", "").replace(" ", "")
    if not compact:
        raise RuntimeError("no AI Hub device matches '': soc must be non-empty")
    for name in names:
        if compact in name.lower().replace("-", "").replace(" ", ""):
            return name
    # IQ-8275 doubles as the marketing name for QCS8275.
    if compact in {"qcs8275", "iq8275"}:
        for name in names:
            if "iq8275" in name.lower().replace("-", "").replace(" ", ""):
                return name
    available = ", ".join(names) or "none"
    raise RuntimeError(
        f"no AI Hub device matches {soc!r}; available: {available} "
        f"(expected a device named like {DEFAULT_DEVICE_NAME!r})"
    )


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of the file at `path`."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_entry(
    model: Path,
    task: str,
    soc: str,
    qairt_version: str,
    fp32_model: str,
    device: str = "",
) -> dict:
    """Build one manifest JSON entry for a compiled model."""
    return {
        "model": Path(model).name,
        "source": fp32_model,
        "task": task,
        "soc": soc,
        "qairt_version": qairt_version,
        "device": device,
        "sha256": sha256_file(model),
        "size_bytes": Path(model).stat().st_size,
    }


def write_manifest(path: Path, entries: list[dict]) -> None:
    """Write the model manifest JSON to `path`, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"models": entries}, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def _extract_if_zip(path: Path) -> Path:
    """Unzip path (or return it) and return the directory to search."""
    if path.suffix == ".zip":
        target = path.with_suffix("").with_suffix("")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(target)
        return target
    return path


def _remove_staging(downloaded: Path) -> None:
    """Remove the raw wrap download (file or directory) after deployment."""
    for candidate in (downloaded, downloaded.with_suffix("")):
        if candidate.is_dir():
            shutil.rmtree(candidate, ignore_errors=True)
        elif candidate.exists():
            candidate.unlink()


def deploy_precompiled_artifact(
    downloaded: Path, output_dir: Path, stem: str
) -> Path | None:
    """Copy a standalone precompiled ONNX into output_dir.

    AI Hub returns the wrapped PrecompiledQnnOnnx as either a plain ONNX or an
    (optionally zipped) directory holding <model>.onnx next to its QNN context
    binary. Only a truly standalone ONNX is a valid single-file deployment for
    the node; a wrap that references an external context binary cannot run
    from a single download, so return None and let the caller fall back to the
    self-caching QDQ model.
    """
    import onnx  # lazy, matching build_calibration

    root = _extract_if_zip(Path(downloaded))
    if root.is_dir():
        candidates = sorted(root.rglob("*.onnx"))
    else:
        candidates = [root]

    if len(candidates) != 1:
        return None
    source = candidates[0]

    if any(source.parent.glob("*.bin")):
        return None

    try:
        onnx.load(str(source))
    except Exception:
        return None

    output_dir = Path(output_dir)
    deployed = output_dir / f"{stem}_w8a8.ctx.onnx"
    shutil.copy2(source, deployed)
    return deployed


def compile_model(
    client,
    hub,
    onnx_path: Path,
    calibration: dict,
    device_name: str,
    output_dir: Path,
    qairt_version: str,
) -> dict:
    """Quantize, compile and wrap one ONNX model; download the artifacts."""
    stem = Path(onnx_path).stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = client.upload_model(onnx_path)
    quantize_job = client.submit_quantize_job(
        model=model,
        calibration_data=calibration,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
    )
    qdq_model = quantize_job.get_target_model()

    device = hub.Device(device_name)
    _, link_job = client.submit_compile_and_link_jobs(
        models=qdq_model,
        device=device,
        compile_options=[f"--qairt_version {qairt_version}"],
        link_options=f"--qairt_version {qairt_version}",
    )
    if link_job is None:
        raise RuntimeError(
            f"AI Hub returned no link job for {onnx_path}; no context binary "
            "was produced"
        )
    context_binary = link_job.get_target_model()

    # Wrapping the context binary takes no options; the result may be a single
    # ONNX or a zipped directory (see deploy_precompiled_artifact).
    wrap_job = client.submit_compile_job(model=context_binary, device=device)
    precompiled = wrap_job.get_target_model()

    qdq_path = output_dir / f"{stem}_w8a8.onnx"
    qdq_model.download(str(qdq_path))

    staged = output_dir / f"{stem}_w8a8_ctx_download"
    downloaded = Path(precompiled.download(str(staged)))
    ctx_path = deploy_precompiled_artifact(downloaded, output_dir, stem)
    _remove_staging(downloaded)
    if ctx_path is None:
        print(
            f"warning: {staged}: AI Hub returned a multi-file/zipped wrap; "
            "deploying the self-caching QDQ model instead",
            file=sys.stderr,
        )

    return {"qdq": qdq_path, "qnn": ctx_path, "deploy": ctx_path or qdq_path}


def build_calibration(onnx_path: Path, image_paths: list[Path]) -> dict:
    """Calibration dict matching the graph input (letterboxed NCHW float32)."""
    import onnx  # imported lazily so the pure functions stay import-light

    graph = onnx.load(str(onnx_path))
    inputs = [value for value in graph.graph.input]
    if len(inputs) != 1:
        raise RuntimeError(f"{onnx_path}: expected exactly one input")
    name = inputs[0].name
    dims = [dim.dim_value for dim in inputs[0].type.tensor_type.shape.dim]
    if len(dims) != 4 or dims[2] <= 0 or dims[3] <= 0:
        raise RuntimeError(
            f"{onnx_path}: QNN needs a static NCHW input, got shape {dims}"
        )
    elem_type = inputs[0].type.tensor_type.elem_type
    if dims[0] != 1 or elem_type != onnx.TensorProto.FLOAT:
        raise RuntimeError(
            f"{onnx_path}: expected a static batch-1 float32 input, "
            f"got shape {dims} element_type {elem_type}"
        )
    height, width = dims[2], dims[3]

    rgb = True
    for prop in graph.metadata_props:
        if prop.key == "input_color":
            rgb = prop.value.lower() != "bgr"

    samples = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        samples.append(to_tensor(letterbox(image, width, height), rgb))
    if not samples:
        raise RuntimeError("no loadable calibration images")
    return {name: samples}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), required=True)
    parser.add_argument("--size", choices=SIZES, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("qnn_models"))
    parser.add_argument("--soc", default=DEFAULT_SOC)
    parser.add_argument("--device-name", default="")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--qairt-version", default="latest")
    parser.add_argument("--hf-repo", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    for name, path in (("models-dir", args.models_dir), ("images", args.images)):
        if not path.is_dir():
            print(f"error: --{name} {path} is not a directory", file=sys.stderr)
            return 2

    try:
        images = collect_images(args.images, args.limit)
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    plan = []
    for task in dict.fromkeys(args.tasks):
        onnx_path = args.models_dir / model_filename(args.size, task)
        if not onnx_path.is_file():
            print(f"error: missing model {onnx_path}", file=sys.stderr)
            return 2
        plan.append((task, onnx_path))
        print(
            f"plan: {onnx_path} -> "
            f"{args.output_dir / (onnx_path.stem + '_w8a8.ctx.onnx')}"
        )

    if args.dry_run:
        return 0

    import qai_hub as hub  # deferred so --dry-run works without credentials

    client = hub.Client()
    names = [device.name for device in client.get_devices()]
    if args.device_name:
        if args.device_name not in names:
            available = ", ".join(names) or "none"
            print(
                f"error: --device-name {args.device_name!r} is not available; "
                f"available: {available}",
                file=sys.stderr,
            )
            return 2
        device_name = args.device_name
    else:
        try:
            device_name = select_device_name(names, args.soc)
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2

    entries = []
    for task, onnx_path in plan:
        try:
            calibration = build_calibration(onnx_path, images)
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        artifacts = compile_model(
            client=client,
            hub=hub,
            onnx_path=onnx_path,
            calibration=calibration,
            device_name=device_name,
            output_dir=args.output_dir,
            qairt_version=args.qairt_version,
        )
        entries.append(
            manifest_entry(
                model=artifacts["deploy"],
                task=task,
                soc=args.soc,
                qairt_version=args.qairt_version,
                fp32_model=onnx_path.name,
                device=device_name,
            )
        )
        print(f"compiled: {artifacts['deploy']}")

    write_manifest(Path(args.output_dir) / "manifest.json", entries)

    if args.hf_repo:
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            folder_path=str(args.output_dir),
            repo_id=args.hf_repo,
            path_in_repo="qnn",
            repo_type="model",
        )
        print(f"uploaded: {args.hf_repo}:qnn")

    return 0


if __name__ == "__main__":
    sys.exit(main())
