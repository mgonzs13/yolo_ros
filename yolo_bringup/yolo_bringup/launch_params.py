# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT

"""Command-line overrides for the yolo_bringup YAML params files.

Every ROS parameter declared by the C++ nodes can be overridden from the
command line, e.g.::

    ros2 launch yolo_bringup yolo.launch.py model:=/x.onnx threshold:=0.5

An argument that is not passed keeps the value from the YAML params file, so
the YAML remains the source of defaults. The empty string is the "not provided"
sentinel; a non-empty YAML string therefore cannot be overridden *to* empty.

Argument names mirror the upstream Python launch where one existed
(``input_image_topic``, ``input_depth_topic``, ``input_depth_info_topic``,
``tracker``); every other argument has the same name as its parameter.
"""

import os
from dataclasses import dataclass
from typing import Optional

import yaml
from launch.actions import DeclareLaunchArgument


@dataclass(frozen=True)
class ParamSpec:
    """A node parameter exposed as a launch argument."""

    name: str
    type: type
    alias: Optional[str] = None

    @property
    def arg(self) -> str:
        """The launch-argument name (upstream alias when one exists)."""
        return self.alias or self.name


#: Node name (Node(name=...)) -> parameters it declares, in declaration order.
NODE_PARAMS = {
    "yolo_node": (
        ParamSpec("model_type", str),
        ParamSpec("model", str),
        ParamSpec("model_repo", str),
        ParamSpec("model_filename", str),
        ParamSpec("cache_dir", str),
        ParamSpec("force_download", bool),
        ParamSpec("device", str),
        ParamSpec("provider", str),
        ParamSpec("trt_fp16_enable", bool),
        ParamSpec("trt_engine_cache_enable", bool),
        ParamSpec("trt_engine_cache_path", str),
        ParamSpec("threshold", float),
        ParamSpec("iou", float),
        ParamSpec("max_det", int),
        ParamSpec("enable", bool),
        ParamSpec("image_reliability", int),
        ParamSpec("image_topic", str, "input_image_topic"),
        ParamSpec("n_threads", int),
        ParamSpec("max_fps", int),
        ParamSpec("top_k", int),
    ),
    "tracking_node": (
        ParamSpec("image_reliability", int),
        ParamSpec("image_topic", str, "input_image_topic"),
        ParamSpec("tracker_type", str),
        ParamSpec("track_high_thresh", float),
        ParamSpec("track_low_thresh", float),
        ParamSpec("new_track_thresh", float),
        ParamSpec("track_buffer", int),
        ParamSpec("match_thresh", float),
        ParamSpec("fuse_score", bool),
        ParamSpec("gmc_method", str),
        ParamSpec("gmc_downscale", int),
        ParamSpec("with_reid", bool),
        ParamSpec("reid_model", str),
        ParamSpec("proximity_thresh", float),
        ParamSpec("appearance_thresh", float),
        ParamSpec("provider", str),
        ParamSpec("device", str),
    ),
    "detect_3d_node": (
        ParamSpec("target_frame", str),
        ParamSpec("depth_image_units_divisor", int),
        ParamSpec("depth_image_reliability", int),
        ParamSpec("depth_info_reliability", int),
        ParamSpec("depth_image_topic", str, "input_depth_topic"),
        ParamSpec("depth_info_topic", str, "input_depth_info_topic"),
        ParamSpec("detections_topic", str),
        ParamSpec("enable_orientation", bool),
        ParamSpec("min_seg_points_for_orientation", int),
    ),
    "debug_node": (
        ParamSpec("image_reliability", int),
        ParamSpec("image_topic", str, "input_image_topic"),
        ParamSpec("detections_topic", str),
        ParamSpec("markers_topic", str),
        ParamSpec("marker_lifetime", float),
    ),
    "yolo_batch_node": (
        ParamSpec("model_type", str),
        ParamSpec("model", str),
        ParamSpec("model_repo", str),
        ParamSpec("model_filename", str),
        ParamSpec("cache_dir", str),
        ParamSpec("force_download", bool),
        ParamSpec("device", str),
        ParamSpec("provider", str),
        ParamSpec("trt_fp16_enable", bool),
        ParamSpec("trt_engine_cache_enable", bool),
        ParamSpec("trt_engine_cache_path", str),
        ParamSpec("threshold", float),
        ParamSpec("iou", float),
        ParamSpec("max_det", int),
        ParamSpec("enable", bool),
        ParamSpec("image_reliability", int),
        ParamSpec("n_threads", int),
        ParamSpec("max_fps", int),
        ParamSpec("top_k", int),
        # Array params are provided by the pipeline file, never a CLI argument.
        ParamSpec("camera_names", str),
        ParamSpec("image_topics", str),
        ParamSpec("max_batch_size", int),
    ),
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _selected_node_params(node_names):
    """Yield (node_name, ParamSpec); raise KeyError on an unknown node."""
    for node_name in node_names:
        if node_name not in NODE_PARAMS:
            raise KeyError(f"unknown node '{node_name}'")
        for spec in NODE_PARAMS[node_name]:
            yield node_name, spec


def param_arg_names(node_names) -> list:
    """Sorted unique launch-argument names for the given nodes."""
    return sorted({spec.arg for _, spec in _selected_node_params(node_names)})


def declare_param_arguments(node_names) -> list:
    """One DeclareLaunchArgument per unique argument, defaulting to ""."""
    targets = {}
    for node_name, spec in _selected_node_params(node_names):
        targets.setdefault(spec.arg, set()).add(f"{node_name}.{spec.name}")
    return [
        DeclareLaunchArgument(
            arg_name,
            default_value="",
            description=(
                f"Override {', '.join(sorted(targets[arg_name]))}. "
                "Empty keeps the value from the YAML params file."
            ),
        )
        for arg_name in sorted(targets)
    ]


def _convert(arg_name: str, raw: str, value_type: type):
    if value_type is bool:
        lowered = raw.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise RuntimeError(
            f"invalid value for '{arg_name}': {raw!r} (expected a boolean)"
        )
    try:
        return value_type(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"invalid value for '{arg_name}': {raw!r} "
            f"(expected {value_type.__name__})"
        ) from exc


def build_overrides(context, node_name: str) -> dict:
    """Assemble {param_name: value} for the node from provided launch args."""
    if node_name not in NODE_PARAMS:
        raise KeyError(f"unknown node '{node_name}'")
    overrides = {}
    for spec in NODE_PARAMS[node_name]:
        raw = context.launch_configurations.get(spec.arg, "")
        if raw is None or str(raw) == "":
            continue
        overrides[spec.name] = _convert(spec.arg, str(raw), spec.type)
    return overrides


def pipeline_tracker(params_file, namespace: str = "yolo") -> str:
    """Read the ``tracker`` selector from a pipeline params file.

    Looks up ``/<namespace>/tracking_node`` -> ``ros__parameters`` -> ``tracker``
    so the launch can pick ``config/trackers/<tracker>.yaml`` without a CLI
    argument. Returns ``"bytetrack"`` when the file or key is absent.
    """
    if not params_file or not os.path.isfile(params_file):
        return "bytetrack"
    with open(params_file) as handle:
        data = yaml.safe_load(handle) or {}
    block = data.get(f"/{namespace}/tracking_node", {})
    parameters = block.get("ros__parameters", {}) if isinstance(block, dict) else {}
    return str(parameters.get("tracker", "bytetrack"))


def tracker_params_file(tracker: str) -> str:
    """Path to the tracker config file selected by the ``tracker`` argument.

    Accepts a tracker config name (resolved to ``config/trackers/<name>.yaml``,
    e.g. ``bytetrack``/``botsort``/``botsort_reid``) or a path/filename to any
    ROS params file. Empty -> the ``bytetrack`` default. Each config file then
    sets the real ``tracker_type`` (the C++ implementation) it runs.
    """
    from ament_index_python.packages import get_package_share_directory

    config_dir = os.path.join(get_package_share_directory("yolo_bringup"), "config")
    trackers_dir = os.path.join(config_dir, "trackers")
    value = (tracker or "bytetrack").strip()
    if os.sep in value or value.endswith((".yaml", ".yml")):
        candidates = (
            [value]
            if os.path.isabs(value)
            else [
                value,
                os.path.join(config_dir, value),
                os.path.join(trackers_dir, value),
            ]
        )
        path = next((c for c in candidates if os.path.isfile(c)), candidates[0])
    else:
        path = os.path.join(trackers_dir, f"{value}.yaml")
    if not os.path.isfile(path):
        available = sorted(
            entry[:-5]
            for entry in os.listdir(trackers_dir)
            if entry.endswith(".yaml")
            and os.path.isfile(os.path.join(trackers_dir, entry))
        )
        raise RuntimeError(
            f"unknown tracker '{value}'; available trackers: {', '.join(available)} "
            "(or pass a path to a params file)"
        )
    return path


def load_params_mapping(path: str) -> dict:
    """Return the first ``ros__parameters`` mapping in a ROS params file.

    Per-tracker files wrap their values under a fully-qualified node name
    (``/yolo/tracking_node: ros__parameters: ...``). The multi-camera launch
    re-keys them per camera, so it needs the inner mapping without the node
    name. Returns ``{}`` for a missing file or a file without such a block.
    """
    if not path or not os.path.isfile(path):
        return {}
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    for block in data.values():
        if isinstance(block, dict) and isinstance(block.get("ros__parameters"), dict):
            return dict(block["ros__parameters"])
    return {}


def node_parameters(params_file, context, node_name: str, extra_files=()) -> list:
    """Parameters list for a Node: the YAML file(s) plus the CLI overrides.

    ``extra_files`` are layered between the pipeline config and the CLI
    overrides (the per-tracker file for the tracking node), so later values win.
    """
    return [params_file, *extra_files, build_overrides(context, node_name)]
