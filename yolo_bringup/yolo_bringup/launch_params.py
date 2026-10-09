# Copyright (c) 2026 Alejandro González Cantón
# Copyright (c) 2026 Miguel Ángel González Santamarta
# SPDX-License-Identifier: MIT

"""Command-line overrides for the yolo_bringup pipeline YAML.

The pipeline YAML (``config/yolo.yaml`` and its task presets) is the single
source of truth for the node: the node-level ``cameras`` list, the ordered
``plugins`` list, and each instance's ``plugin`` class, selected ``cameras``
and parameters. This module only exposes the scalar plugin parameters as
launch arguments, e.g.::

    ros2 launch yolo_bringup yolo.launch.py threshold:=0.5 tracker_type:=botsort

A provided argument becomes the ``<instance>.<param>`` override that rclcpp
layers on top of the YAML. Array parameters and structural fields
(``plugins``, ``<camera>.rgb_topic``, ``<instance>.plugin``,
``<instance>.cameras``) are YAML-only: a CLI value is always a plain string,
which rclcpp rejects for a ``std::vector<std::string>`` parameter.

An argument that is not passed keeps the value from the YAML params file, so
the YAML remains the source of defaults. The empty string is the "not provided"
sentinel; a non-empty YAML string therefore cannot be overridden *to* empty.

Each plugin instance reads its parameters under ``<instance>.<name>``, so the
override keys are instance-prefixed (``detection.threshold``,
``tracking.tracker_type``). Tracker knobs are inline under ``tracking:`` in the
pipeline YAML; the tracking instance declares only the knobs of its selected
``tracker_type``.
"""

from dataclasses import dataclass
from typing import Optional

from launch.actions import DeclareLaunchArgument


@dataclass(frozen=True)
class ParamSpec:
    """A plugin parameter exposed as a launch argument."""

    name: str
    type: type
    alias: Optional[str] = None
    cli: bool = True

    @property
    def arg(self) -> str:
        """The launch-argument name (upstream alias when one exists)."""
        return self.alias or self.name


#: Pluginlib class -> parameters it declares (order preserved), and the
#: instance name used by yolo.launch.py.
PLUGINS = {
    "yolo_ros/DetectionPlugin": "detection",
    "yolo_ros/TrackingPlugin": "tracking",
    "yolo_ros/Detect3DPlugin": "detection3d",
    "yolo_ros/DebugPlugin": "debug",
}

PLUGIN_PARAMS = {
    "yolo_ros/DetectionPlugin": (
        ParamSpec("model_type", str),
        ParamSpec("model_path", str),
        ParamSpec("model_repo", str),
        ParamSpec("model_filename", str),
        ParamSpec("cache_dir", str),
        ParamSpec("force_download", bool),
        ParamSpec("device", str),
        ParamSpec("provider", str),
        ParamSpec("cuda_graph_enable", bool),
        ParamSpec("trt_fp16_enable", bool),
        ParamSpec("trt_engine_cache_enable", bool),
        ParamSpec("trt_engine_cache_path", str),
        ParamSpec("threshold", float),
        ParamSpec("iou", float),
        ParamSpec("max_det", int),
        ParamSpec("n_threads", int),
        ParamSpec("max_fps", int),
        ParamSpec("top_k", int),
        ParamSpec("max_batch_size", int),
    ),
    "yolo_ros/TrackingPlugin": (
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
    "yolo_ros/Detect3DPlugin": (
        ParamSpec("target_frame", str),
        ParamSpec("depth_image_units_divisor", int),
        ParamSpec("enable_orientation", bool),
        ParamSpec("min_seg_points_for_orientation", int),
    ),
    "yolo_ros/DebugPlugin": (ParamSpec("marker_lifetime", float),),
}

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


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


def plugin_names(types) -> list:
    """['detection', ...] for the `plugins` parameter, in selection order."""
    return [PLUGINS[t] for t in types]


def param_arg_names(types) -> list:
    """Sorted unique launch-argument names for the given plugin types."""
    return sorted({spec.arg for t in types for spec in PLUGIN_PARAMS[t] if spec.cli})


def declare_param_arguments(types) -> list:
    """One DeclareLaunchArgument per unique argument, defaulting to ""."""
    targets = {}
    for plugin_type in types:
        for spec in PLUGIN_PARAMS[plugin_type]:
            if not spec.cli:
                continue
            targets.setdefault(spec.arg, set()).add(f"{PLUGINS[plugin_type]}.{spec.name}")
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


def build_overrides(context, plugin_type: str) -> dict:
    """{instance.param: value} for the plugin from provided launch args."""
    instance = PLUGINS[plugin_type]
    overrides = {}
    for spec in PLUGIN_PARAMS[plugin_type]:
        if not spec.cli:
            continue
        raw = context.launch_configurations.get(spec.arg, "")
        if raw is None or str(raw) == "":
            continue
        overrides[f"{instance}.{spec.name}"] = _convert(spec.arg, str(raw), spec.type)
    return overrides


def build_all_overrides(context, types) -> dict:
    """Merge every plugin's scalar CLI overrides into one dict.

    ``plugin`` classes, the ``plugins`` order and the per-instance ``cameras``
    selection live in the pipeline YAML and are never overridden from here.
    """
    overrides = {}
    for plugin_type in types:
        overrides.update(build_overrides(context, plugin_type))
    return overrides
