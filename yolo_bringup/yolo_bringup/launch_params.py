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

A provided argument becomes the ``<instance>.<namespace>.<param>`` override that
rclcpp layers on top of the YAML. Array parameters and structural fields
(``plugins``, ``<camera>.rgb.topic``, ``<instance>.plugin``,
``<instance>.cameras``) are YAML-only: a CLI value is always a plain string,
which rclcpp rejects for a ``std::vector<std::string>`` parameter.

An argument that is not passed keeps the value from the YAML params file, so
the YAML remains the source of defaults. The empty string is the "not provided"
sentinel; a non-empty YAML string therefore cannot be overridden *to* empty.

Each plugin instance reads its parameters under ``<instance>.<namespace>.<name>``,
so the override keys are instance-prefixed and namespaced
(``detection.inference.threshold``, ``tracking.track.high_thresh``). The
launch-argument names keep the historical flat aliases (``threshold``,
``track_high_thresh``, ...) and are translated to those namespaced parameters.
Tracker knobs are grouped under ``track.*``/``gmc.*``/``reid.*`` in the pipeline
YAML; the tracking instance declares only the knobs of its selected
``tracker_type``.
"""

from dataclasses import dataclass
from typing import Optional

from launch.actions import DeclareLaunchArgument


@dataclass(frozen=True)
class ParamSpec:
    """A plugin parameter exposed as a launch argument.

    ``name`` is the ROS parameter name relative to the plugin instance
    (e.g. ``model.path``); ``alias`` is the historical flat launch-argument
    name (e.g. ``model_path``). The override is always written with the
    namespaced ``name``.
    """

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
        ParamSpec("model.type", str, alias="model_type"),
        ParamSpec("model.path", str, alias="model_path"),
        ParamSpec("model.repo", str, alias="model_repo"),
        ParamSpec("model.filename", str, alias="model_filename"),
        ParamSpec("model.cache_dir", str, alias="cache_dir"),
        ParamSpec("model.force_download", bool, alias="force_download"),
        ParamSpec("device", str),
        ParamSpec("provider", str),
        ParamSpec("cuda.graph_enable", bool, alias="cuda_graph_enable"),
        ParamSpec("trt.fp16_enable", bool, alias="trt_fp16_enable"),
        ParamSpec("trt.engine_cache_enable", bool, alias="trt_engine_cache_enable"),
        ParamSpec("trt.engine_cache_path", str, alias="trt_engine_cache_path"),
        ParamSpec("qnn.htp_performance_mode", str, alias="qnn_htp_performance_mode"),
        ParamSpec("qnn.htp_arch", str, alias="qnn_htp_arch"),
        ParamSpec("qnn.soc_model", str, alias="qnn_soc_model"),
        ParamSpec("qnn.backend_path", str, alias="qnn_backend_path"),
        ParamSpec(
            "qnn.offload_graph_io_quantization",
            bool,
            alias="qnn_offload_graph_io_quantization",
        ),
        ParamSpec(
            "qnn.enable_htp_fp16_precision",
            bool,
            alias="qnn_enable_htp_fp16_precision",
        ),
        ParamSpec("qnn.context_cache_enable", bool, alias="qnn_context_cache_enable"),
        ParamSpec("qnn.context_cache_path", str, alias="qnn_context_cache_path"),
        ParamSpec("inference.threshold", float, alias="threshold"),
        ParamSpec("inference.iou", float, alias="iou"),
        ParamSpec("inference.max_det", int, alias="max_det"),
        ParamSpec("cpu.n_threads", int, alias="n_threads"),
        ParamSpec("inference.max_fps", int, alias="max_fps"),
        ParamSpec("input.width", int, alias="img_width"),
        ParamSpec("input.height", int, alias="img_height"),
        ParamSpec("inference.top_k", int, alias="top_k"),
        ParamSpec("batch.max_size", int, alias="max_batch_size"),
        ParamSpec("batch.pad_to", int, alias="pad_to_batch"),
    ),
    "yolo_ros/TrackingPlugin": (
        ParamSpec("tracker_type", str),
        ParamSpec("track.high_thresh", float, alias="track_high_thresh"),
        ParamSpec("track.low_thresh", float, alias="track_low_thresh"),
        ParamSpec("track.new_thresh", float, alias="new_track_thresh"),
        ParamSpec("track.buffer", int, alias="track_buffer"),
        ParamSpec("track.match_thresh", float, alias="match_thresh"),
        ParamSpec("track.fuse_score", bool, alias="fuse_score"),
        ParamSpec("gmc.method", str, alias="gmc_method"),
        ParamSpec("gmc.downscale", int, alias="gmc_downscale"),
        ParamSpec("reid.enable", bool, alias="with_reid"),
        ParamSpec("reid.model", str, alias="reid_model"),
        ParamSpec("reid.proximity_thresh", float, alias="proximity_thresh"),
        ParamSpec("reid.appearance_thresh", float, alias="appearance_thresh"),
        ParamSpec("reid.provider", str, alias="provider"),
        ParamSpec("reid.device", str, alias="device"),
    ),
    "yolo_ros/Detect3DPlugin": (
        ParamSpec("target_frame", str),
        ParamSpec("depth.units_divisor", int, alias="depth_image_units_divisor"),
        ParamSpec("orientation.enable", bool, alias="enable_orientation"),
        ParamSpec(
            "orientation.min_seg_points", int, alias="min_seg_points_for_orientation"
        ),
    ),
    "yolo_ros/DebugPlugin": (
        ParamSpec("marker_lifetime", float),
        ParamSpec("compressed.enable", bool, alias="debug_compressed_enable"),
        ParamSpec("compressed.quality", int, alias="debug_compressed_quality"),
    ),
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
