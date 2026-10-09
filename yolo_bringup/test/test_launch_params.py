# Copyright (c) 2026 Alejandro González Cantón
# Copyright (c) 2026 Miguel Ángel González Santamarta
# SPDX-License-Identifier: MIT

import os
import re

import pytest

import yolo_bringup.launch_params as launch_params
from yolo_bringup.launch_params import (
    build_all_overrides,
    build_overrides,
    declare_param_arguments,
    param_arg_names,
)

DET = "yolo_ros/DetectionPlugin"
TRACK = "yolo_ros/TrackingPlugin"
D3D = "yolo_ros/Detect3DPlugin"
DBG = "yolo_ros/DebugPlugin"


class FakeContext:
    def __init__(self, configs):
        self.launch_configurations = configs


REMOVED_PARAMS = {
    DET: {"image_topics", "camera_names", "image_reliability"},
    TRACK: {
        "image_topic",
        "image_reliability",
        "detections_topic",
        "output_topic",
    },
    D3D: {
        "depth_image_topic",
        "depth_info_topic",
        "depth_image_reliability",
        "depth_info_reliability",
        "detections_topic",
        "output_topic",
    },
    DBG: {
        "image_topic",
        "image_reliability",
        "detections_topic",
        "markers_topic",
        "output_prefix",
    },
}


def test_types_are_converted():
    context = FakeContext(
        {
            "model_path": "/x.onnx",
            "threshold": "0.5",
            "max_det": "300",
            "force_download": "false",
        }
    )
    overrides = build_overrides(context, DET)
    assert overrides["detection.model_path"] == "/x.onnx"
    assert overrides["detection.threshold"] == pytest.approx(0.5)
    assert overrides["detection.max_det"] == 300
    assert overrides["detection.force_download"] is False


def test_empty_values_are_omitted():
    context = FakeContext({"model_path": "", "threshold": "", "iou": ""})
    assert build_overrides(context, DET) == {}


def test_bool_variants():
    for raw, expected in [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ]:
        context = FakeContext({"force_download": raw})
        assert build_overrides(context, DET)["detection.force_download"] is expected


def test_invalid_value_raises_with_arg_name():
    context = FakeContext({"threshold": "high"})
    with pytest.raises(RuntimeError, match="threshold"):
        build_overrides(context, DET)


def test_invalid_bool_raises_with_arg_name():
    context = FakeContext({"force_download": "maybe"})
    with pytest.raises(RuntimeError, match="force_download"):
        build_overrides(context, DET)


@pytest.mark.parametrize("plugin_type", sorted(REMOVED_PARAMS))
def test_removed_params_are_absent_from_catalogue(plugin_type):
    names = {spec.name for spec in launch_params.PLUGIN_PARAMS[plugin_type]}
    assert names.isdisjoint(REMOVED_PARAMS[plugin_type])
    declared = {argument.name for argument in declare_param_arguments([plugin_type])}
    assert declared.isdisjoint(REMOVED_PARAMS[plugin_type])


def test_removed_topic_aliases_are_not_declared():
    declared = {
        argument.name for argument in declare_param_arguments(list(launch_params.PLUGINS))
    }
    for alias in (
        "input_image_topic",
        "input_depth_topic",
        "input_depth_info_topic",
        "tracker",
    ):
        assert alias not in declared


def test_structural_params_are_not_in_catalogue():
    for specs in launch_params.PLUGIN_PARAMS.values():
        names = {spec.name for spec in specs}
        assert "plugin" not in names
        assert "cameras" not in names


def test_shared_arg_applied_to_each_target_plugin():
    context = FakeContext({"provider": "cpu", "device": "cpu:0"})
    assert build_overrides(context, DET)["detection.provider"] == "cpu"
    assert build_overrides(context, TRACK)["tracking.provider"] == "cpu"
    assert build_overrides(context, DET)["detection.device"] == "cpu:0"
    assert build_overrides(context, TRACK)["tracking.device"] == "cpu:0"
    # Neither the 3D nor the debug plugin declares provider/device.
    assert build_overrides(context, D3D) == {}
    assert build_overrides(context, DBG) == {}


def test_plugin_names_are_instance_names():
    assert launch_params.plugin_names([DET, TRACK, D3D, DBG]) == [
        "detection",
        "tracking",
        "detection3d",
        "debug",
    ]
    assert launch_params.plugin_names([DET, DBG]) == ["detection", "debug"]
    assert launch_params.plugin_names([D3D, DET]) == ["detection3d", "detection"]


def test_build_all_overrides_does_not_inject_plugin_classes():
    overrides = build_all_overrides(FakeContext({}), [DET, TRACK, D3D, DBG])
    assert overrides == {}
    assert not any(key.endswith(".plugin") for key in overrides)


def test_build_all_overrides_merges_plugins():
    context = FakeContext(
        {
            "threshold": "0.5",
            "tracker_type": "botsort",
            "marker_lifetime": "1.5",
        }
    )
    overrides = build_all_overrides(context, [DET, TRACK, DBG])
    assert overrides["detection.threshold"] == pytest.approx(0.5)
    assert overrides["tracking.tracker_type"] == "botsort"
    assert overrides["debug.marker_lifetime"] == pytest.approx(1.5)
    assert "detection.plugin" not in overrides
    assert "tracking.plugin" not in overrides
    assert "debug.plugin" not in overrides


def test_arg_names_are_unique():
    names = param_arg_names([DET, TRACK, DBG])
    assert names == sorted(set(names))
    assert names.count("provider") == 1
    assert names.count("device") == 1
    assert "model_path" in names
    assert "tracker_type" in names
    assert "marker_lifetime" in names
    assert "input_image_topic" not in names
    assert "image_reliability" not in names
    assert "output_prefix" not in names


def test_declare_returns_one_per_arg_name():
    types = [DET, TRACK, D3D, DBG]
    assert len(declare_param_arguments(types)) == len(param_arg_names(types))


def test_unknown_plugin_type_raises():
    with pytest.raises(KeyError, match="nope"):
        build_overrides(FakeContext({}), "nope")


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_CPP_PLUGIN_SOURCES = {
    DET: ["yolo_ros/src/plugins/detection_plugin.cpp"],
    TRACK: [
        "yolo_ros/src/plugins/tracking_plugin.cpp",
        "yolo_ros/include/yolo_ros/tracking/byte_tracker.hpp",
        "yolo_ros/include/yolo_ros/tracking/bot_sort.hpp",
    ],
    D3D: ["yolo_ros/src/plugins/detect_3d_plugin.cpp"],
    DBG: ["yolo_ros/src/plugins/debug_plugin.cpp"],
}

_DECLARE_PARAM = re.compile(
    r'declare(?:_parameter)?(?:<[^(){}]*>)*\s*\(\s*(?:\w+\s*\+\s*)?"([^"]+)"'
)

_DECLARE_PARAM_TYPED = re.compile(
    r'declare(?:_parameter)?\s*<(.+?)>\s*\(\s*(?:\w+\s*\+\s*)?"([^"]+)"'
)


@pytest.mark.parametrize("plugin_type", sorted(_CPP_PLUGIN_SOURCES))
def test_catalogue_params_are_declared_in_cpp(plugin_type):
    declared = set()
    for relative_path in _CPP_PLUGIN_SOURCES[plugin_type]:
        with open(os.path.join(_REPO_ROOT, relative_path)) as source:
            declared.update(_DECLARE_PARAM.findall(source.read()))
    catalogue = {spec.name for spec in launch_params.PLUGIN_PARAMS[plugin_type]}
    assert (
        catalogue <= declared
    ), f"{plugin_type}: catalogue-only={sorted(catalogue - declared)}"


@pytest.mark.parametrize("plugin_type", sorted(_CPP_PLUGIN_SOURCES))
def test_non_cli_catalogue_params_are_vectors_in_cpp(plugin_type):
    declared = {}
    for relative_path in _CPP_PLUGIN_SOURCES[plugin_type]:
        with open(os.path.join(_REPO_ROOT, relative_path)) as source:
            declared.update(
                {
                    name: cpp_type
                    for cpp_type, name in _DECLARE_PARAM_TYPED.findall(source.read())
                }
            )
    for spec in launch_params.PLUGIN_PARAMS[plugin_type]:
        if spec.cli:
            continue
        assert "std::vector" in declared.get(spec.name, ""), (
            f"{plugin_type}: non-CLI parameter {spec.name!r} must be declared "
            f"as std::vector in C++, got {declared.get(spec.name)!r}"
        )
