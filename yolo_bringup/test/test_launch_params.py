# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT

import os
import re

import pytest
from launch.substitutions import LaunchConfiguration

import yolo_bringup.launch_params as launch_params
from yolo_bringup.launch_params import (
    build_overrides,
    declare_param_arguments,
    node_parameters,
    param_arg_names,
)


class FakeContext:
    def __init__(self, configs):
        self.launch_configurations = configs


def test_types_are_converted():
    context = FakeContext(
        {
            "model": "/x.onnx",
            "threshold": "0.5",
            "max_det": "300",
            "enable": "false",
            "input_image_topic": "/cam",
        }
    )
    overrides = build_overrides(context, "yolo_node")
    assert overrides["model"] == "/x.onnx"
    assert overrides["threshold"] == pytest.approx(0.5)
    assert overrides["max_det"] == 300
    assert overrides["enable"] is False
    assert overrides["image_topic"] == "/cam"


def test_empty_values_are_omitted():
    context = FakeContext({"model": "", "threshold": "", "iou": ""})
    assert build_overrides(context, "yolo_node") == {}


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
        context = FakeContext({"enable": raw})
        assert build_overrides(context, "yolo_node")["enable"] is expected


def test_invalid_value_raises_with_arg_name():
    context = FakeContext({"threshold": "high"})
    with pytest.raises(RuntimeError, match="threshold"):
        build_overrides(context, "yolo_node")


def test_invalid_bool_raises_with_arg_name():
    context = FakeContext({"enable": "maybe"})
    with pytest.raises(RuntimeError, match="enable"):
        build_overrides(context, "yolo_node")


def test_aliases_map_to_param_names():
    context = FakeContext(
        {
            "input_image_topic": "/cam",
            "input_depth_topic": "/depth",
            "input_depth_info_topic": "/depth_info",
        }
    )
    assert build_overrides(context, "tracking_node")["image_topic"] == "/cam"
    assert build_overrides(context, "detect_3d_node")["depth_image_topic"] == "/depth"
    assert build_overrides(context, "detect_3d_node")["depth_info_topic"] == "/depth_info"


def test_shared_arg_applied_to_each_target_node():
    context = FakeContext({"image_reliability": "1", "detections_topic": "detections"})
    assert build_overrides(context, "yolo_node")["image_reliability"] == 1
    assert build_overrides(context, "tracking_node")["image_reliability"] == 1
    assert build_overrides(context, "debug_node")["image_reliability"] == 1
    assert build_overrides(context, "detect_3d_node")["detections_topic"] == "detections"
    assert build_overrides(context, "debug_node")["detections_topic"] == "detections"


def test_arg_names_are_unique():
    names = param_arg_names(["yolo_node", "tracking_node", "debug_node"])
    assert names == sorted(set(names))
    assert names.count("input_image_topic") == 1
    assert names.count("image_reliability") == 1
    assert "model" in names


def test_declare_returns_one_per_arg_name():
    nodes = ["yolo_node", "tracking_node", "detect_3d_node", "debug_node"]
    assert len(declare_param_arguments(nodes)) == len(param_arg_names(nodes))


def test_node_parameters_layers_over_file():
    params_file = LaunchConfiguration("params_file")
    context = FakeContext({"threshold": "0.5"})
    result = node_parameters(params_file, context, "yolo_node")
    assert result[0] is params_file
    assert result[1] == {"threshold": 0.5}


def test_node_parameters_layers_extra_files_before_overrides():
    params_file = LaunchConfiguration("params_file")
    context = FakeContext({"with_reid": "true"})
    result = node_parameters(params_file, context, "tracking_node", ["/t.yaml"])
    assert result[0] is params_file
    assert result[1] == "/t.yaml"
    assert result[2] == {"with_reid": True}


def test_tracker_params_file_resolves_and_validates(tmp_path, monkeypatch):
    import ament_index_python.packages as ament

    trackers = tmp_path / "config" / "trackers"
    trackers.mkdir(parents=True)
    for name in ("bytetrack", "botsort", "botsort_reid"):
        (trackers / f"{name}.yaml").write_text("")
    reid = trackers / "botsort_reid.yaml"

    monkeypatch.setattr(ament, "get_package_share_directory", lambda _: str(tmp_path))

    assert launch_params.tracker_params_file("botsort").endswith(
        os.path.join("config", "trackers", "botsort.yaml")
    )
    # empty selects the bytetrack default (matching the tracking node)
    assert launch_params.tracker_params_file("").endswith(
        os.path.join("config", "trackers", "bytetrack.yaml")
    )
    # a name resolves to its file; a filename or absolute path also works
    assert launch_params.tracker_params_file("botsort_reid") == str(reid)
    assert launch_params.tracker_params_file("botsort_reid.yaml") == str(reid)
    assert launch_params.tracker_params_file(str(reid)) == str(reid)
    with pytest.raises(RuntimeError, match="unknown tracker 'nope'"):
        launch_params.tracker_params_file("nope")


def test_pipeline_tracker_reads_the_tracking_block(tmp_path):
    path = tmp_path / "yolo.yaml"
    path.write_text(
        "/yolo/tracking_node:\n  ros__parameters:\n    tracker: botsort_reid\n"
    )
    assert launch_params.pipeline_tracker(str(path)) == "botsort_reid"
    # namespace-aware: another namespace has no block -> bytetrack default
    assert launch_params.pipeline_tracker(str(path), "other") == "bytetrack"
    # missing file / no value -> the bytetrack default
    assert launch_params.pipeline_tracker("") == "bytetrack"
    assert launch_params.pipeline_tracker(str(tmp_path / "nope.yaml")) == "bytetrack"


def test_load_params_mapping_unwraps_ros_parameters(tmp_path):
    path = tmp_path / "botsort.yaml"
    path.write_text(
        "/yolo/tracking_node:\n"
        "  ros__parameters:\n"
        "    tracker_type: botsort\n"
        "    track_high_thresh: 0.3\n"
    )
    assert launch_params.load_params_mapping(str(path)) == {
        "tracker_type": "botsort",
        "track_high_thresh": 0.3,
    }
    # A flat file (a dict of node blocks only) yields the first block's params.
    assert launch_params.load_params_mapping(str(tmp_path / "nope.yaml")) == {}


def test_unknown_node_raises():
    with pytest.raises(KeyError, match="nope"):
        build_overrides(FakeContext({}), "nope")


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_CPP_SOURCES = {
    "yolo_node": ["yolo_ros/src/node/yolo_node.cpp"],
    "tracking_node": [
        "yolo_ros/src/node/tracking_node.cpp",
        "yolo_ros/include/yolo_ros/tracking/byte_tracker.hpp",
        "yolo_ros/include/yolo_ros/tracking/bot_sort.hpp",
    ],
    "detect_3d_node": ["yolo_ros/src/node/detect_3d_node.cpp"],
    "debug_node": ["yolo_ros/src/node/debug_node.cpp"],
    "yolo_batch_node": ["yolo_ros/src/node/batch_node.cpp"],
}

_DECLARE_PARAMETER = re.compile(r'declare_parameter\s*(?:<[^(]*>)?\s*\(\s*"([^"]+)"')


@pytest.mark.parametrize("node", sorted(_CPP_SOURCES))
def test_catalogue_matches_declared_cpp_params(node):
    declared = set()
    for relative_path in _CPP_SOURCES[node]:
        with open(os.path.join(_REPO_ROOT, relative_path)) as source:
            declared.update(_DECLARE_PARAMETER.findall(source.read()))
    catalogue = {spec.name for spec in launch_params.NODE_PARAMS[node]}
    assert declared == catalogue, (
        f"{node}: C++-only={sorted(declared - catalogue)}, "
        f"catalogue-only={sorted(catalogue - declared)}"
    )
