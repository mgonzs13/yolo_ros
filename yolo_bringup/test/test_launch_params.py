# Copyright (C) 2026 Alejandro González Cantón
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
    assert overrides["detection.model.path"] == "/x.onnx"
    assert overrides["detection.inference.threshold"] == pytest.approx(0.5)
    assert overrides["detection.inference.max_det"] == 300
    assert overrides["detection.model.force_download"] is False


def test_params_use_namespaced_names():
    detection = {spec.name for spec in launch_params.PLUGIN_PARAMS[DET]}
    assert {
        "model.type",
        "model.path",
        "model.repo",
        "model.filename",
        "model.cache_dir",
        "model.force_download",
        "inference.threshold",
        "inference.iou",
        "inference.max_det",
        "inference.max_fps",
        "inference.top_k",
        "input.width",
        "input.height",
        "batch.max_size",
        "batch.pad_to",
        "cpu.n_threads",
        "trt.fp16_enable",
        "trt.engine_cache_enable",
        "trt.engine_cache_path",
        "cuda.graph_enable",
        "qnn.htp_performance_mode",
        "qnn.htp_arch",
        "qnn.soc_model",
        "qnn.backend_path",
        "qnn.offload_graph_io_quantization",
        "qnn.enable_htp_fp16_precision",
        "qnn.context_cache_enable",
        "qnn.context_cache_path",
    } <= detection
    assert "model_path" not in detection
    assert "threshold" not in detection
    assert "qnn_htp_arch" not in detection

    tracking = {spec.name for spec in launch_params.PLUGIN_PARAMS[TRACK]}
    assert {
        "track.high_thresh",
        "track.low_thresh",
        "track.new_thresh",
        "track.buffer",
        "track.match_thresh",
        "track.fuse_score",
        "gmc.method",
        "gmc.downscale",
        "reid.enable",
        "reid.model",
        "reid.proximity_thresh",
        "reid.appearance_thresh",
        "reid.provider",
        "reid.device",
    } <= tracking
    assert "track_high_thresh" not in tracking
    assert "with_reid" not in tracking

    detection3d = {spec.name for spec in launch_params.PLUGIN_PARAMS[D3D]}
    assert {
        "target_frame",
        "depth.units_divisor",
        "orientation.enable",
        "orientation.min_seg_points",
    } == detection3d


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
        assert build_overrides(context, DET)["detection.model.force_download"] is expected


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
    assert build_overrides(context, TRACK)["tracking.reid.provider"] == "cpu"
    assert build_overrides(context, DET)["detection.device"] == "cpu:0"
    assert build_overrides(context, TRACK)["tracking.reid.device"] == "cpu:0"
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
    assert overrides["detection.inference.threshold"] == pytest.approx(0.5)
    assert overrides["tracking.tracker_type"] == "botsort"
    assert overrides["debug.marker_lifetime"] == pytest.approx(1.5)
    assert "detection.plugin" not in overrides
    assert "tracking.plugin" not in overrides
    assert "debug.plugin" not in overrides


def test_debug_compressed_overrides():
    context = FakeContext(
        {
            "debug_compressed_enable": "false",
            "debug_compressed_quality": "60",
        }
    )
    overrides = build_overrides(context, DBG)
    assert overrides["debug.compressed.enable"] is False
    assert overrides["debug.compressed.quality"] == 60


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


def test_qnn_overrides_are_detection_only():
    context = FakeContext(
        {
            "qnn_htp_performance_mode": "sustained_high_performance",
            "qnn_htp_arch": "73",
            "qnn_soc_model": "79",
            "qnn_backend_path": "/opt/qairt/lib/libQnnHtp.so",
            "qnn_offload_graph_io_quantization": "false",
            "qnn_enable_htp_fp16_precision": "true",
            "qnn_context_cache_enable": "false",
            "qnn_context_cache_path": "/tmp/qnn",
        }
    )
    overrides = build_overrides(context, DET)
    assert overrides["detection.qnn.htp_performance_mode"] == "sustained_high_performance"
    assert overrides["detection.qnn.htp_arch"] == "73"
    assert overrides["detection.qnn.soc_model"] == "79"
    assert overrides["detection.qnn.backend_path"] == "/opt/qairt/lib/libQnnHtp.so"
    assert overrides["detection.qnn.offload_graph_io_quantization"] is False
    assert overrides["detection.qnn.enable_htp_fp16_precision"] is True
    assert overrides["detection.qnn.context_cache_enable"] is False
    assert overrides["detection.qnn.context_cache_path"] == "/tmp/qnn"
    assert build_overrides(context, TRACK) == {}
    assert build_overrides(context, D3D) == {}
    assert build_overrides(context, DBG) == {}
