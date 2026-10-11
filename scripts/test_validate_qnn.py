# Copyright (c) 2026 Miguel Ángel González Santamarta
# SPDX-License-Identifier: MIT

import json
import sys
import types

import cv2
import numpy as np
import onnx
import pytest
from onnx import helper

import compile_qnn_models as cqm
import validate_qnn as vq


def test_cosine_identical_is_one():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert vq.cosine(a, a) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert vq.cosine(a, b) == pytest.approx(0.0)


def test_cosine_zero_norm_branches():
    empty = np.zeros(0, dtype=np.float32)
    assert vq.cosine(empty, empty) == pytest.approx(1.0)
    assert vq.cosine(np.zeros(2), np.array([1.0, 0.0])) == pytest.approx(0.0)


def test_max_abs_difference():
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([1.5, 1.0], dtype=np.float32)
    assert vq.max_abs_diff(a, b) == pytest.approx(1.0)


def test_verdict_fails_below_threshold():
    assert vq.verdict(0.995, 0.99) is True
    assert vq.verdict(0.98, 0.99) is False


def test_verdict_fails_closed_for_nan():
    assert vq.verdict(float("nan"), 0.99) is False


def test_merge_manifest_records_accuracy(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"models": [{"model": "yolo26s_w8a8.ctx.onnx", "task": "detect"}]})
    )
    update = {"worst_cosine": 0.995}
    assert vq.merge_manifest(manifest, "yolo26s_w8a8.ctx.onnx", update) is True
    entry = json.loads(manifest.read_text())["models"][0]
    assert entry["task"] == "detect"
    assert entry["accuracy"]["worst_cosine"] == pytest.approx(0.995)


def test_merge_manifest_preserves_existing_accuracy_keys(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"models": [{"model": "m.onnx", "accuracy": {"worst_iou": 0.9}}]})
    )
    assert vq.merge_manifest(manifest, "m.onnx", {"worst_cosine": 0.5}) is True
    entry = json.loads(manifest.read_text())["models"][0]
    assert entry["accuracy"] == {"worst_iou": 0.9, "worst_cosine": 0.5}


def test_merge_manifest_replaces_non_dict_accuracy(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": [{"model": "m.onnx", "accuracy": "x"}]}))
    assert vq.merge_manifest(manifest, "m.onnx", {"worst_cosine": 0.5}) is True
    entry = json.loads(manifest.read_text())["models"][0]
    assert entry["accuracy"] == {"worst_cosine": 0.5}


def test_merge_manifest_ignores_unknown_model(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": [{"model": "other.onnx"}]}))
    assert vq.merge_manifest(manifest, "missing.onnx", {"worst_cosine": 0.5}) is False
    assert "accuracy" not in json.loads(manifest.read_text())["models"][0]


def test_merge_manifest_creates_parent_and_leaves_no_tmp(tmp_path):
    manifest = tmp_path / "nested" / "manifest.json"
    assert vq.merge_manifest(manifest, "m.onnx", {"worst_cosine": 0.5}) is False
    assert json.loads(manifest.read_text()) == {"models": []}
    assert list(manifest.parent.glob("*.tmp")) == []


def test_parse_args_rejects_out_of_range_min_cosine():
    base = ["--fp32", "a.onnx", "--qnn", "b.onnx", "--images", "images"]
    with pytest.raises(SystemExit):
        vq.parse_args([*base, "--min-cosine", "1.5"])
    with pytest.raises(SystemExit):
        vq.parse_args([*base, "--min-cosine", "-0.1"])
    args = vq.parse_args([*base, "--min-cosine", "0.95"])
    assert args.min_cosine == pytest.approx(0.95)
    assert args.input_color == "auto"


class _FakeNode:
    def __init__(self, name, shape=()):
        self.name = name
        self.shape = list(shape)


def _fake_onnxruntime(
    fp32_values,
    qnn_values,
    *,
    fp32_name="images",
    qnn_name="images",
    fp32_shape=(1, 3, 8, 8),
    fp32_inputs=None,
    qnn_inputs=None,
    fp32_metadata=None,
    qnn_providers=None,
    raise_on=None,
):
    class FakeSession:
        instances = []

        def __init__(self, path, providers=None, provider_options=None):
            providers = list(providers or [])
            self.is_qnn = "QNNExecutionProvider" in providers
            self.feeds = []
            if raise_on == ("qnn" if self.is_qnn else "fp32"):
                raise RuntimeError("session init failed")
            if self.is_qnn:
                self._providers = list(qnn_providers or ["QNNExecutionProvider"])
                inputs = qnn_inputs
                if inputs is None:
                    inputs = [_FakeNode(qnn_name, fp32_shape)]
                self._inputs = list(inputs)
                self._values = list(qnn_values)
            else:
                self._providers = ["CPUExecutionProvider"]
                inputs = fp32_inputs
                if inputs is None:
                    inputs = [_FakeNode(fp32_name, fp32_shape)]
                self._inputs = list(inputs)
                self._values = list(fp32_values)
            self._outputs = [
                _FakeNode(f"out{index}") for index in range(len(self._values))
            ]
            self._metadata = {} if self.is_qnn else dict(fp32_metadata or {})
            FakeSession.instances.append(self)

        def get_inputs(self):
            return self._inputs

        def get_outputs(self):
            return self._outputs

        def get_providers(self):
            return self._providers

        def get_modelmeta(self):
            return types.SimpleNamespace(custom_metadata_map=self._metadata)

        def run(self, output_names, feed):
            self.feeds.append(feed)
            return self._values

    module = types.SimpleNamespace(InferenceSession=FakeSession)
    return module, FakeSession


def _write_onnx(path, *, shape=(1, 3, 8, 8)):
    inputs = [
        helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, list(shape))
    ]
    outputs = [
        helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, list(shape))
    ]
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["output"])],
        "test",
        inputs,
        outputs,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(path))
    return path


def _sample_image():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30
    return image


def _write_image(path):
    image = _sample_image()
    assert cv2.imwrite(str(path), image)
    return image


def _argv(tmp_path, images, *extra):
    return [
        "validate_qnn.py",
        "--fp32",
        str(tmp_path / "yolo26s.onnx"),
        "--qnn",
        str(tmp_path / "yolo26s_w8a8.onnx"),
        "--images",
        str(images),
        *extra,
    ]


def _prepare_run(monkeypatch, tmp_path, *, extra=(), **fake_kwargs):
    _write_onnx(tmp_path / "yolo26s.onnx")
    _write_onnx(tmp_path / "yolo26s_w8a8.onnx")
    images = tmp_path / "images"
    images.mkdir()
    _write_image(images / "frame.png")
    values = [np.ones((1, 3, 8, 8), dtype=np.float32)]
    fake, session_cls = _fake_onnxruntime(
        fake_kwargs.pop("fp32_values", values),
        fake_kwargs.pop("qnn_values", values),
        **fake_kwargs,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, images, *extra))
    return session_cls


def test_main_records_accuracy_and_passes(monkeypatch, tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"models": [{"model": "yolo26s_w8a8.ctx.onnx", "task": "detect"}]})
    )
    sessions = _prepare_run(
        monkeypatch,
        tmp_path,
        extra=("--manifest", str(manifest), "--model-name", "yolo26s_w8a8.ctx.onnx"),
    )
    assert vq.main() == 0
    entry = json.loads(manifest.read_text())["models"][0]
    assert entry["accuracy"]["worst_cosine"] == pytest.approx(1.0)
    assert entry["accuracy"]["worst_max_abs"] == pytest.approx(0.0)
    latency = entry["accuracy"]["mean_qnn_ms"]
    assert isinstance(latency, float)
    assert latency >= 0.0
    printed = capsys.readouterr().out
    assert "PASS" in printed
    assert "qnn latency:" in printed
    assert len(sessions.instances) == 2


def test_main_feeds_each_session_its_own_input_name(monkeypatch, tmp_path):
    sessions = _prepare_run(
        monkeypatch, tmp_path, fp32_name="images", qnn_name="qnn_input"
    )
    assert vq.main() == 0
    fp32_session, qnn_session = sessions.instances
    assert list(fp32_session.feeds[0]) == ["images"]
    assert list(qnn_session.feeds[0]) == ["qnn_input"]


def test_main_fails_when_outputs_are_wiped(monkeypatch, tmp_path, capsys):
    zeros = [np.zeros((1, 3, 8, 8), dtype=np.float32)]
    _prepare_run(monkeypatch, tmp_path, qnn_values=zeros)
    assert vq.main() == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_fails_closed_on_nan(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": [{"model": "m.onnx"}]}))
    nan = [np.full((1, 3, 8, 8), np.nan, dtype=np.float32)]
    _prepare_run(
        monkeypatch,
        tmp_path,
        qnn_values=nan,
        extra=("--manifest", str(manifest), "--model-name", "m.onnx"),
    )
    assert vq.main() == 1
    text = manifest.read_text()

    def _reject_constant(name):
        raise ValueError(f"non-finite JSON constant {name}")

    entry = json.loads(text, parse_constant=_reject_constant)["models"][0]
    assert entry["accuracy"]["worst_cosine"] is None
    assert entry["accuracy"]["worst_max_abs"] == pytest.approx(0.0)


def test_main_rejects_cpu_fallback_for_qnn_session(monkeypatch, tmp_path, capsys):
    _prepare_run(monkeypatch, tmp_path, qnn_providers=["CPUExecutionProvider"])
    assert vq.main() == 2
    assert "QNN execution provider is not available" in capsys.readouterr().err


def test_main_rejects_when_no_outputs_compared(monkeypatch, tmp_path, capsys):
    _prepare_run(monkeypatch, tmp_path)
    (tmp_path / "images" / "frame.png").write_bytes(b"not an image")
    assert vq.main() == 2
    assert "no outputs were compared" in capsys.readouterr().err


def test_main_rejects_empty_images_dir(monkeypatch, tmp_path, capsys):
    _prepare_run(monkeypatch, tmp_path)
    (tmp_path / "images" / "frame.png").unlink()
    assert vq.main() == 2
    assert "no calibration images" in capsys.readouterr().err


def test_main_reports_missing_onnxruntime(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, tmp_path / "images"))
    assert vq.main() == 2
    assert "error: onnxruntime is not installed" in capsys.readouterr().err


def test_main_rejects_dynamic_input(monkeypatch, tmp_path, capsys):
    _prepare_run(monkeypatch, tmp_path, fp32_shape=(1, 3, "height", 8))
    assert vq.main() == 2
    assert "static NCHW" in capsys.readouterr().err


def test_main_rejects_mismatched_output_counts(monkeypatch, tmp_path, capsys):
    values = [np.ones((1, 3, 8, 8), dtype=np.float32)]
    _prepare_run(monkeypatch, tmp_path, qnn_values=[values[0], values[0]])
    assert vq.main() == 2
    assert "counts differ" in capsys.readouterr().err


def test_main_rejects_mismatched_input_counts(monkeypatch, tmp_path, capsys):
    _prepare_run(
        monkeypatch,
        tmp_path,
        qnn_inputs=[_FakeNode("a", (1, 3, 8, 8)), _FakeNode("b", (1, 3, 8, 8))],
    )
    assert vq.main() == 2
    assert "counts differ" in capsys.readouterr().err


def test_main_rejects_models_without_inputs(monkeypatch, tmp_path, capsys):
    _prepare_run(monkeypatch, tmp_path, fp32_inputs=[], qnn_inputs=[])
    assert vq.main() == 2
    assert "must each declare an input" in capsys.readouterr().err


def test_main_reports_session_load_failure(monkeypatch, tmp_path, capsys):
    _prepare_run(monkeypatch, tmp_path, raise_on="qnn")
    assert vq.main() == 2
    err = capsys.readouterr().err
    assert "could not load" in err
    assert "yolo26s_w8a8.onnx" in err


def test_main_warns_when_model_name_is_absent(monkeypatch, tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": [{"model": "other.onnx"}]}))
    _prepare_run(
        monkeypatch,
        tmp_path,
        extra=("--manifest", str(manifest), "--model-name", "yolo26s_w8a8.ctx.onnx"),
    )
    assert vq.main() == 0
    assert "warning:" in capsys.readouterr().err


def test_main_auto_input_color_follows_metadata(monkeypatch, tmp_path):
    sessions = _prepare_run(monkeypatch, tmp_path, fp32_metadata={"input_color": "bgr"})
    assert vq.main() == 0
    expected = cqm.to_tensor(cqm.letterbox(_sample_image(), 8, 8), rgb=False)
    assert np.array_equal(sessions.instances[0].feeds[0]["images"], expected)


def test_main_explicit_input_color_overrides_metadata(monkeypatch, tmp_path):
    sessions = _prepare_run(
        monkeypatch,
        tmp_path,
        fp32_metadata={"input_color": "bgr"},
        extra=("--input-color", "rgb"),
    )
    assert vq.main() == 0
    expected = cqm.to_tensor(cqm.letterbox(_sample_image(), 8, 8), rgb=True)
    assert np.array_equal(sessions.instances[0].feeds[0]["images"], expected)


def test_resolve_input_color_requested_wins_and_metadata_fallback():
    assert vq.resolve_input_color("rgb", {"input_color": "bgr"}) is True
    assert vq.resolve_input_color("bgr", {"input_color": "rgb"}) is False
    assert vq.resolve_input_color("auto", {"input_color": "bgr"}) is False
    assert vq.resolve_input_color("auto", {"input_color": "BGR"}) is False
    assert vq.resolve_input_color("auto", {}) is True
