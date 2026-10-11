# Copyright (c) 2026 Miguel Ángel González Santamarta
# SPDX-License-Identifier: MIT

import hashlib
import json
import sys
import types
import zipfile
from pathlib import Path

import cv2
import numpy as np
import onnx
import pytest
from onnx import helper

import compile_qnn_models as cqm


def test_model_filename_for_every_task():
    assert cqm.model_filename("s", "detect") == "yolo26s.onnx"
    assert cqm.model_filename("n", "segment") == "yolo26n-seg.onnx"
    assert cqm.model_filename("m", "pose") == "yolo26m-pose.onnx"
    assert cqm.model_filename("l", "obb") == "yolo26l-obb.onnx"
    assert cqm.model_filename("x", "classify") == "yolo26x-cls.onnx"


def test_model_filename_rejects_unknown_size_and_task():
    with pytest.raises(ValueError, match="size"):
        cqm.model_filename("q", "detect")
    with pytest.raises(ValueError, match="task"):
        cqm.model_filename("s", "depth")


def test_letterbox_geometry_landscape():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    out = cqm.letterbox(image, 640, 640)
    assert out.shape == (640, 640, 3)
    # ratio = min(640/640, 640/480) = 1.0 -> 640x480, padded to 640x640
    assert np.all(out[0] == 114)  # top pad
    assert np.all(out[-1] == 114)  # bottom pad
    assert np.all(out[80] == 0)  # image row


def test_letterbox_geometry_portrait():
    image = np.zeros((640, 320, 3), dtype=np.uint8)
    out = cqm.letterbox(image, 640, 640)
    assert out.shape == (640, 640, 3)
    # ratio = min(640/320, 640/640) = 1.0 -> 320x640, centered horizontally
    assert np.all(out[:, 0] == 114)
    assert np.all(out[:, -1] == 114)
    assert np.all(out[:, 160] == 0)


def test_letterbox_extreme_aspect_ratio():
    image = np.zeros((1, 10000, 3), dtype=np.uint8)
    out = cqm.letterbox(image, 640, 640)
    assert out.shape == (640, 640, 3)


def test_to_tensor_matches_cpp_contract():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[..., 0] = 255  # blue (BGR)
    tensor = cqm.to_tensor(image, rgb=True)
    assert tensor.shape == (1, 3, 2, 2)
    assert tensor.dtype == np.float32
    assert tensor[0, 2, 0, 0] == pytest.approx(1.0)  # BGR blue lands last after BGR->RGB


def test_collect_images_evenly_samples(tmp_path):
    names = [f"img{i:03d}.jpg" for i in range(10)]
    for name in names:
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("ignore")
    selected = cqm.collect_images(tmp_path, limit=4)
    assert len(selected) == 4
    assert selected[0].name == "img000.jpg"
    assert selected[-1].name == "img009.jpg"


def test_collect_images_rejects_empty_dir(tmp_path):
    with pytest.raises(RuntimeError, match="no calibration images"):
        cqm.collect_images(tmp_path, limit=0)


def test_collect_images_limit_one_samples_first(tmp_path):
    for i in range(10):
        (tmp_path / f"img{i:03d}.jpg").write_bytes(b"x")
    selected = cqm.collect_images(tmp_path, limit=1)
    assert [path.name for path in selected] == ["img000.jpg"]


def test_collect_images_limit_above_count_keeps_all(tmp_path):
    for i in range(3):
        (tmp_path / f"img{i:03d}.jpg").write_bytes(b"x")
    selected = cqm.collect_images(tmp_path, limit=10)
    assert [path.name for path in selected] == [
        "img000.jpg",
        "img001.jpg",
        "img002.jpg",
    ]


def test_collect_images_rejects_negative_limit(tmp_path):
    (tmp_path / "img.jpg").write_bytes(b"x")
    with pytest.raises(ValueError, match="limit"):
        cqm.collect_images(tmp_path, limit=-1)


def test_collect_images_finds_nested_images(tmp_path):
    nested = tmp_path / "session" / "cam0"
    nested.mkdir(parents=True)
    (nested / "frame.jpg").write_bytes(b"x")
    (tmp_path / "top.png").write_bytes(b"x")
    selected = cqm.collect_images(tmp_path, limit=0)
    assert [path.name for path in selected] == ["frame.jpg", "top.png"]


def test_select_device_name_matches_soc():
    names = ["Samsung Galaxy S24", "Dragonwing IQ-8275", "Snapdragon X Elite"]
    assert cqm.select_device_name(names, "qcs8275") == "Dragonwing IQ-8275"
    assert cqm.select_device_name(names, "iq-8275") == "Dragonwing IQ-8275"


def test_select_device_name_matches_dashed_soc():
    names = ["Samsung Galaxy S24", "Dragonwing IQ-8275"]
    assert cqm.select_device_name(names, "qcs-8275") == "Dragonwing IQ-8275"


def test_select_device_name_rejects_empty_soc():
    with pytest.raises(RuntimeError, match="non-empty"):
        cqm.select_device_name(["Dragonwing IQ-8275"], "")


def test_select_device_name_rejects_empty_candidates():
    with pytest.raises(RuntimeError, match="available: none"):
        cqm.select_device_name([], "qcs8275")


def test_select_device_name_reports_candidates():
    with pytest.raises(RuntimeError, match="Dragonwing IQ-8275"):
        cqm.select_device_name(["Samsung Galaxy S24"], "qcs8275")


def test_manifest_roundtrip_and_sha(tmp_path):
    model = tmp_path / "yolo26s_w8a8.ctx.onnx"
    model.write_bytes(b"ctx")
    entry = cqm.manifest_entry(
        model=model,
        task="detect",
        soc="qcs8275",
        qairt_version="latest",
        fp32_model="yolo26s.onnx",
    )
    assert entry["sha256"] == hashlib.sha256(b"ctx").hexdigest()
    assert entry["task"] == "detect"
    manifest = tmp_path / "manifest.json"
    cqm.write_manifest(manifest, [entry])
    assert json.loads(manifest.read_text())["models"][0]["model"] == model.name


def test_write_manifest_creates_parent_directory(tmp_path):
    manifest = tmp_path / "nested" / "deeper" / "manifest.json"
    cqm.write_manifest(manifest, [])
    assert json.loads(manifest.read_text()) == {"models": []}


def _tiny_onnx_bytes() -> bytes:
    inputs = [
        helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 8, 8])
    ]
    outputs = [
        helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 3, 8, 8])
    ]
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["output"])], "tiny", inputs, outputs
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model.SerializeToString()


def _write_tiny_onnx(path: Path) -> None:
    Path(path).write_bytes(_tiny_onnx_bytes())


class FakeModel:
    def __init__(self, name, payload=None, suffix=""):
        self.name = name
        self.payload = payload
        self.suffix = suffix

    def download(self, path):
        target = Path(str(path) + self.suffix)
        if self.payload is None:
            target.write_bytes(b"compiled")
        else:
            self.payload(target)
        return target


class FakeJob:
    def __init__(self, target):
        self._target = target

    def get_target_model(self):
        return self._target


class FakeClient:
    def __init__(self):
        self.calls = []

    def upload_model(self, path):
        self.calls.append(("upload", str(path)))
        return "uploaded.onnx"

    def submit_quantize_job(self, model, calibration_data, **kwargs):
        self.calls.append(("quantize", model, kwargs))
        return FakeJob(FakeModel("qdq.onnx", payload=_write_tiny_onnx))

    def submit_compile_and_link_jobs(self, models, device, **kwargs):
        self.calls.append(("compile_and_link", models, device, kwargs))
        return FakeJob(FakeModel("compiled.bin")), FakeJob(FakeModel("context.bin"))

    def submit_compile_job(self, model, device, **kwargs):
        self.calls.append(("compile", model, device, kwargs))
        return FakeJob(FakeModel("ctx.onnx", payload=_write_tiny_onnx))


class FakeHub:
    Device = staticmethod(lambda name: name)
    QuantizeDtype = type("Q", (), {"INT8": "int8"})


def test_compile_model_runs_quantize_compile_wrap(tmp_path):
    calibration = {"images": [np.zeros((1, 3, 640, 640), dtype=np.float32)]}
    client = FakeClient()
    out = cqm.compile_model(
        client=client,
        hub=FakeHub,
        onnx_path=tmp_path / "yolo26s.onnx",
        calibration=calibration,
        device_name=cqm.DEFAULT_DEVICE_NAME,
        output_dir=tmp_path,
        qairt_version="latest",
    )
    assert out["qnn"].name == "yolo26s_w8a8.ctx.onnx"
    assert out["qdq"].name == "yolo26s_w8a8.onnx"
    assert out["deploy"] == out["qnn"]
    assert onnx.load(str(tmp_path / "yolo26s_w8a8.ctx.onnx")).graph.name == "tiny"

    assert client.calls[0] == ("upload", str(tmp_path / "yolo26s.onnx"))

    kind, model, kwargs = client.calls[1]
    assert kind == "quantize"
    assert model == "uploaded.onnx"
    assert kwargs["weights_dtype"] == "int8"
    assert kwargs["activations_dtype"] == "int8"

    kind, model, device, kwargs = client.calls[2]
    assert kind == "compile_and_link"
    assert model.name == "qdq.onnx"
    assert device == cqm.DEFAULT_DEVICE_NAME
    assert kwargs["compile_options"] == ["--qairt_version latest"]
    assert kwargs["link_options"] == "--qairt_version latest"

    kind, model, device, kwargs = client.calls[3]
    assert kind == "compile"
    assert model.name == "context.bin"
    assert device == cqm.DEFAULT_DEVICE_NAME
    assert "options" not in kwargs

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "yolo26s_w8a8.ctx.onnx",
        "yolo26s_w8a8.onnx",
    ]


def test_compile_model_deploys_when_download_appends_server_suffix(tmp_path):
    class SuffixedClient(FakeClient):
        def submit_compile_job(self, model, device, **kwargs):
            self.calls.append(("compile", model, device, kwargs))
            return FakeJob(
                FakeModel("ctx.onnx", payload=_write_tiny_onnx, suffix=".ctx.onnx")
            )

    calibration = {"images": [np.zeros((1, 3, 640, 640), dtype=np.float32)]}
    out = cqm.compile_model(
        client=SuffixedClient(),
        hub=FakeHub,
        onnx_path=tmp_path / "yolo26s.onnx",
        calibration=calibration,
        device_name=cqm.DEFAULT_DEVICE_NAME,
        output_dir=tmp_path,
        qairt_version="latest",
    )

    assert out["qnn"] == tmp_path / "yolo26s_w8a8.ctx.onnx"
    assert out["deploy"] == out["qnn"]
    assert onnx.load(str(out["qnn"])).graph.name == "tiny"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "yolo26s_w8a8.ctx.onnx",
        "yolo26s_w8a8.onnx",
    ]


def test_deploy_precompiled_artifact_plain_onnx(tmp_path):
    staged = tmp_path / "staged.onnx"
    staged.write_bytes(_tiny_onnx_bytes())
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    deployed = cqm.deploy_precompiled_artifact(staged, output_dir, "yolo26s")

    assert deployed == output_dir / "yolo26s_w8a8.ctx.onnx"
    assert onnx.load(str(deployed)).graph.name == "tiny"


def test_deploy_precompiled_artifact_rejects_zip_with_context_bin(tmp_path):
    staged = tmp_path / "download.zip"
    with zipfile.ZipFile(staged, "w") as archive:
        archive.writestr("model.onnx", _tiny_onnx_bytes())
        archive.writestr("model.bin", b"context-binary")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    assert cqm.deploy_precompiled_artifact(staged, output_dir, "yolo26s") is None
    assert list(output_dir.iterdir()) == []


def test_deploy_precompiled_artifact_standalone_zip_deploys(tmp_path):
    staged = tmp_path / "download.zip"
    with zipfile.ZipFile(staged, "w") as archive:
        archive.writestr("model.onnx", _tiny_onnx_bytes())
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    deployed = cqm.deploy_precompiled_artifact(staged, output_dir, "yolo26s")

    assert deployed == output_dir / "yolo26s_w8a8.ctx.onnx"
    assert onnx.load(str(deployed)).graph.name == "tiny"
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "yolo26s_w8a8.ctx.onnx",
    ]


def test_deploy_precompiled_artifact_rejects_unloadable_bytes(tmp_path):
    staged = tmp_path / "staged.onnx"
    staged.write_bytes(b"compiled")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    assert cqm.deploy_precompiled_artifact(staged, output_dir, "yolo26s") is None
    assert list(output_dir.iterdir()) == []


def test_deploy_precompiled_artifact_rejects_multiple_onnx(tmp_path):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "model.onnx").write_bytes(_tiny_onnx_bytes())
    (staged / "other.onnx").write_bytes(_tiny_onnx_bytes())
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    assert cqm.deploy_precompiled_artifact(staged, output_dir, "yolo26s") is None
    assert list(output_dir.iterdir()) == []


def test_compile_model_falls_back_to_qdq_when_wrap_unloadable(tmp_path, capsys):
    class RawWrapClient(FakeClient):
        def submit_compile_job(self, model, device, **kwargs):
            self.calls.append(("compile", model, device, kwargs))
            return FakeJob(
                FakeModel("ctx.onnx", payload=lambda path: path.write_bytes(b"compiled"))
            )

    calibration = {"images": [np.zeros((1, 3, 640, 640), dtype=np.float32)]}
    out = cqm.compile_model(
        client=RawWrapClient(),
        hub=FakeHub,
        onnx_path=tmp_path / "yolo26s.onnx",
        calibration=calibration,
        device_name=cqm.DEFAULT_DEVICE_NAME,
        output_dir=tmp_path,
        qairt_version="latest",
    )

    assert out["qnn"] is None
    assert out["deploy"] == out["qdq"]
    assert onnx.load(str(out["qdq"])).graph.name == "tiny"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["yolo26s_w8a8.onnx"]
    err = capsys.readouterr().err
    assert "warning:" in err
    assert "self-caching QDQ" in err


def test_compile_model_cleans_extracted_zip_staging_on_fallback(tmp_path, capsys):
    def write_context_zip(path):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("model.onnx", _tiny_onnx_bytes())
            archive.writestr("model.bin", b"context-binary")

    class ZippedClient(FakeClient):
        def submit_compile_job(self, model, device, **kwargs):
            self.calls.append(("compile", model, device, kwargs))
            return FakeJob(FakeModel("ctx.zip", payload=write_context_zip, suffix=".zip"))

    calibration = {"images": [np.zeros((1, 3, 640, 640), dtype=np.float32)]}
    out = cqm.compile_model(
        client=ZippedClient(),
        hub=FakeHub,
        onnx_path=tmp_path / "yolo26s.onnx",
        calibration=calibration,
        device_name=cqm.DEFAULT_DEVICE_NAME,
        output_dir=tmp_path,
        qairt_version="latest",
    )

    assert out["qnn"] is None
    assert out["deploy"] == out["qdq"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["yolo26s_w8a8.onnx"]
    assert "self-caching QDQ" in capsys.readouterr().err


def test_compile_model_raises_when_link_job_missing(tmp_path):
    class NoLinkClient(FakeClient):
        def submit_compile_and_link_jobs(self, models, device, **kwargs):
            self.calls.append(("compile_and_link", models, device, kwargs))
            return FakeJob(FakeModel("compiled.bin")), None

    with pytest.raises(RuntimeError, match="no link job"):
        cqm.compile_model(
            client=NoLinkClient(),
            hub=FakeHub,
            onnx_path=tmp_path / "yolo26s.onnx",
            calibration={"images": []},
            device_name=cqm.DEFAULT_DEVICE_NAME,
            output_dir=tmp_path,
            qairt_version="latest",
        )


def test_main_dry_run_prints_plan_without_hub(monkeypatch, tmp_path, capsys):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "yolo26s.onnx").write_bytes(b"onnx")
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.jpg").write_bytes(b"jpg")
    monkeypatch.delitem(sys.modules, "qai_hub", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compile_qnn_models.py",
            "--models-dir",
            str(tmp_path / "models"),
            "--images",
            str(tmp_path / "images"),
            "--tasks",
            "detect",
            "--size",
            "s",
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ],
    )
    assert cqm.main() == 0
    printed = capsys.readouterr().out
    assert "yolo26s_w8a8.ctx.onnx" in printed
    assert "qai_hub" not in sys.modules


def test_main_uploads_when_hf_repo_given(monkeypatch, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "yolo26s.onnx").write_bytes(b"onnx")
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"jpg")

    client = FakeClient()
    client.get_devices = lambda: [types.SimpleNamespace(name="Dragonwing IQ-8275")]
    fake_hub = types.SimpleNamespace(
        Client=lambda: client,
        Device=lambda name: name,
        QuantizeDtype=FakeHub.QuantizeDtype,
    )
    monkeypatch.setitem(sys.modules, "qai_hub", fake_hub)
    monkeypatch.setattr(cqm, "build_calibration", lambda *a, **k: {"images": []})

    uploaded = {}
    fake_hf = types.SimpleNamespace(
        HfApi=lambda: types.SimpleNamespace(
            upload_folder=lambda **kwargs: uploaded.update(kwargs)
        )
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compile_qnn_models.py",
            "--models-dir",
            str(models),
            "--images",
            str(images),
            "--tasks",
            "detect",
            "--size",
            "s",
            "--output-dir",
            str(tmp_path / "out"),
            "--hf-repo",
            "org/repo",
        ],
    )
    assert cqm.main() == 0
    assert uploaded["repo_id"] == "org/repo"
    assert uploaded["path_in_repo"] == "qnn"
    assert uploaded["repo_type"] == "model"
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["models"][0]["device"] == "Dragonwing IQ-8275"
    assert manifest["models"][0]["model"] == "yolo26s_w8a8.ctx.onnx"
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        "manifest.json",
        "yolo26s_w8a8.ctx.onnx",
        "yolo26s_w8a8.onnx",
    ]


def _write_identity_onnx(
    path: Path,
    shape=(1, 3, 64, 64),
    elem_type: int = onnx.TensorProto.FLOAT,
    metadata: dict | None = None,
) -> Path:
    inputs = [helper.make_tensor_value_info("images", elem_type, list(shape))]
    outputs = [helper.make_tensor_value_info("output", elem_type, list(shape))]
    node = helper.make_node("Identity", ["images"], ["output"])
    graph = helper.make_graph([node], "test", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    for key, value in (metadata or {}).items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, str(path))
    return path


def _write_calibration_image(path: Path) -> np.ndarray:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return image


def test_build_calibration_reads_static_float_input(tmp_path):
    onnx_path = _write_identity_onnx(tmp_path / "yolo26s.onnx")
    image = _write_calibration_image(tmp_path / "frame.png")

    calibration = cqm.build_calibration(onnx_path, [tmp_path / "frame.png"])

    assert list(calibration) == ["images"]
    tensor = calibration["images"][0]
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == np.float32
    expected = cqm.to_tensor(cqm.letterbox(image, 64, 64), rgb=True)
    assert np.array_equal(tensor, expected)


def test_build_calibration_honors_bgr_metadata(tmp_path):
    onnx_path = _write_identity_onnx(
        tmp_path / "yolo26s.onnx", metadata={"input_color": "bgr"}
    )
    image = _write_calibration_image(tmp_path / "frame.png")

    calibration = cqm.build_calibration(onnx_path, [tmp_path / "frame.png"])

    expected = cqm.to_tensor(cqm.letterbox(image, 64, 64), rgb=False)
    assert np.array_equal(calibration["images"][0], expected)


def test_build_calibration_rejects_batch_two(tmp_path):
    onnx_path = _write_identity_onnx(tmp_path / "yolo26s.onnx", shape=(2, 3, 64, 64))
    with pytest.raises(RuntimeError, match="batch-1"):
        cqm.build_calibration(onnx_path, [])


def test_build_calibration_rejects_non_float_input(tmp_path):
    onnx_path = _write_identity_onnx(
        tmp_path / "yolo26s.onnx", elem_type=onnx.TensorProto.INT8
    )
    with pytest.raises(RuntimeError, match="float32"):
        cqm.build_calibration(onnx_path, [])


def test_build_calibration_rejects_dynamic_input(tmp_path):
    onnx_path = _write_identity_onnx(tmp_path / "yolo26s.onnx", shape=("N", 3, 64, 64))
    with pytest.raises(RuntimeError, match="static batch-1"):
        cqm.build_calibration(onnx_path, [])


def test_manifest_entry_records_device(tmp_path):
    model = tmp_path / "yolo26s_w8a8.ctx.onnx"
    model.write_bytes(b"ctx")
    entry = cqm.manifest_entry(
        model=model,
        task="detect",
        soc="qcs8275",
        qairt_version="latest",
        fp32_model="yolo26s.onnx",
        device="Dragonwing IQ-8275",
    )
    assert entry["device"] == "Dragonwing IQ-8275"


def _main_argv(tmp_path: Path, models: Path, images: Path, *extra: str) -> list[str]:
    return [
        "compile_qnn_models.py",
        "--models-dir",
        str(models),
        "--images",
        str(images),
        "--tasks",
        "detect",
        "--size",
        "s",
        "--output-dir",
        str(tmp_path / "out"),
        *extra,
    ]


def test_main_rejects_non_directory_models_dir(monkeypatch, tmp_path, capsys):
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"jpg")
    monkeypatch.setattr(sys, "argv", _main_argv(tmp_path, tmp_path / "missing", images))
    assert cqm.main() == 2
    assert "not a directory" in capsys.readouterr().err


def test_main_rejects_non_directory_images(monkeypatch, tmp_path, capsys):
    models = tmp_path / "models"
    models.mkdir()
    (models / "yolo26s.onnx").write_bytes(b"onnx")
    monkeypatch.setattr(sys, "argv", _main_argv(tmp_path, models, tmp_path / "missing"))
    assert cqm.main() == 2
    assert "not a directory" in capsys.readouterr().err


def test_main_rejects_missing_model(monkeypatch, tmp_path, capsys):
    models = tmp_path / "models"
    models.mkdir()
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"jpg")
    monkeypatch.setattr(sys, "argv", _main_argv(tmp_path, models, images))
    assert cqm.main() == 2
    assert "missing model" in capsys.readouterr().err


def test_main_rejects_negative_limit(monkeypatch, tmp_path, capsys):
    models = tmp_path / "models"
    models.mkdir()
    (models / "yolo26s.onnx").write_bytes(b"onnx")
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"jpg")
    monkeypatch.setattr(
        sys, "argv", _main_argv(tmp_path, models, images, "--limit", "-1")
    )
    assert cqm.main() == 2
    assert "limit" in capsys.readouterr().err


def test_main_rejects_empty_images_dir(monkeypatch, tmp_path, capsys):
    models = tmp_path / "models"
    models.mkdir()
    (models / "yolo26s.onnx").write_bytes(b"onnx")
    images = tmp_path / "images"
    images.mkdir()
    monkeypatch.setattr(sys, "argv", _main_argv(tmp_path, models, images))
    assert cqm.main() == 2
    assert "no calibration images" in capsys.readouterr().err


def test_main_rejects_unknown_device_name(monkeypatch, tmp_path, capsys):
    models = tmp_path / "models"
    models.mkdir()
    (models / "yolo26s.onnx").write_bytes(b"onnx")
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"jpg")

    client = FakeClient()
    client.get_devices = lambda: [types.SimpleNamespace(name="Dragonwing IQ-8275")]
    fake_hub = types.SimpleNamespace(
        Client=lambda: client,
        Device=lambda name: name,
        QuantizeDtype=FakeHub.QuantizeDtype,
    )
    monkeypatch.setitem(sys.modules, "qai_hub", fake_hub)
    monkeypatch.setattr(
        sys,
        "argv",
        _main_argv(tmp_path, models, images, "--device-name", "No Such Device"),
    )
    assert cqm.main() == 2
    captured = capsys.readouterr()
    assert "--device-name" in captured.err
    assert "is not available" in captured.err


def test_main_deduplicates_tasks_keeping_order(monkeypatch, tmp_path, capsys):
    models = tmp_path / "models"
    models.mkdir()
    (models / "yolo26s.onnx").write_bytes(b"onnx")
    (models / "yolo26s-seg.onnx").write_bytes(b"onnx")
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"jpg")
    monkeypatch.setattr(
        sys,
        "argv",
        _main_argv(
            tmp_path,
            models,
            images,
            "--tasks",
            "segment",
            "detect",
            "segment",
            "--dry-run",
        ),
    )
    assert cqm.main() == 0
    lines = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("plan:")
    ]
    assert len(lines) == 2
    assert "yolo26s-seg.onnx" in lines[0]
    assert "yolo26s.onnx ->" in lines[1]
