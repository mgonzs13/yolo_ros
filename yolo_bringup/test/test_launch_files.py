# Copyright (c) 2026 Alejandro González Cantón
# Copyright (c) 2026 Miguel Ángel González Santamarta
# SPDX-License-Identifier: MIT

import importlib.util
import os

import pytest
import yaml
from launch import LaunchContext, LaunchDescription, Substitution
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from yolo_bringup import launch_params

LAUNCH_DIR = os.path.join(os.path.dirname(__file__), "..", "launch")
LAUNCH_FILES = {
    "yolo": "yolo.launch.py",
}


def _load(name):
    path = os.path.join(LAUNCH_DIR, LAUNCH_FILES[name])
    spec = importlib.util.spec_from_file_location(f"{name}_launch", path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unsub(value):
    """Evaluate the substitutions launch_ros wraps plain params values in."""
    if isinstance(value, Substitution):
        return value.perform(LaunchContext())
    if isinstance(value, dict):
        return {_unsub(key): _unsub(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        evaluated = [_unsub(item) for item in value]
        # A scalar string parameter normalizes to a 1-tuple of substitutions,
        # while a one-element array is a 1-tuple holding a list, so only the
        # former collapses here.
        if len(evaluated) == 1 and isinstance(value[0], Substitution):
            return evaluated[0]
        return type(value)(evaluated)
    return value


def _decoded(value):
    """Decode launch_ros' yaml.dumped string parameter values, recursively."""
    value = _unsub(value)
    if isinstance(value, str):
        return yaml.safe_load(value)
    if isinstance(value, (list, tuple)):
        return [_decoded(item) for item in value]
    return value


def _node_params(node):
    """Merge the Node's parameter dictionaries into decoded {param: value}."""
    merged = {}
    for entry in node._Node__parameters:
        if isinstance(entry, dict):
            merged.update(_unsub(entry))
    return {key: _decoded(value) for key, value in merged.items()}


class _FakeContext:
    def __init__(self, configs):
        self.launch_configurations = configs

    def perform_substitution(self, substitution):
        return substitution.perform(self)


def _description(name):
    return _load(name).generate_launch_description()


def _declared(name):
    description = _description(name)
    collect = getattr(
        description, "get_launch_arguments_with_include_launch_description_actions", None
    )
    if collect is not None:
        arguments = collect()
    else:
        # Foxy and Galactic only expose the arguments declared directly in the
        # description, which is all this launch file declares.
        arguments = [(argument, None) for argument in description.get_launch_arguments()]
    return {argument.name for argument, _ in arguments}


def _param_files(node):
    """Parameter file substitutions of a Node, across launch_ros versions."""
    entry = node._Node__parameters[0]
    if hasattr(entry, "param_file"):
        return list(entry.param_file)
    # Foxy and Galactic store (param_files, overrides) as a two-tuple.
    return list(entry)


def _single_node(module, configs, params_file=None):
    if params_file is None:
        params_file = LaunchConfiguration("params_file")
    nodes = module._launch_setup(
        _FakeContext(configs),
        params_file,
        LaunchConfiguration("namespace"),
    )
    assert len(nodes) == 1
    return nodes[0]


@pytest.fixture
def bringup_share(monkeypatch):
    import ament_index_python.packages as ament

    bringup_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    monkeypatch.setattr(ament, "get_package_share_directory", lambda _: bringup_dir)
    return bringup_dir


@pytest.mark.parametrize("name", sorted(LAUNCH_FILES))
def test_launch_description_builds(name, bringup_share):
    assert isinstance(_description(name), LaunchDescription)


def test_declares_only_params_file_namespace_and_scalar_args(bringup_share):
    expected = {"params_file", "namespace"} | set(
        launch_params.param_arg_names(list(launch_params.PLUGINS))
    )
    assert _declared("yolo") == expected


def test_removed_args_are_not_declared(bringup_share):
    declared = _declared("yolo")
    for name in (
        "use_tracking",
        "use_3d",
        "use_debug",
        "tracker",
        "image_topic",
        "depth_image_topic",
        "depth_info_topic",
        "input_image_topic",
        "input_depth_topic",
        "input_depth_info_topic",
    ):
        assert name not in declared


def test_base_defaults(bringup_share):
    defaults = {
        entity.name: _unsub(entity.default_value)
        for entity in _description("yolo").entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    assert defaults["namespace"] == "yolo"
    assert "yolo.yaml" in str(defaults["params_file"])


def test_yolo_launch_setup_creates_single_node(bringup_share):
    configs = {"namespace": "yolo"}
    node = _single_node(_load("yolo"), configs)
    assert node._Node__node_name == "yolo_node"
    assert node._Node__node_executable == "yolo_node"
    assert (
        _FakeContext(configs).perform_substitution(node._Node__node_namespace) == "yolo"
    )
    # No plugin classes are injected: the YAML is the source of truth.
    assert _node_params(node) == {}


def test_yolo_launch_layers_instance_prefixed_overrides(bringup_share):
    configs = {"namespace": "yolo", "threshold": "0.5"}
    params_file = LaunchConfiguration("params_file")
    node = _single_node(_load("yolo"), configs, params_file)
    assert _param_files(node)[0] is params_file
    params = _node_params(node)
    assert params == {"detection.threshold": pytest.approx(0.5)}
    assert "detection.plugin" not in params


def test_yolo_launch_has_no_remappings(bringup_share):
    node = _single_node(_load("yolo"), {"namespace": "yolo"})
    assert not node._Node__remappings
