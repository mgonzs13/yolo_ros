# Copyright (c) 2026 Miguel Ángel González Santamarta
# SPDX-License-Identifier: MIT

"""Thin launcher for the single yolo_node.

The pipeline (cameras, ordered plugins, plugin classes and all parameters)
lives in the YAML params file; this launch only sets the params file, the
namespace and the scalar plugin CLI overrides on top.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from yolo_bringup import launch_params


def _launch_setup(context, params_file, namespace):
    types = list(launch_params.PLUGINS)
    overrides = launch_params.build_all_overrides(context, types)
    node = Node(
        package="yolo_ros",
        executable="yolo_node",
        name="yolo_node",
        namespace=namespace,
        output="screen",
        parameters=[params_file, overrides],
    )
    return [node]


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    namespace = LaunchConfiguration("namespace")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=os.path.join(
                    get_package_share_directory("yolo_bringup"), "config", "yolo.yaml"
                ),
                description="Pipeline YAML: cameras, ordered plugins and params",
            ),
            DeclareLaunchArgument(
                "namespace", default_value="yolo", description="Namespace for the node"
            ),
            *launch_params.declare_param_arguments(list(launch_params.PLUGINS)),
            OpaqueFunction(function=_launch_setup, args=[params_file, namespace]),
        ]
    )
