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
