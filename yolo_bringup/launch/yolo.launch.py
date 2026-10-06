# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT


import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from yolo_bringup.launch_params import (
    declare_param_arguments,
    node_parameters,
    pipeline_tracker,
    tracker_params_file,
)

NODES = ("yolo_node", "tracking_node", "detect_3d_node", "debug_node")


def _launch_setup(context, params_file, namespace, use_tracking, use_3d, use_debug):
    # Per-tracker knobs live in config/trackers/<name>.yaml and are layered
    # over the pipeline config for the tracking node only. The pipeline YAML's
    # `tracker` selector picks the file; the `tracker` argument overrides it (by
    # name or path). Each config file sets the real `tracker_type` it runs.
    tracker_override = context.launch_configurations.get("tracker", "")
    if tracker_override:
        tracker_file = tracker_params_file(tracker_override)
    else:
        tracker_file = tracker_params_file(
            pipeline_tracker(
                context.launch_configurations.get("params_file", ""),
                context.launch_configurations.get("namespace", "yolo"),
            )
        )

    # C++ inference node (ONNX Runtime). Every parameter comes from the YAML
    # params file, optionally overridden by the matching launch argument.
    yolo_node_cmd = Node(
        package="yolo_ros",
        executable="yolo_node",
        name="yolo_node",
        namespace=namespace,
        parameters=node_parameters(params_file, context, "yolo_node"),
    )

    # C++ tracking node (ByteTrack, Kalman-filtered ids on `tracking`).
    tracking_node_cmd = Node(
        package="yolo_ros",
        executable="tracking_node",
        name="tracking_node",
        namespace=namespace,
        parameters=node_parameters(params_file, context, "tracking_node", [tracker_file]),
        condition=IfCondition(use_tracking),
    )

    # C++ 3D detection node (lifts the 2D detections using the depth image).
    detect_3d_node_cmd = Node(
        package="yolo_ros",
        executable="detect_3d_node",
        name="detect_3d_node",
        namespace=namespace,
        parameters=node_parameters(params_file, context, "detect_3d_node"),
        condition=IfCondition(use_3d),
    )

    # C++ debug node (visualizes detections/tracks + RViz 3D markers).
    debug_node_cmd = Node(
        package="yolo_ros",
        executable="debug_node",
        name="debug_node",
        namespace=namespace,
        parameters=node_parameters(params_file, context, "debug_node"),
        condition=IfCondition(use_debug),
    )

    return [yolo_node_cmd, tracking_node_cmd, detect_3d_node_cmd, debug_node_cmd]


def generate_launch_description():

    use_tracking = LaunchConfiguration("use_tracking")
    use_tracking_cmd = DeclareLaunchArgument(
        "use_tracking",
        default_value="True",
        description="Whether to enable the ByteTrack tracking node (True/False)",
    )

    use_3d = LaunchConfiguration("use_3d")
    use_3d_cmd = DeclareLaunchArgument(
        "use_3d",
        default_value="False",
        description="Whether to enable the 3D detection node (needs a "
        "depth image + CameraInfo, see the config file)",
    )

    use_debug = LaunchConfiguration("use_debug")
    use_debug_cmd = DeclareLaunchArgument(
        "use_debug",
        default_value="True",
        description="Whether to enable the debug/visualization node (True/False)",
    )

    params_file = LaunchConfiguration("params_file")
    params_file_cmd = DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(
            get_package_share_directory("yolo_bringup"),
            "config",
            "yolo.yaml",
        ),
        description="Path to the ROS 2 parameters file (YAML) with the config "
        "for the yolo_node, tracking_node, detect_3d_node and debug_node "
        "blocks. All tuning lives here; any parameter can be overridden by "
        "the matching launch argument (run with --show-args for the list).",
    )

    namespace = LaunchConfiguration("namespace")
    namespace_cmd = DeclareLaunchArgument(
        "namespace",
        default_value="yolo",
        description="Namespace for the nodes",
    )

    tracker_cmd = DeclareLaunchArgument(
        "tracker",
        default_value="",
        description="Override the tracker file selected by the pipeline YAML's "
        "tracker_type: a name (config/trackers/<name>.yaml, e.g. botsort) or a "
        "path to a params file. Empty keeps the YAML's tracker_type.",
    )

    return LaunchDescription(
        [
            use_tracking_cmd,
            use_3d_cmd,
            use_debug_cmd,
            params_file_cmd,
            namespace_cmd,
            tracker_cmd,
            *declare_param_arguments(NODES),
            OpaqueFunction(
                function=_launch_setup,
                args=[params_file, namespace, use_tracking, use_3d, use_debug],
            ),
        ]
    )
