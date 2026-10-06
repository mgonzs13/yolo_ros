# Copyright (c) 2026 Alejandro González Cantón
# SPDX-License-Identifier: MIT

"""Multi-camera pipeline launch: one batched detector + per-camera stages."""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from yolo_bringup.launch_params import load_params_mapping, tracker_params_file


def _read_pipeline(path):
    if not path or not os.path.isfile(path):
        raise RuntimeError(f"pipeline file not found: {path!r}")
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    cameras = data.get("cameras") or []
    if not cameras:
        raise RuntimeError(f"pipeline file {path!r} has no cameras")
    names = [camera.get("name") for camera in cameras]
    if any(not name for name in names):
        raise RuntimeError("every camera needs a name")
    if len(set(names)) != len(names):
        raise RuntimeError(f"camera names must be unique: {names}")
    for camera in cameras:
        if not camera.get("image_topic"):
            raise RuntimeError(f"camera {camera.get('name')!r} needs an image_topic")
    return data


def _launch_setup(context, pipeline_file):
    # OpaqueFunction passes `args` through raw, so `pipeline_file` arrives as a
    # LaunchConfiguration that must be resolved against the launch context.
    # Tests may hand a plain path string instead.
    if isinstance(pipeline_file, str):
        path = pipeline_file
    else:
        path = context.launch_configurations.get("pipeline_file", "")
    data = _read_pipeline(path)
    namespace = str(data.get("namespace", "yolo"))
    model = dict(data.get("model") or {})
    cameras = data["cameras"]
    default_tracker = str(data.get("tracker", "bytetrack"))
    image_reliability = int(model.get("image_reliability", 2))

    nodes = [
        Node(
            package="yolo_ros",
            executable="yolo_batch_node",
            name="yolo_batch_node",
            namespace=namespace,
            parameters=[
                {
                    **model,
                    "camera_names": [str(c["name"]) for c in cameras],
                    "image_topics": [str(c["image_topic"]) for c in cameras],
                }
            ],
        )
    ]

    for camera in cameras:
        name = str(camera["name"])
        camera_ns = f"{namespace}/{name}"
        image_topic = str(camera["image_topic"])
        tracking = bool(camera.get("tracking", True))
        detections_topic = "tracking" if tracking else "detections"

        if tracking:
            tracker = str(camera.get("tracker") or default_tracker)
            base = load_params_mapping(tracker_params_file(tracker))
            nodes.append(
                Node(
                    package="yolo_ros",
                    executable="tracking_node",
                    name="tracking_node",
                    namespace=camera_ns,
                    parameters=[
                        {
                            **base,
                            "image_topic": image_topic,
                            "image_reliability": image_reliability,
                        }
                    ],
                )
            )

        depth = camera.get("depth")
        if depth:
            nodes.append(
                Node(
                    package="yolo_ros",
                    executable="detect_3d_node",
                    name="detect_3d_node",
                    namespace=camera_ns,
                    parameters=[
                        {
                            "depth_image_topic": str(depth["image_topic"]),
                            "depth_info_topic": str(depth["info_topic"]),
                            "depth_image_units_divisor": int(
                                depth.get("units_divisor", 1000)
                            ),
                            "depth_image_reliability": int(
                                depth.get("image_reliability", 2)
                            ),
                            "depth_info_reliability": int(
                                depth.get("info_reliability", 2)
                            ),
                            "detections_topic": detections_topic,
                            "target_frame": str(depth.get("target_frame", "")),
                            "enable_orientation": bool(
                                depth.get("enable_orientation", False)
                            ),
                            "min_seg_points_for_orientation": int(
                                depth.get("min_seg_points_for_orientation", 20)
                            ),
                        }
                    ],
                )
            )

        if bool(camera.get("debug", True)):
            nodes.append(
                Node(
                    package="yolo_ros",
                    executable="debug_node",
                    name="debug_node",
                    namespace=camera_ns,
                    parameters=[
                        {
                            "image_topic": image_topic,
                            "image_reliability": image_reliability,
                            "detections_topic": detections_topic,
                            "markers_topic": "detections_3d",
                        }
                    ],
                )
            )

    return nodes


def generate_launch_description():
    pipeline_file = LaunchConfiguration("pipeline_file")
    pipeline_file_cmd = DeclareLaunchArgument(
        "pipeline_file",
        default_value=os.path.join(
            get_package_share_directory("yolo_bringup"), "config", "pipelines.yaml"
        ),
        description="Path to the multi-camera pipeline config "
        "(namespace, shared model, cameras).",
    )

    return LaunchDescription(
        [
            pipeline_file_cmd,
            OpaqueFunction(function=_launch_setup, args=[pipeline_file]),
        ]
    )
