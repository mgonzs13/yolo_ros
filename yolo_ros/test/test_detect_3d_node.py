# Copyright (C) 2026 Miguel Ángel González Santamarta
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import pytest
from geometry_msgs.msg import TransformStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from yolo_msgs.msg import BoundingBox3D
from yolo_msgs.msg import DetectionArray

from yolo_ros.detect_3d_node import Detect3DNode

from conftest import best_effort_qos
from conftest import make_camera_info
from conftest import make_depth_image
from conftest import make_detection
from conftest import spin
from conftest import spin_until


def test_detections_3d_pipeline():
    tf_node = Node("tf_broadcaster")
    broadcaster = StaticTransformBroadcaster(tf_node)
    transform = TransformStamped()
    transform.header.stamp = tf_node.get_clock().now().to_msg()
    transform.header.frame_id = "base_link"
    transform.child_frame_id = "camera_link"
    transform.transform.translation.x = 0.1
    transform.transform.translation.z = 0.3
    transform.transform.rotation.w = 1.0
    broadcaster.sendTransform(transform)

    node = Detect3DNode()
    node.set_parameters(
        [
            Parameter("target_frame", value="base_link"),
            Parameter("enable_orientation", value=False),
        ]
    )

    executor = SingleThreadedExecutor()
    executor.add_node(tf_node)
    executor.add_node(node)

    received = []
    node.create_subscription(
        DetectionArray, "detections_3d", lambda msg: received.append(msg), 10
    )

    depth_pub = node.create_publisher(Image, "depth_image", best_effort_qos())
    info_pub = node.create_publisher(CameraInfo, "depth_info", best_effort_qos())
    det_pub = node.create_publisher(DetectionArray, "detections", 10)

    assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
    assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS

    # Let the TF listener receive the static transform.
    spin(executor, 1.0)

    stamp = node.get_clock().now().to_msg()
    depth_msg = make_depth_image()
    depth_msg.header.stamp = stamp
    info_msg = make_camera_info()
    info_msg.header.stamp = stamp
    detections_msg = DetectionArray()
    detections_msg.header.stamp = stamp
    detections_msg.detections.append(make_detection())

    depth_pub.publish(depth_msg)
    info_pub.publish(info_msg)
    det_pub.publish(detections_msg)

    assert spin_until(executor, lambda: len(received) > 0)

    out = received[0]
    assert len(out.detections) == 1
    bbox3d = out.detections[0].bbox3d
    assert bbox3d.frame_id == "base_link"
    assert bbox3d.center.position.x == pytest.approx(0.1, abs=0.1)
    assert bbox3d.center.position.y == pytest.approx(0.0, abs=0.1)
    assert bbox3d.center.position.z == pytest.approx(2.3, abs=0.1)

    executor.shutdown()
    node.destroy_node()
    tf_node.destroy_node()


def test_convert_bb_to_3d():
    node = Detect3DNode()
    node.depth_image_units_divisor = 1000
    node.enable_orientation = False

    depth = np.zeros((480, 640), dtype=np.uint16)
    depth[100:380, 200:440] = 2000
    info = make_camera_info()
    detection = make_detection()

    bbox3d = node.convert_bb_to_3d(depth, info, detection)

    assert bbox3d is not None
    assert bbox3d.center.position.x == pytest.approx(0.0, abs=0.05)
    assert bbox3d.center.position.y == pytest.approx(0.0, abs=0.05)
    assert bbox3d.center.position.z == pytest.approx(2.0, abs=0.05)
    assert bbox3d.size.x == pytest.approx(0.87, abs=0.05)
    assert bbox3d.size.y == pytest.approx(0.92, abs=0.05)
    assert bbox3d.size.z <= 0.1

    node.destroy_node()


def test_transform_3d_box():
    bbox = BoundingBox3D()
    bbox.center.position.x = 1.0
    bbox.center.position.y = 2.0
    bbox.center.position.z = 3.0

    translation = np.array([0.1, 0.0, 0.3])
    rotation = np.array([1.0, 0.0, 0.0, 0.0])

    out = Detect3DNode.transform_3d_box(bbox, translation, rotation)

    assert out.center.position.x == pytest.approx(1.1)
    assert out.center.position.y == pytest.approx(2.0)
    assert out.center.position.z == pytest.approx(3.3)
    assert out.center.orientation.w == pytest.approx(1.0)
    assert out.center.orientation.x == pytest.approx(0.0)
    assert out.center.orientation.y == pytest.approx(0.0)
    assert out.center.orientation.z == pytest.approx(0.0)


def test_compute_depth_bounds_weighted():
    depths = np.full(100, 2.0)
    weights = np.ones(100)
    z, z_min, z_max = Detect3DNode._compute_depth_bounds_weighted(depths, weights)
    assert z == pytest.approx(2.0, abs=0.01)
    assert z_min <= 2.0 + 1e-6
    assert z_max >= 2.0 - 1e-6
