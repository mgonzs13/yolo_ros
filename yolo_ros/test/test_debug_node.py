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


import copy

import numpy as np
import pytest
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import TransitionCallbackReturn
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from yolo_msgs.msg import DetectionArray
from yolo_msgs.msg import KeyPoint3D

from yolo_ros.debug_node import DebugNode

from conftest import best_effort_qos
from conftest import make_detection
from conftest import spin
from conftest import spin_until


def test_debug_pipeline(fixed_image):
    node = DebugNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    dbg_images = []
    bb_markers = []
    kp_markers = []
    node.create_subscription(Image, "dbg_image", lambda msg: dbg_images.append(msg), 10)
    node.create_subscription(
        MarkerArray, "dgb_bb_markers", lambda msg: bb_markers.append(msg), 10
    )
    node.create_subscription(
        MarkerArray, "dgb_kp_markers", lambda msg: kp_markers.append(msg), 10
    )

    img_pub = node.create_publisher(Image, "image_raw", best_effort_qos())
    det_pub = node.create_publisher(DetectionArray, "detections", 10)

    assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
    assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS

    # Let the subscriptions match before publishing (BEST_EFFORT image).
    spin(executor, 0.5)

    stamp = node.get_clock().now().to_msg()
    image = copy.deepcopy(fixed_image)
    image.header.stamp = stamp

    detection = make_detection()
    detection.bbox3d.frame_id = "base_link"
    detection.bbox3d.center.position.x = 1.0
    detection.bbox3d.center.position.z = 2.0
    detection.bbox3d.size.x = 0.5
    kp = KeyPoint3D()
    kp.id = 1
    kp.point.x = 1.0
    kp.point.y = 2.0
    kp.point.z = 2.0
    kp.score = 0.9
    detection.keypoints3d.frame_id = "base_link"
    detection.keypoints3d.data.append(kp)

    detections_msg = DetectionArray()
    detections_msg.header = image.header
    detections_msg.detections.append(detection)

    img_pub.publish(image)
    det_pub.publish(detections_msg)

    assert spin_until(
        executor,
        lambda: len(dbg_images) > 0 and len(bb_markers) > 0 and len(kp_markers) > 0,
    )

    assert dbg_images[0].height == fixed_image.height
    assert dbg_images[0].width == fixed_image.width
    assert len(bb_markers) >= 1
    assert len(bb_markers[0].markers) >= 1
    assert len(kp_markers) >= 1
    assert len(kp_markers[0].markers) >= 1

    executor.shutdown()
    node.destroy_node()


def test_draw_box():
    node = DebugNode()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    detection = make_detection()
    out = node.draw_box(image.copy(), detection, (255, 0, 0))
    assert out.shape == image.shape
    assert not np.array_equal(out, image)
    node.destroy_node()


def test_create_bb_marker():
    node = DebugNode()
    detection = make_detection()
    detection.bbox3d.frame_id = "base_link"
    detection.bbox3d.center.position.x = 1.0
    detection.bbox3d.center.position.y = 2.0
    detection.bbox3d.center.position.z = 3.0
    detection.bbox3d.size.x = 0.5
    detection.bbox3d.size.y = 0.6
    detection.bbox3d.size.z = 0.7
    marker = node.create_bb_marker(detection, (10, 20, 30))
    assert marker.header.frame_id == "base_link"
    assert marker.type == Marker.CUBE
    assert marker.pose.position.x == pytest.approx(1.0)
    assert marker.pose.position.y == pytest.approx(2.0)
    assert marker.pose.position.z == pytest.approx(3.0)
    assert marker.scale.x == pytest.approx(0.5)
    assert marker.scale.y == pytest.approx(0.6)
    assert marker.scale.z == pytest.approx(0.7)
    node.destroy_node()
