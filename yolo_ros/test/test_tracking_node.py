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
import os
import tempfile

import pytest
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import TransitionCallbackReturn
from sensor_msgs.msg import Image

from yolo_msgs.msg import DetectionArray

from yolo_ros.tracking_node import TrackingNode

from conftest import best_effort_qos
from conftest import predict_detections
from conftest import spin
from conftest import spin_until


def test_tracking_pipeline(fixed_image, yolo_model):
    node = TrackingNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    received = []
    node.create_subscription(
        DetectionArray, "tracking", lambda msg: received.append(msg), 10
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
    detections = predict_detections(yolo_model, image, conf=0.5)
    detections.header.stamp = stamp

    img_pub.publish(image)
    det_pub.publish(detections)

    assert spin_until(executor, lambda: len(received) > 0)

    tracked = received[0]
    assert len(tracked.detections) >= 1
    for detection in tracked.detections:
        assert detection.id
        assert detection.bbox.size.x > 0.0
        assert detection.bbox.size.y > 0.0

    executor.shutdown()
    node.destroy_node()


def test_create_tracker_invalid():
    node = TrackingNode()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("tracker_type: invalid_tracker\n")
        path = f.name
    try:
        with pytest.raises(AssertionError):
            node.create_tracker(path)
    finally:
        os.unlink(path)
    node.destroy_node()
