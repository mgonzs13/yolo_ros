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


import os
import time

import cv2
import numpy as np
import pytest
import rclpy
from cv_bridge import CvBridge
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from ultralytics import YOLO
from ultralytics.utils import ASSETS

from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray

BRIDGE = CvBridge()
BUS_IMAGE_PATH = os.path.join(ASSETS, "bus.jpg")


def spin(executor, seconds=1.0):
    """Spin the executor for the given number of seconds."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        executor.spin_once(timeout_sec=0.05)


def spin_until(executor, predicate, timeout=30.0):
    """Spin the executor until predicate() is True or timeout elapses."""
    start = time.monotonic()
    while not predicate() and time.monotonic() - start < timeout:
        executor.spin_once(timeout_sec=0.05)
    return predicate()


def best_effort_qos(depth=1):
    """BEST_EFFORT QoS profile used by the nodes' image subscriptions."""
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=depth,
    )


def predict_detections(model, image_msg, conf=0.5):
    """Run real inference and build a DetectionArray message."""
    cv_image = BRIDGE.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
    results = model.predict(
        source=cv_image, verbose=False, conf=conf, device="cpu"
    )[0]
    detections = DetectionArray()
    detections.header = image_msg.header
    for box, cls, score in zip(
        results.boxes.xywh, results.boxes.cls, results.boxes.conf
    ):
        detection = Detection()
        detection.class_id = int(cls)
        detection.class_name = model.names[int(cls)]
        detection.score = float(score)
        detection.bbox.center.position.x = float(box[0])
        detection.bbox.center.position.y = float(box[1])
        detection.bbox.size.x = float(box[2])
        detection.bbox.size.y = float(box[3])
        detections.detections.append(detection)
    return detections


def make_detection(
    cx=320.0, cy=240.0, size_x=240.0, size_y=280.0,
    class_id=0, class_name="person", score=0.9,
):
    """Create a synthetic detection message."""
    detection = Detection()
    detection.class_id = class_id
    detection.class_name = class_name
    detection.score = score
    detection.bbox.center.position.x = cx
    detection.bbox.center.position.y = cy
    detection.bbox.size.x = size_x
    detection.bbox.size.y = size_y
    return detection


def make_camera_info(frame_id="camera_link"):
    """Create a synthetic 640x480 CameraInfo (fx=fy=500, cx=320, cy=240)."""
    info = CameraInfo()
    info.header.frame_id = frame_id
    info.width = 640
    info.height = 480
    info.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
    return info


def make_depth_image(depth_mm=2000):
    """Create a uint16 depth image with a constant-depth rectangle."""
    msg = Image()
    msg.header.frame_id = "camera_link"
    msg.height = 480
    msg.width = 640
    msg.encoding = "16UC1"
    msg.is_bigendian = 0
    msg.step = 640 * 2
    depth = np.zeros((480, 640), dtype=np.uint16)
    depth[100:380, 200:440] = depth_mm
    msg.data = depth.tobytes()
    return msg


@pytest.fixture(scope="session", autouse=True)
def rclpy_context():
    """Initialize and shutdown rclpy for the whole test session."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture(scope="session")
def fixed_image():
    """Load the ultralytics bus.jpg sample as a bgr8 sensor_msgs/Image."""
    cv_image = cv2.imread(BUS_IMAGE_PATH)
    assert cv_image is not None, f"Could not load {BUS_IMAGE_PATH}"
    msg = BRIDGE.cv2_to_imgmsg(cv_image, encoding="bgr8")
    msg.header.frame_id = "camera_link"
    return msg


@pytest.fixture(scope="session")
def yolo_model():
    """YOLOv8n model on CPU for generating real detections."""
    model = YOLO("yolov8n.pt")
    model.to("cpu")
    return model
