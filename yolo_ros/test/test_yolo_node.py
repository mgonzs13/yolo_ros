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


import pytest
from rclpy.executors import SingleThreadedExecutor
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool
from ultralytics import YOLO

from yolo_msgs.msg import DetectionArray

from yolo_ros.yolo_node import YoloNode

from conftest import BRIDGE
from conftest import best_effort_qos
from conftest import spin_until


def make_node():
    """Create a YoloNode configured for CPU inference with yolov8n."""
    node = YoloNode()
    node.set_parameters(
        [
            Parameter("model", value="yolov8n.pt"),
            Parameter("device", value="cpu"),
            Parameter("fuse_model", value=False),
        ]
    )
    return node


def test_detection_pipeline(fixed_image):
    node = make_node()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    received = []
    node.create_subscription(
        DetectionArray, "detections", lambda msg: received.append(msg), 10
    )
    pub = node.create_publisher(Image, "image_raw", best_effort_qos())

    assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
    assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS

    pub.publish(fixed_image)
    assert spin_until(executor, lambda: len(received) > 0)

    detections = received[0]
    assert len(detections.detections) > 0
    for detection in detections.detections:
        assert detection.class_name
        assert detection.score >= 0.5
        assert 0.0 <= detection.bbox.center.position.x <= fixed_image.width
        assert 0.0 <= detection.bbox.center.position.y <= fixed_image.height
        assert detection.bbox.size.x > 0.0
        assert detection.bbox.size.y > 0.0

    executor.shutdown()
    node.destroy_node()


def test_enable_cb():
    node = YoloNode()
    request = SetBool.Request()
    response = SetBool.Response()

    request.data = False
    node.enable_cb(request, response)
    assert response.success is True
    assert node.enable is False

    request.data = True
    node.enable_cb(request, response)
    assert response.success is True
    assert node.enable is True

    node.destroy_node()


def test_parse_hypothesis(fixed_image, yolo_model):
    cv_image = BRIDGE.imgmsg_to_cv2(fixed_image, desired_encoding="bgr8")
    results = yolo_model.predict(source=cv_image, verbose=False, conf=0.5, device="cpu")[
        0
    ]

    node = YoloNode()
    node.yolo = yolo_model
    hypothesis = node.parse_hypothesis(results)

    assert len(hypothesis) == len(results.boxes)
    for h in hypothesis:
        assert h["class_id"] in yolo_model.names
        assert h["class_name"] == yolo_model.names[h["class_id"]]
        assert 0.0 <= h["score"] <= 1.0

    node.destroy_node()


def test_parse_boxes(fixed_image, yolo_model):
    cv_image = BRIDGE.imgmsg_to_cv2(fixed_image, desired_encoding="bgr8")
    results = yolo_model.predict(source=cv_image, verbose=False, conf=0.5, device="cpu")[
        0
    ]

    node = YoloNode()
    boxes = node.parse_boxes(results)

    assert len(boxes) == len(results.boxes)
    for msg, box in zip(boxes, results.boxes.xywh):
        assert msg.center.position.x == pytest.approx(float(box[0]))
        assert msg.center.position.y == pytest.approx(float(box[1]))
        assert msg.size.x == pytest.approx(float(box[2]))
        assert msg.size.y == pytest.approx(float(box[3]))

    node.destroy_node()


@pytest.fixture(scope="session")
def yolo_seg_model():
    """YOLOv8n-seg model on CPU for mask parsing tests."""
    model = YOLO("yolov8n-seg.pt")
    model.to("cpu")
    return model


def test_parse_masks(fixed_image, yolo_seg_model):
    cv_image = BRIDGE.imgmsg_to_cv2(fixed_image, desired_encoding="bgr8")
    results = yolo_seg_model.predict(
        source=cv_image, verbose=False, conf=0.5, device="cpu"
    )[0]

    node = YoloNode()
    masks = node.parse_masks(results)

    assert len(masks) == len(results.masks)
    for mask in masks:
        assert len(mask.data) > 0
        assert mask.width == results.orig_img.shape[1]
        assert mask.height == results.orig_img.shape[0]

    node.destroy_node()


@pytest.fixture(scope="session")
def yolo_pose_model():
    """YOLOv8n-pose model on CPU for keypoint parsing tests."""
    model = YOLO("yolov8n-pose.pt")
    model.to("cpu")
    return model


def test_parse_keypoints(fixed_image, yolo_pose_model):
    cv_image = BRIDGE.imgmsg_to_cv2(fixed_image, desired_encoding="bgr8")
    results = yolo_pose_model.predict(
        source=cv_image, verbose=False, conf=0.5, device="cpu"
    )[0]

    node = YoloNode()
    node.threshold = 0.5
    keypoints_list = node.parse_keypoints(results)

    assert len(keypoints_list) == len(results.keypoints)
    for kp_array in keypoints_list:
        for kp in kp_array.data:
            assert kp.id >= 1
            assert kp.score >= node.threshold

    node.destroy_node()
