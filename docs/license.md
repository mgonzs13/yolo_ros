# License

The whole repository is licensed under the **MIT License**. See the root [`LICENSE`](../LICENSE) file; the same license applies to every ROS 2 package in this repository (`yolo_ros`, `yolo_msgs`, `yolo_bringup` and `yolo_hfhub_vendor`).

Third-party notices are installed with the applicable packages (`yolo_ros/THIRD_PARTY_NOTICES.md` and `yolo_hfhub_vendor/THIRD_PARTY_NOTICES.md`). ONNX Runtime is provided by the ROS 2 `onnxruntime_vendor` package at run time; for a custom ONNX Runtime prefix, keep the `LICENSE` and `ThirdPartyNotices.txt` from that distribution with the runtime files.

The C++ pipeline adapts behavior from the original `yolo_ros` Python nodes; those contributions were authorized by their copyright holder for release in the MIT-licensed C++ pipeline (see `yolo_ros/THIRD_PARTY_NOTICES.md`).

Model weights and exported ONNX files are separate artifacts and remain subject to their respective licenses; the MIT license for the C++ pipeline does not relicense them. In particular, Ultralytics models are **AGPL-3.0** — see the [model guide](models.md) and <https://ultralytics.com/license>.
