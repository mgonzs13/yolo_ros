# Copyright (C) 2023 Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import cv2
import numpy as np
from typing import List, Tuple

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import message_filters
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray
from yolo_msgs.msg import KeyPoint3D
from yolo_msgs.msg import KeyPoint3DArray
from yolo_msgs.msg import BoundingBox3D


class Detect3DNode(LifecycleNode):
    """
    ROS 2 Lifecycle Node for 3D object detection.

    This node converts 2D detections to 3D by using depth information from a depth camera.
    It subscribes to detections, depth images, and camera info, then publishes 3D bounding
    boxes and keypoints in a target reference frame.
    """

    def __init__(self) -> None:
        """
        Initialize the 3D detection node.

        Declares ROS parameters and initializes TF buffer and CV bridge.
        """
        super().__init__("bbox3d_node")

        # Parameters
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("depth_image_units_divisor", 1000)
        self.declare_parameter(
            "depth_image_reliability", QoSReliabilityPolicy.BEST_EFFORT
        )
        self.declare_parameter("depth_info_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        # Auxiliary variables
        self.tf_buffer = Buffer()
        self.cv_bridge = CvBridge()

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Configure lifecycle callback.

        Retrieves parameters, sets up QoS profiles, creates publishers, and initializes TF listener.

        @param state Current lifecycle state
        @return Transition callback return status
        """
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        self.target_frame = (
            self.get_parameter("target_frame").get_parameter_value().string_value
        )
        self.depth_image_units_divisor = (
            self.get_parameter("depth_image_units_divisor")
            .get_parameter_value()
            .integer_value
        )
        dimg_reliability = (
            self.get_parameter("depth_image_reliability")
            .get_parameter_value()
            .integer_value
        )

        self.depth_image_qos_profile = QoSProfile(
            reliability=dimg_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        dinfo_reliability = (
            self.get_parameter("depth_info_reliability")
            .get_parameter_value()
            .integer_value
        )

        self.depth_info_qos_profile = QoSProfile(
            reliability=dinfo_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Pubs
        self._pub = self.create_publisher(DetectionArray, "detections_3d", 10)

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Activate lifecycle callback.

        Creates subscriptions to depth image, camera info, and detections with time synchronization.

        @param state Current lifecycle state
        @return Transition callback return status
        """
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        # Subs
        self.depth_sub = message_filters.Subscriber(
            self, Image, "depth_image", qos_profile=self.depth_image_qos_profile
        )
        self.depth_info_sub = message_filters.Subscriber(
            self, CameraInfo, "depth_info", qos_profile=self.depth_info_qos_profile
        )
        self.detections_sub = message_filters.Subscriber(
            self, DetectionArray, "detections"
        )

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.depth_sub, self.depth_info_sub, self.detections_sub), 10, 0.5
        )
        self._synchronizer.registerCallback(self.on_detections)

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Deactivate lifecycle callback.

        Destroys subscriptions and cleans up the synchronizer.

        @param state Current lifecycle state
        @return Transition callback return status
        """
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        self.destroy_subscription(self.depth_sub.sub)
        self.destroy_subscription(self.depth_info_sub.sub)
        self.destroy_subscription(self.detections_sub.sub)

        del self._synchronizer

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Cleanup lifecycle callback.

        Destroys the TF listener and publisher, cleaning up resources.

        @param state Current lifecycle state
        @return Transition callback return status
        """
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        del self.tf_listener
        self.destroy_publisher(self._pub)

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Shutdown lifecycle callback.

        Performs final cleanup before node shutdown.

        @param state Current lifecycle state
        @return Transition callback return status
        """
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_shutdown(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def on_detections(
        self,
        depth_msg: Image,
        depth_info_msg: CameraInfo,
        detections_msg: DetectionArray,
    ) -> None:
        """
        Synchronized callback for depth image, camera info, and detections.

        Processes detections to add 3D information and publishes the results.

        @param depth_msg Depth image message
        @param depth_info_msg Camera info message
        @param detections_msg Detections message
        """

        new_detections_msg = DetectionArray()
        new_detections_msg.header = detections_msg.header
        new_detections_msg.detections = self.process_detections(
            depth_msg, depth_info_msg, detections_msg
        )
        self._pub.publish(new_detections_msg)

    def process_detections(
        self,
        depth_msg: Image,
        depth_info_msg: CameraInfo,
        detections_msg: DetectionArray,
    ) -> List[Detection]:
        """
        Process 2D detections to add 3D bounding boxes and keypoints.

        Converts depth image to OpenCV format, looks up TF transform, and converts
        each detection to 3D coordinates in the target frame.

        @param depth_msg Depth image message
        @param depth_info_msg Camera info message
        @param detections_msg Array of 2D detections
        @return List of detections with 3D information added
        """

        # Check if there are detections
        if not detections_msg.detections:
            return []

        transform = self.get_transform(depth_info_msg.header.frame_id)

        if transform is None:
            return []

        new_detections = []
        depth_image = self.cv_bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding="passthrough"
        )

        for detection in detections_msg.detections:
            bbox3d = self.convert_bb_to_3d(depth_image, depth_info_msg, detection)

            if bbox3d is not None:
                new_detections.append(detection)

                bbox3d = Detect3DNode.transform_3d_box(bbox3d, transform[0], transform[1])
                bbox3d.frame_id = self.target_frame
                new_detections[-1].bbox3d = bbox3d

                if detection.keypoints.data:
                    keypoints3d = self.convert_keypoints_to_3d(
                        depth_image, depth_info_msg, detection
                    )
                    keypoints3d = Detect3DNode.transform_3d_keypoints(
                        keypoints3d, transform[0], transform[1]
                    )
                    keypoints3d.frame_id = self.target_frame
                    new_detections[-1].keypoints3d = keypoints3d

        return new_detections

    @staticmethod
    def compute_depth_bounds(depth_values: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute robust depth statistics for the foreground object using
        advanced density-based analysis and multimodal distribution handling.

        Args:
            depth_values: 1D array of valid depth values (> 0)

        Returns:
            Tuple of (z_center, z_min, z_max) representing the object's depth
        """
        # Basic input validation
        if not isinstance(depth_values, np.ndarray):
            return 0.0, 0.0, 0.0

        if len(depth_values) == 0:
            return 0.0, 0.0, 0.0

        # Ensure all values are numeric and finite
        try:
            depth_values = np.asarray(depth_values, dtype=np.float64)
            valid_mask = np.isfinite(depth_values) & (depth_values > 0)
            depth_values = depth_values[valid_mask]
        except (ValueError, TypeError):
            return 0.0, 0.0, 0.0

        if len(depth_values) == 0:
            return 0.0, 0.0, 0.0

        if len(depth_values) < 4:
            z_center = float(np.median(depth_values))
            return z_center, float(np.min(depth_values)), float(np.max(depth_values))

        sorted_depths = np.sort(depth_values)
        n = len(sorted_depths)

        # Step 1: Identify foreground cluster using multi-criteria analysis
        # 1a. Gap-based detection
        depth_diffs = np.diff(sorted_depths)
        median_diff = np.median(depth_diffs)
        mad_diff = np.median(np.abs(depth_diffs - median_diff))
        gap_threshold = max(median_diff + 3.0 * mad_diff, 0.05)

        large_gaps = np.where(depth_diffs > gap_threshold)[0]

        # 1b. Histogram-based density analysis for mode detection
        depth_range = sorted_depths[-1] - sorted_depths[0]
        # Adaptive bin size: ~1-2cm resolution
        if not np.isfinite(depth_range) or depth_range <= 0:
            n_bins = 30
        else:
            n_bins = max(15, min(50, int(depth_range / 0.015)))
        hist, bin_edges = np.histogram(depth_values, bins=n_bins)

        # Find peak (mode) - highest density region
        peak_bin_idx = np.argmax(hist)
        mode_depth = (bin_edges[peak_bin_idx] + bin_edges[peak_bin_idx + 1]) / 2

        # 1c. Combine gap detection and density analysis
        if len(large_gaps) > 0:
            # Use gap to separate foreground, but validate with density
            cutoff_idx = large_gaps[0] + 1
            candidate_cluster = sorted_depths[:cutoff_idx]

            # Verify the mode is in this cluster (ensures we got the right cluster)
            if mode_depth <= sorted_depths[cutoff_idx]:
                object_depths = candidate_cluster
            else:
                # Mode is beyond the gap - use density-based selection
                object_depths = Detect3DNode._density_based_cluster(
                    depth_values, mode_depth, sorted_depths
                )
        else:
            # No clear gap - use density-based clustering around mode
            object_depths = Detect3DNode._density_based_cluster(
                depth_values, mode_depth, sorted_depths
            )

        # Safety check
        if len(object_depths) < max(4, n * 0.03):
            # Fallback: use percentile-based selection biased toward foreground
            p5 = np.percentile(sorted_depths, 5)
            p70 = np.percentile(sorted_depths, 70)
            object_depths = depth_values[(depth_values >= p5) & (depth_values <= p70)]

        if len(object_depths) == 0:
            object_depths = depth_values

        # Step 2: Compute precise center using density-weighted approach
        z_center = Detect3DNode._compute_weighted_center(object_depths)

        # Step 3: Compute extent using robust percentiles
        z_min = np.percentile(object_depths, 1)
        z_max = np.percentile(object_depths, 99)

        # Ensure minimum depth size
        min_depth_size = 0.01  # 1cm
        if (z_max - z_min) < min_depth_size:
            half_min = min_depth_size / 2
            z_min = z_center - half_min
            z_max = z_center + half_min

        return float(z_center), float(z_min), float(z_max)

    @staticmethod
    def _density_based_cluster(
        depth_values: np.ndarray, mode_depth: float, sorted_depths: np.ndarray
    ) -> np.ndarray:
        """
        Extract foreground cluster based on local density around the mode.

        Args:
            depth_values: Original depth values
            mode_depth: The detected mode (peak density)
            sorted_depths: Sorted depth values

        Returns:
            Filtered depth values representing the foreground object
        """
        # Use adaptive threshold based on data spread around mode
        deviations = np.abs(depth_values - mode_depth)
        mad = np.median(deviations)

        # Adaptive threshold: 2-3 * MAD, bounded by [5cm, 25cm]
        # Tighter for uniform objects, looser for complex shapes
        q25_dev = np.percentile(deviations, 25)
        q75_dev = np.percentile(deviations, 75)
        iqr_dev = q75_dev - q25_dev

        if iqr_dev < 0.02:  # Very uniform depth (< 2cm variation)
            threshold = np.clip(2.0 * mad, 0.05, 0.15)
        else:  # More depth variation
            threshold = np.clip(3.0 * mad, 0.08, 0.25)

        # Keep depths within threshold from mode
        cluster_mask = deviations <= threshold
        cluster = depth_values[cluster_mask]

        # Additional check: ensure we capture reasonable portion
        if len(cluster) < len(depth_values) * 0.1:
            # Fall back to percentile-based approach
            p70 = np.percentile(sorted_depths, 70)
            cluster = depth_values[depth_values <= p70]

        return cluster if len(cluster) > 0 else depth_values

    @staticmethod
    def _compute_weighted_center(object_depths: np.ndarray) -> float:
        """
        Compute the center depth using density-weighted mean for accuracy.
        This gives more weight to depths with more neighboring points,
        resulting in a center that better represents the object's mass.

        Args:
            object_depths: Filtered depth values of the object

        Returns:
            Weighted center depth value
        """
        if len(object_depths) < 10:
            # For small samples, use trimmed mean (remove extreme 5%)
            return Detect3DNode._trimmed_mean(object_depths, 0.05)

        # Use histogram to estimate local density
        depth_range = np.ptp(object_depths)
        n_bins = max(10, min(30, int(depth_range / 0.01)))  # ~1cm bins

        hist, bin_edges = np.histogram(object_depths, bins=n_bins)

        # Assign density weight to each depth value
        # Find which bin each depth belongs to
        bin_indices = np.digitize(object_depths, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(hist) - 1)

        # Weight each depth by its bin's density
        weights = hist[bin_indices]
        weights = weights.astype(float)

        # Avoid division by zero
        if np.sum(weights) == 0:
            return np.median(object_depths)

        # Compute weighted mean
        weighted_center = np.average(object_depths, weights=weights)

        # Sanity check: should be within the data range
        if weighted_center < np.min(object_depths) or weighted_center > np.max(
            object_depths
        ):
            return np.median(object_depths)

        return weighted_center

    @staticmethod
    def _trimmed_mean(values: np.ndarray, trim_fraction: float) -> float:
        """
        Compute trimmed mean by removing extreme values.

        Args:
            values: Input values
            trim_fraction: Fraction to trim from each end (e.g., 0.05 = 5%)

        Returns:
            Trimmed mean value
        """
        if len(values) < 4:
            return np.mean(values)

        lower_percentile = trim_fraction * 100
        upper_percentile = (1 - trim_fraction) * 100

        lower_bound = np.percentile(values, lower_percentile)
        upper_bound = np.percentile(values, upper_percentile)

        trimmed = values[(values >= lower_bound) & (values <= upper_bound)]

        return np.mean(trimmed) if len(trimmed) > 0 else np.mean(values)

    def convert_bb_to_3d(
        self,
        depth_image: np.ndarray,
        depth_info: CameraInfo,
        detection: Detection,
    ) -> BoundingBox3D:
        """
        Convert 2D bounding box to 3D using depth information.

        Uses depth image to estimate 3D position and size of detected objects.
        Supports both mask-based and bbox-based depth sampling with spatial weighting.

        @param depth_image Depth image as numpy array
        @param depth_info Camera intrinsic parameters
        @param detection 2D detection to convert
        @return 3D bounding box or None if conversion fails
        """
        # Basic input validations
        if depth_image is None or not isinstance(depth_image, np.ndarray):
            return None

        if depth_image.size == 0:
            return None

        center_x = int(detection.bbox.center.position.x)
        center_y = int(detection.bbox.center.position.y)
        size_x = int(detection.bbox.size.x)
        size_y = int(detection.bbox.size.y)

        if detection.mask.data:
            # Crop depth image by mask
            mask_array = np.array(
                [[int(ele.x), int(ele.y)] for ele in detection.mask.data]
            )
            mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(mask_array, dtype=np.int32)], 255)
            roi = cv2.bitwise_and(depth_image, depth_image, mask=mask)

            # Get pixel coordinates for spatial weighting
            y_coords, x_coords = np.where(mask > 0)
            pixel_coords = np.column_stack([x_coords, y_coords])

        else:
            # Crop depth image by the 2D BB
            u_min = max(center_x - size_x // 2, 0)
            u_max = min(center_x + size_x // 2, depth_image.shape[1] - 1)
            v_min = max(center_y - size_y // 2, 0)
            v_max = min(center_y + size_y // 2, depth_image.shape[0] - 1)

            roi = depth_image[v_min:v_max, u_min:u_max]

            # Generate pixel coordinates for spatial weighting
            roi_h, roi_w = roi.shape
            y_grid, x_grid = np.meshgrid(
                np.arange(roi_h) + v_min, np.arange(roi_w) + u_min, indexing="ij"
            )
            pixel_coords = np.column_stack([x_grid.flatten(), y_grid.flatten()])

        roi = roi / self.depth_image_units_divisor  # Convert to meters

        # Validate that division did not produce NaN or inf
        if not np.any(np.isfinite(roi)):
            return None

        if not np.any(roi):
            return None

        # Extract valid depth values with their spatial positions
        valid_depths = roi.flatten()

        # Ensure correct numeric type
        try:
            valid_depths = np.asarray(valid_depths, dtype=np.float64)
        except (ValueError, TypeError):
            return None

        valid_mask = (valid_depths > 0) & np.isfinite(valid_depths)
        valid_depths = valid_depths[valid_mask]
        valid_coords = pixel_coords[valid_mask]

        if len(valid_depths) == 0:
            return None

        # Compute spatial weights based on distance from 2D bbox center
        # Pixels closer to center are more likely to be the actual object
        spatial_weights = self._compute_spatial_weights(
            valid_coords, center_x, center_y, size_x, size_y
        )

        # Compute robust depth statistics with spatial weighting
        z, z_min, z_max = Detect3DNode._compute_depth_bounds_weighted(
            valid_depths, spatial_weights
        )

        if not np.isfinite(z) or z == 0:
            return None

        # Compute height (y-axis) statistics from actual 3D points
        y_center, y_min, y_max = Detect3DNode._compute_height_bounds(
            valid_coords, valid_depths, spatial_weights, depth_info
        )

        # Validate results
        if not all(np.isfinite([y_center, y_min, y_max])):
            return None

        # Compute width (x-axis) statistics from actual 3D points
        x_center, x_min, x_max = Detect3DNode._compute_width_bounds(
            valid_coords, valid_depths, spatial_weights, depth_info
        )

        # Validate results
        if not all(np.isfinite([x_center, x_min, x_max])):
            return None

        # All dimensions come from actual 3D point analysis
        x = x_center
        y = y_center
        w = float(x_max - x_min)
        h = float(y_max - y_min)

        # Create 3D BB
        msg = BoundingBox3D()
        msg.center.position.x = x
        msg.center.position.y = y
        msg.center.position.z = z
        msg.size.x = w
        msg.size.y = h
        msg.size.z = float(z_max - z_min)

        return msg

    @staticmethod
    def _compute_spatial_weights(
        coords: np.ndarray, center_x: int, center_y: int, size_x: int, size_y: int
    ) -> np.ndarray:
        """
        Compute spatial weights for depth values based on distance from 2D bbox center.
        Pixels near the center get higher weight to handle occlusions better.

        Args:
            coords: Nx2 array of pixel coordinates [x, y]
            center_x: X coordinate of bbox center
            center_y: Y coordinate of bbox center
            size_x: Width of bbox
            size_y: Height of bbox

        Returns:
            Array of weights (0-1) for each coordinate
        """
        # Compute normalized distance from center
        dx = (coords[:, 0] - center_x) / (size_x / 2 + 1e-6)
        dy = (coords[:, 1] - center_y) / (size_y / 2 + 1e-6)
        normalized_dist = np.sqrt(dx**2 + dy**2)

        # Use Gaussian-like weighting: higher weight at center, lower at edges
        # sigma = 0.8 means ~80% of bbox radius has high weight
        weights = np.exp(-0.5 * (normalized_dist / 0.8) ** 2)

        # Ensure minimum weight of 0.3 to not completely ignore edge pixels
        weights = np.maximum(weights, 0.3)

        return weights

    @staticmethod
    def _compute_height_bounds(
        valid_coords: np.ndarray,
        valid_depths: np.ndarray,
        spatial_weights: np.ndarray,
        depth_info: CameraInfo,
    ) -> Tuple[float, float, float]:
        """
        Compute 3D height (y-axis) statistics from valid depth points.
        Uses actual 3D point positions instead of just projecting 2D bbox.

        Args:
            valid_coords: Nx2 array of pixel coordinates [x, y]
            valid_depths: N array of depth values in meters
            spatial_weights: N array of spatial weights
            depth_info: Camera intrinsic parameters

        Returns:
            Tuple of (y_center, y_min, y_max) in meters
        """
        # Input validations
        try:
            valid_depths = np.asarray(valid_depths, dtype=np.float64)
            spatial_weights = np.asarray(spatial_weights, dtype=np.float64)
        except (ValueError, TypeError):
            return 0.0, 0.0, 0.0

        if len(valid_coords) == 0 or len(valid_depths) == 0:
            return 0.0, 0.0, 0.0

        if len(valid_coords) < 4:
            # Fallback: just use simple projection
            k = depth_info.k
            py, fy = k[5], k[4]

            # Validate camera parameters
            if fy == 0:
                return 0.0, 0.0, 0.0

            # Validate depths are finite
            if not np.all(np.isfinite(valid_depths)):
                return 0.0, 0.0, 0.0

            y_coords_pixel = valid_coords[:, 1]
            y_3d = valid_depths * (y_coords_pixel - py) / fy

            # Validate result
            if not np.all(np.isfinite(y_3d)):
                return 0.0, 0.0, 0.0

            return float(np.median(y_3d)), float(np.min(y_3d)), float(np.max(y_3d))

        # Convert pixel coordinates to 3D y-coordinates
        k = depth_info.k
        py, fy = k[5], k[4]

        # Validate camera parameters
        if fy == 0:
            return 0.0, 0.0, 0.0

        # Validate depths are finite before calculation
        if not np.all(np.isfinite(valid_depths)):
            return 0.0, 0.0, 0.0

        y_coords_pixel = valid_coords[:, 1]
        y_3d = valid_depths * (y_coords_pixel - py) / fy

        # Validate result
        if not np.any(np.isfinite(y_3d)):
            return 0.0, 0.0, 0.0

        # Filter outliers using robust statistics
        # Compute weighted median as reference
        sorted_idx = np.argsort(y_3d)
        sorted_y = y_3d[sorted_idx]
        sorted_weights = spatial_weights[sorted_idx]
        cumsum_weights = np.cumsum(sorted_weights)
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0
        median_idx = np.searchsorted(cumsum_weights, 0.5)
        y_median = sorted_y[median_idx]

        # Compute MAD (Median Absolute Deviation)
        deviations = np.abs(y_3d - y_median)
        mad = np.median(deviations)

        # Filter outliers: keep points within 4.5*MAD from median
        # Balanced threshold to handle tall objects while avoiding background
        threshold = np.clip(4.5 * mad, 0.06, 0.50)
        valid_mask = deviations <= threshold
        filtered_y = y_3d[valid_mask]
        filtered_weights = spatial_weights[valid_mask]

        # Ensure we have enough points (at least 12% of data)
        if len(filtered_y) < max(4, len(y_3d) * 0.12):
            filtered_y = y_3d
            filtered_weights = spatial_weights

        # Compute weighted center using trimmed mean
        sorted_idx = np.argsort(filtered_y)
        sorted_y = filtered_y[sorted_idx]
        sorted_weights = filtered_weights[sorted_idx]
        cumsum_weights = np.cumsum(sorted_weights)
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0

        # Trim 5% from each end for robust center estimation
        trim_low_idx = np.searchsorted(cumsum_weights, 0.05)
        trim_high_idx = np.searchsorted(cumsum_weights, 0.95)

        if trim_high_idx > trim_low_idx:
            trimmed_y = sorted_y[trim_low_idx:trim_high_idx]
            trimmed_weights = sorted_weights[trim_low_idx:trim_high_idx]
            if np.sum(trimmed_weights) > 0:
                y_center = np.average(trimmed_y, weights=trimmed_weights)
            else:
                y_center = np.median(filtered_y)
        else:
            y_center = np.median(filtered_y)

        # Compute extent using balanced percentiles (3rd and 97th)
        # Good balance between capturing object extent and avoiding outliers
        sorted_idx = np.argsort(filtered_y)
        sorted_y = filtered_y[sorted_idx]
        sorted_weights = filtered_weights[sorted_idx]
        cumsum_weights = np.cumsum(sorted_weights)
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0

        p3_idx = np.searchsorted(cumsum_weights, 0.03)
        p97_idx = np.searchsorted(cumsum_weights, 0.97)

        y_min = sorted_y[p3_idx]
        y_max = sorted_y[p97_idx]

        # Ensure minimum height of 2cm
        min_height = 0.02
        if (y_max - y_min) < min_height:
            half_min = min_height / 2
            y_min = y_center - half_min
            y_max = y_center + half_min

        return float(y_center), float(y_min), float(y_max)

    @staticmethod
    def _compute_width_bounds(
        valid_coords: np.ndarray,
        valid_depths: np.ndarray,
        spatial_weights: np.ndarray,
        depth_info: CameraInfo,
    ) -> Tuple[float, float, float]:
        """
        Compute 3D width (x-axis) statistics from valid depth points.
        Uses actual 3D point positions instead of just projecting 2D bbox.

        Args:
            valid_coords: Nx2 array of pixel coordinates [x, y]
            valid_depths: N array of depth values in meters
            spatial_weights: N array of spatial weights
            depth_info: Camera intrinsic parameters

        Returns:
            Tuple of (x_center, x_min, x_max) in meters
        """
        # Input validations
        try:
            valid_depths = np.asarray(valid_depths, dtype=np.float64)
            spatial_weights = np.asarray(spatial_weights, dtype=np.float64)
        except (ValueError, TypeError):
            return 0.0, 0.0, 0.0

        if len(valid_coords) == 0 or len(valid_depths) == 0:
            return 0.0, 0.0, 0.0

        if len(valid_coords) < 4:
            # Fallback: just use simple projection
            k = depth_info.k
            px, fx = k[2], k[0]

            # Validate camera parameters
            if fx == 0:
                return 0.0, 0.0, 0.0

            # Validate depths are finite
            if not np.all(np.isfinite(valid_depths)):
                return 0.0, 0.0, 0.0

            x_coords_pixel = valid_coords[:, 0]
            x_3d = valid_depths * (x_coords_pixel - px) / fx

            # Validate result
            if not np.all(np.isfinite(x_3d)):
                return 0.0, 0.0, 0.0

            return float(np.median(x_3d)), float(np.min(x_3d)), float(np.max(x_3d))

        # Convert pixel coordinates to 3D x-coordinates
        k = depth_info.k
        px, fx = k[2], k[0]

        # Validate camera parameters
        if fx == 0:
            return 0.0, 0.0, 0.0

        # Validate depths are finite before calculation
        if not np.all(np.isfinite(valid_depths)):
            return 0.0, 0.0, 0.0

        x_coords_pixel = valid_coords[:, 0]
        x_3d = valid_depths * (x_coords_pixel - px) / fx

        # Validate result
        if not np.any(np.isfinite(x_3d)):
            return 0.0, 0.0, 0.0

        # Filter outliers using robust statistics
        # Compute weighted median as reference
        sorted_idx = np.argsort(x_3d)
        sorted_x = x_3d[sorted_idx]
        sorted_weights = spatial_weights[sorted_idx]
        cumsum_weights = np.cumsum(sorted_weights)
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0
        median_idx = np.searchsorted(cumsum_weights, 0.5)
        x_median = sorted_x[median_idx]

        # Compute MAD (Median Absolute Deviation)
        deviations = np.abs(x_3d - x_median)
        mad = np.median(deviations)

        # Adaptive threshold based on depth variance (helps with occlusions)
        # Check if object has varying depth (might indicate occlusion)
        depth_std = np.std(valid_depths)
        if depth_std > 0.15:  # High depth variation - likely occlusion or 3D object
            # Use tighter threshold to avoid including background
            threshold = np.clip(4.0 * mad, 0.06, 0.40)
        else:  # Uniform depth - flat object
            # Can be more permissive
            threshold = np.clip(4.5 * mad, 0.08, 0.50)

        valid_mask = deviations <= threshold
        filtered_x = x_3d[valid_mask]
        filtered_weights = spatial_weights[valid_mask]

        # Ensure we have enough points (at least 12% of data)
        if len(filtered_x) < max(4, len(x_3d) * 0.12):
            filtered_x = x_3d
            filtered_weights = spatial_weights

        # Compute weighted center using trimmed mean
        sorted_idx = np.argsort(filtered_x)
        sorted_x = filtered_x[sorted_idx]
        sorted_weights = filtered_weights[sorted_idx]
        cumsum_weights = np.cumsum(sorted_weights)
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0

        # Trim 5% from each end for robust center estimation
        trim_low_idx = np.searchsorted(cumsum_weights, 0.05)
        trim_high_idx = np.searchsorted(cumsum_weights, 0.95)

        if trim_high_idx > trim_low_idx:
            trimmed_x = sorted_x[trim_low_idx:trim_high_idx]
            trimmed_weights = sorted_weights[trim_low_idx:trim_high_idx]
            if np.sum(trimmed_weights) > 0:
                x_center = np.average(trimmed_x, weights=trimmed_weights)
            else:
                x_center = np.median(filtered_x)
        else:
            x_center = np.median(filtered_x)

        # Compute extent using balanced percentiles (3rd and 97th)
        # Good balance between capturing object extent and avoiding outliers
        sorted_idx = np.argsort(filtered_x)
        sorted_x = filtered_x[sorted_idx]
        sorted_weights = filtered_weights[sorted_idx]
        cumsum_weights = np.cumsum(sorted_weights)
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0

        p3_idx = np.searchsorted(cumsum_weights, 0.03)
        p97_idx = np.searchsorted(cumsum_weights, 0.97)

        x_min = sorted_x[p3_idx]
        x_max = sorted_x[p97_idx]

        # Ensure minimum width of 2cm
        min_width = 0.02
        if (x_max - x_min) < min_width:
            half_min = min_width / 2
            x_min = x_center - half_min
            x_max = x_center + half_min

        return float(x_center), float(x_min), float(x_max)

    @staticmethod
    def _compute_depth_bounds_weighted(
        depth_values: np.ndarray, spatial_weights: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Compute robust depth statistics with spatial weighting to handle occlusions.

        Args:
            depth_values: 1D array of valid depth values (> 0)
            spatial_weights: 1D array of spatial weights (0-1) for each depth

        Returns:
            Tuple of (z_center, z_min, z_max) representing the object's depth
        """
        # Input validations
        try:
            depth_values = np.asarray(depth_values, dtype=np.float64)
            spatial_weights = np.asarray(spatial_weights, dtype=np.float64)
        except (ValueError, TypeError):
            return 0.0, 0.0, 0.0

        if len(depth_values) == 0:
            return 0.0, 0.0, 0.0

        # Validate that all values are finite
        valid_mask = np.isfinite(depth_values) & np.isfinite(spatial_weights)
        depth_values = depth_values[valid_mask]
        spatial_weights = spatial_weights[valid_mask]

        if len(depth_values) == 0:
            return 0.0, 0.0, 0.0

        if len(depth_values) < 4:
            z_center = float(np.median(depth_values))
            return z_center, float(np.min(depth_values)), float(np.max(depth_values))

        # Step 1: Multi-scale histogram analysis for robust mode detection
        depth_range = np.ptp(depth_values)
        if not np.isfinite(depth_range) or depth_range <= 0:
            n_bins = 30
        else:
            n_bins = max(20, min(60, int(depth_range / 0.01)))

        # Create weighted histogram
        hist, bin_edges = np.histogram(depth_values, bins=n_bins, weights=spatial_weights)

        # Smooth histogram to reduce noise while preserving peaks
        if len(hist) >= 5:
            # Simple moving average smoothing
            kernel_size = min(5, len(hist) // 4)
            kernel = np.ones(kernel_size) / kernel_size
            hist_smooth = np.convolve(hist, kernel, mode="same")
        else:
            hist_smooth = hist

        # Find peak (mode) - highest weighted density region
        peak_bin_idx = np.argmax(hist_smooth)
        mode_depth = (bin_edges[peak_bin_idx] + bin_edges[peak_bin_idx + 1]) / 2

        # Step 2: Adaptive outlier filtering with less aggressive thresholds
        deviations = np.abs(depth_values - mode_depth)

        # Compute robust MAD without inverse weighting to avoid over-filtering
        mad = np.median(deviations)

        # More permissive threshold - adjust based on object size and uniformity
        # Check depth distribution uniformity
        q25 = np.percentile(depth_values, 25)
        q75 = np.percentile(depth_values, 75)
        iqr = q75 - q25

        # Adaptive threshold: looser for varied depth, tighter for uniform
        if iqr < 0.03:  # Very uniform depth (<3cm IQR)
            # For flat objects, use tighter bounds
            threshold = np.clip(3.5 * mad, 0.08, 0.30)
        elif iqr < 0.10:  # Moderate variation (<10cm IQR)
            # Standard threshold
            threshold = np.clip(4.0 * mad, 0.12, 0.40)
        else:  # High variation (>10cm IQR)
            # For complex 3D objects, use very permissive bounds
            threshold = np.clip(5.0 * mad, 0.15, 0.60)

        # Keep depths within threshold
        object_mask = deviations <= threshold
        object_depths = depth_values[object_mask]
        object_weights = spatial_weights[object_mask]

        # Fallback if filtering was too aggressive
        min_points = max(6, int(len(depth_values) * 0.15))  # Keep at least 15% of points
        if len(object_depths) < min_points:
            # Use weighted percentiles with wider range
            sorted_idx = np.argsort(depth_values)
            cumsum_weights = np.cumsum(spatial_weights[sorted_idx])
            cumsum_weights /= cumsum_weights[-1]

            # Find 2nd and 85th weighted percentiles (wider range)
            p2_idx = np.searchsorted(cumsum_weights, 0.02)
            p85_idx = np.searchsorted(cumsum_weights, 0.85)

            p2_val = depth_values[sorted_idx[p2_idx]]
            p85_val = depth_values[sorted_idx[p85_idx]]

            object_mask = (depth_values >= p2_val) & (depth_values <= p85_val)
            object_depths = depth_values[object_mask]
            object_weights = spatial_weights[object_mask]

        if len(object_depths) == 0:
            object_depths = depth_values
            object_weights = spatial_weights

        # Step 3: Compute robust weighted center using trimmed mean
        if np.sum(object_weights) > 0:
            # Use weighted average, but trim extreme 2% on each side first
            sorted_idx = np.argsort(object_depths)
            sorted_depths = object_depths[sorted_idx]
            sorted_weights = object_weights[sorted_idx]

            cumsum_weights = np.cumsum(sorted_weights)
            cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0

            # Trim 2% from each end
            trim_low_idx = np.searchsorted(cumsum_weights, 0.02)
            trim_high_idx = np.searchsorted(cumsum_weights, 0.98)

            if trim_high_idx > trim_low_idx:
                trimmed_depths = sorted_depths[trim_low_idx:trim_high_idx]
                trimmed_weights = sorted_weights[trim_low_idx:trim_high_idx]

                if np.sum(trimmed_weights) > 0:
                    z_center = np.average(trimmed_depths, weights=trimmed_weights)
                else:
                    z_center = np.median(object_depths)
            else:
                z_center = np.average(object_depths, weights=object_weights)
        else:
            z_center = np.median(object_depths)

        # Step 4: Compute extent using balanced weighted percentiles
        sorted_idx = np.argsort(object_depths)
        cumsum_weights = np.cumsum(object_weights[sorted_idx])
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0

        # Use 1st and 99th percentiles for depth (slightly more coverage than width/height)
        p1_idx = np.searchsorted(cumsum_weights, 0.01)
        p99_idx = np.searchsorted(cumsum_weights, 0.99)

        z_min = object_depths[sorted_idx[p1_idx]]
        z_max = object_depths[sorted_idx[p99_idx]]

        # Validate and adjust bounds relative to center
        # Ensure center is within bounds (sanity check)
        if z_center < z_min or z_center > z_max:
            # Recompute bounds symmetrically around center
            depth_extent = max(z_max - z_min, 0.02)  # At least 2cm
            z_min = z_center - depth_extent / 2
            z_max = z_center + depth_extent / 2

        # Ensure minimum depth size of 2cm (more realistic for real objects)
        min_depth_size = 0.02
        if (z_max - z_min) < min_depth_size:
            # Expand around center
            half_min = min_depth_size / 2
            z_min = z_center - half_min
            z_max = z_center + half_min

        return float(z_center), float(z_min), float(z_max)

    def convert_keypoints_to_3d(
        self,
        depth_image: np.ndarray,
        depth_info: CameraInfo,
        detection: Detection,
    ) -> KeyPoint3DArray:
        """
        Convert 2D keypoints to 3D using depth information.

        Samples depth at keypoint locations and projects to 3D coordinates.

        @param depth_image Depth image as numpy array
        @param depth_info Camera intrinsic parameters
        @param detection Detection containing 2D keypoints
        @return Array of 3D keypoints
        """
        # Validate input
        if depth_image is None or not isinstance(depth_image, np.ndarray):
            return KeyPoint3DArray()

        # Build an array of 2D keypoints
        keypoints_2d = np.array(
            [[p.point.x, p.point.y] for p in detection.keypoints.data], dtype=np.int16
        )
        u = np.array(keypoints_2d[:, 1]).clip(0, depth_info.height - 1)
        v = np.array(keypoints_2d[:, 0]).clip(0, depth_info.width - 1)

        # Sample depth image and project to 3D
        z = depth_image[u, v]

        # Validate and convert to float
        try:
            z = np.asarray(z, dtype=np.float64)
        except (ValueError, TypeError):
            return KeyPoint3DArray()

        k = depth_info.k
        px, py, fx, fy = k[2], k[5], k[0], k[4]

        # Validate camera parameters
        if fx == 0 or fy == 0:
            return KeyPoint3DArray()

        x = z * (v - px) / fx
        y = z * (u - py) / fy
        points_3d = (
            np.dstack([x, y, z]).reshape(-1, 3) / self.depth_image_units_divisor
        )  # Convert to meters

        # Generate message
        msg_array = KeyPoint3DArray()
        for p, d in zip(points_3d, detection.keypoints.data):
            if not np.isnan(p).any() and np.all(np.isfinite(p)):
                msg = KeyPoint3D()
                msg.point.x = float(p[0])
                msg.point.y = float(p[1])
                msg.point.z = float(p[2])
                msg.id = d.id
                msg.score = d.score
                msg_array.data.append(msg)

        return msg_array

    def get_transform(self, frame_id: str) -> Tuple[np.ndarray]:
        """
        Get TF transform from source frame to target frame.

        Looks up the transform from the camera frame to the configured target frame.

        @param frame_id Source frame ID (usually camera frame)
        @return Tuple of (translation, rotation) as numpy arrays, or None if transform fails
        """
        # Transform position from image frame to target_frame
        rotation = None
        translation = None

        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.target_frame, frame_id, rclpy.time.Time()
            )

            translation = np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ]
            )

            rotation = np.array(
                [
                    transform.transform.rotation.w,
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                ]
            )

            return translation, rotation

        except TransformException as ex:
            self.get_logger().error(f"Could not transform: {ex}")
            return None

    @staticmethod
    def transform_3d_box(
        bbox: BoundingBox3D,
        translation: np.ndarray,
        rotation: np.ndarray,
    ) -> BoundingBox3D:
        """
        Transform a 3D bounding box to a different reference frame.

        Applies rotation and translation to both position and size of the bbox.

        @param bbox Bounding box to transform
        @param translation Translation vector
        @param rotation Rotation quaternion [w, x, y, z]
        @return Transformed bounding box
        """

        # Position
        position = (
            Detect3DNode.qv_mult(
                rotation,
                np.array(
                    [
                        bbox.center.position.x,
                        bbox.center.position.y,
                        bbox.center.position.z,
                    ]
                ),
            )
            + translation
        )

        bbox.center.position.x = position[0]
        bbox.center.position.y = position[1]
        bbox.center.position.z = position[2]

        # Size (only rotation, no translation)
        size = Detect3DNode.qv_mult(
            rotation, np.array([bbox.size.x, bbox.size.y, bbox.size.z])
        )

        bbox.size.x = abs(size[0])
        bbox.size.y = abs(size[1])
        bbox.size.z = abs(size[2])

        return bbox

    @staticmethod
    def transform_3d_keypoints(
        keypoints: KeyPoint3DArray,
        translation: np.ndarray,
        rotation: np.ndarray,
    ) -> KeyPoint3DArray:
        """
        Transform 3D keypoints to a different reference frame.

        Applies rotation and translation to each keypoint position.

        @param keypoints Array of keypoints to transform
        @param translation Translation vector
        @param rotation Rotation quaternion [w, x, y, z]
        @return Transformed keypoint array
        """

        for point in keypoints.data:
            position = (
                Detect3DNode.qv_mult(
                    rotation, np.array([point.point.x, point.point.y, point.point.z])
                )
                + translation
            )

            point.point.x = position[0]
            point.point.y = position[1]
            point.point.z = position[2]

        return keypoints

    @staticmethod
    def qv_mult(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Multiply a quaternion with a vector (rotate vector by quaternion).

        Performs quaternion-vector multiplication to rotate a 3D vector.

        @param q Quaternion [w, x, y, z]
        @param v 3D vector [x, y, z]
        @return Rotated vector
        """
        q = np.array(q, dtype=np.float64)
        v = np.array(v, dtype=np.float64)
        qvec = q[1:]
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        return v + 2 * (uv * q[0] + uuv)


def main():
    rclpy.init()
    node = Detect3DNode()
    node.trigger_configure()
    node.trigger_activate()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
