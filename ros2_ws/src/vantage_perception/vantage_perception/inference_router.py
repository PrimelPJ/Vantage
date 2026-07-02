"""Vantage perception router node.

Per camera frame:
  1. Run the edge ONNX detector (always).
  2. Ask the Router policy whether to escalate to the cloud.
  3. If yes and affordable, invoke the SageMaker endpoint.
  4. Publish the best available detections as vision_msgs/Detection2DArray.
  5. If the edge was uncertain, capture the frame to S3 for retraining.
  6. Periodically log metrics.

Run (edge-only, no cloud, no AWS):
  ros2 run vantage_perception router --ros-args \
    -p model_path:=/models/edge_ssdlite.onnx \
    -p labels_path:=/models/edge_ssdlite_labels.txt

Add cloud:
    -p sagemaker_endpoint:=vantage-detector \
    -p hard_case_bucket:=vantage-hard-cases-<acct>
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose

from vantage_perception.edge_detector import EdgeDetector
from vantage_perception.router_policy import Router, RouterConfig
from vantage_perception.metrics import Metrics
from vantage_perception.detection import top_score, summarize


def _jpeg_encode(image: np.ndarray) -> bytes:
    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", image[:, :, ::-1])  # RGB -> BGR for cv2
        return buf.tobytes() if ok else b""
    except ImportError:
        from io import BytesIO
        from PIL import Image as PILImage
        bio = BytesIO()
        PILImage.fromarray(image).save(bio, format="JPEG")
        return bio.getvalue()


class PerceptionRouter(Node):
    def __init__(self):
        super().__init__("vantage_router")

        self.declare_parameter("device_id", "robot-001")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("model_path", "")
        self.declare_parameter("labels_path", "")
        self.declare_parameter("sagemaker_endpoint", "")
        self.declare_parameter("hard_case_bucket", "")
        self.declare_parameter("region", "us-west-2")
        self.declare_parameter("conf_high", 0.70)
        self.declare_parameter("conf_low", 0.30)

        self._device_id = self.get_parameter("device_id").value
        model_path = self.get_parameter("model_path").value
        labels_path = self.get_parameter("labels_path").value or None

        self._edge = EdgeDetector(model_path, labels_path)
        self._router = Router(RouterConfig(
            conf_high=float(self.get_parameter("conf_high").value),
            conf_low=float(self.get_parameter("conf_low").value),
        ))
        self._metrics = Metrics()

        # Optional cloud path.
        endpoint = self.get_parameter("sagemaker_endpoint").value
        region = self.get_parameter("region").value
        self._cloud = None
        if endpoint:
            from vantage_perception.cloud_client import SageMakerCloudDetector
            self._cloud = SageMakerCloudDetector(endpoint, region)
            self.get_logger().info(f"cloud escalation enabled via {endpoint}")

        # Optional hard-case capture.
        bucket = self.get_parameter("hard_case_bucket").value
        self._capture = None
        if bucket:
            from vantage_perception.hard_case_capture import HardCaseCapture
            self._capture = HardCaseCapture(bucket, region=region)
            self.get_logger().info(f"hard-case capture -> s3://{bucket}")

        self._pub = self.create_publisher(Detection2DArray, "/vantage/detections", 10)
        self.create_subscription(
            Image, self.get_parameter("image_topic").value, self._on_image, 5)
        self.create_timer(10.0, self._log_metrics)
        self.get_logger().info("vantage router up")

    def _on_image(self, msg: Image) -> None:
        image = self._image_to_numpy(msg)
        if image is None:
            return

        edge_dets, edge_ms = self._edge.infer(image)
        escalate, reason = self._router.should_escalate(
            top_score(edge_dets), len(edge_dets))

        final = edge_dets
        cloud_ms = None
        cloud_dets = None
        if escalate and self._cloud is not None:
            jpeg = _jpeg_encode(image)
            try:
                cloud_dets, cloud_ms = self._cloud.infer(jpeg)
                final = cloud_dets  # trust the heavier model when consulted
            except Exception as e:
                self.get_logger().warn(f"cloud call failed, keeping edge: {e}")

        # Capture uncertain frames for the retraining flywheel.
        if self._capture is not None and reason in ("escalated_uncertain", "cloud_budget_exhausted"):
            self._capture.submit(_jpeg_encode(image), self._device_id, edge_dets, cloud_dets)
            self._metrics.hard_cases_captured = self._capture.count

        self._metrics.record(edge_ms, escalate, reason, cloud_ms)
        self._publish(final, msg.header)

    def _publish(self, dets, header) -> None:
        out = Detection2DArray()
        out.header = header
        for d in dets:
            det = Detection2D()
            det.header = header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = d.label
            hyp.hypothesis.score = d.score
            det.results.append(hyp)
            x1, y1, x2, y2 = d.box
            det.bbox.center.position.x = (x1 + x2) / 2.0
            det.bbox.center.position.y = (y1 + y2) / 2.0
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)
            out.detections.append(det)
        self._pub.publish(out)

    def _log_metrics(self) -> None:
        self.get_logger().info(f"metrics: {self._metrics.summary()}")

    @staticmethod
    def _image_to_numpy(msg: Image):
        """Convert sensor_msgs/Image (rgb8 or bgr8) to an HxWx3 RGB uint8 array."""
        if msg.encoding not in ("rgb8", "bgr8"):
            return None
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionRouter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
