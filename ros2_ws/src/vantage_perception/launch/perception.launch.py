"""Launch the Vantage router. Cloud + capture are optional; leave the env
vars unset for an edge-only run."""
import os
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="vantage_perception",
            executable="router",
            name="vantage_router",
            output="screen",
            parameters=[{
                "device_id": os.environ.get("DEVICE_ID", "robot-001"),
                "image_topic": os.environ.get("IMAGE_TOPIC", "/camera/image_raw"),
                "model_path": os.environ.get("MODEL_PATH", "/models/edge_ssdlite.onnx"),
                "labels_path": os.environ.get("LABELS_PATH", "/models/edge_ssdlite_labels.txt"),
                "sagemaker_endpoint": os.environ.get("SAGEMAKER_ENDPOINT", ""),
                "hard_case_bucket": os.environ.get("HARD_CASE_BUCKET", ""),
                "region": os.environ.get("AWS_REGION", "us-west-2"),
            }],
        ),
    ])
