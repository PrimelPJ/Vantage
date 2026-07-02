"""Deploy the Vantage cloud detector to a SageMaker real-time endpoint.

The model weights download at container start from the torchvision cache, so
the model.tar.gz only needs the inference code. For a production setup you would
bake the weights into the artifact instead of downloading at cold start.

Usage:
  python deploy_endpoint.py \
    --role arn:aws:iam::<acct>:role/<sagemaker-exec-role> \
    --endpoint vantage-detector \
    --instance ml.m5.xlarge
"""

from __future__ import annotations

import argparse
import tarfile
import tempfile
import os

import boto3


def build_model_tar(code_dir: str) -> str:
    """Package inference.py + requirements.txt into model.tar.gz under code/."""
    tmp = tempfile.mkdtemp()
    tar_path = os.path.join(tmp, "model.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for fname in ("inference.py", "requirements.txt"):
            tar.add(os.path.join(code_dir, fname), arcname=f"code/{fname}")
    return tar_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, help="SageMaker execution role ARN")
    ap.add_argument("--endpoint", default="vantage-detector")
    ap.add_argument("--instance", default="ml.m5.xlarge")
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--bucket", required=True, help="S3 bucket for the model artifact")
    args = ap.parse_args()

    try:
        import sagemaker
        from sagemaker.pytorch import PyTorchModel
    except ImportError:
        raise SystemExit("pip install sagemaker")

    session = sagemaker.Session(boto3.Session(region_name=args.region))
    code_dir = os.path.dirname(os.path.abspath(__file__))
    tar_path = build_model_tar(code_dir)

    s3_uri = session.upload_data(tar_path, bucket=args.bucket, key_prefix="vantage/model")
    print(f"uploaded artifact to {s3_uri}")

    model = PyTorchModel(
        model_data=s3_uri,
        role=args.role,
        entry_point="inference.py",
        framework_version="2.1",
        py_version="py310",
        sagemaker_session=session,
    )
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=args.instance,
        endpoint_name=args.endpoint,
    )
    print(f"endpoint live: {predictor.endpoint_name}")


if __name__ == "__main__":
    main()
