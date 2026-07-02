"""Cloud detector: invoke a SageMaker real-time endpoint.

This is the heavy path. Higher accuracy model, but every call pays a network
round trip, so it is only worth it when the edge result is uncertain. The
client sends a JPEG-encoded image and gets back detections in the shared
format.

For local demos without a deployed endpoint, LocalCloudStub runs a heavier
torchvision model in-process and adds artificial network latency, so the
benchmark can still tell the edge-vs-cloud story on a laptop.
"""

from __future__ import annotations

import io
import json
import time
from typing import List

import numpy as np

from vantage_perception.detection import Detection


class SageMakerCloudDetector:
    def __init__(self, endpoint_name: str, region: str = "us-west-2"):
        import boto3  # imported here so the edge path never needs boto3
        self._client = boto3.client("sagemaker-runtime", region_name=region)
        self._endpoint = endpoint_name

    def infer(self, jpeg_bytes: bytes) -> tuple[List[Detection], float]:
        start = time.perf_counter()
        resp = self._client.invoke_endpoint(
            EndpointName=self._endpoint,
            ContentType="image/jpeg",
            Body=jpeg_bytes,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        payload = json.loads(resp["Body"].read())
        dets = [Detection.from_dict(d) for d in payload.get("detections", [])]
        return dets, latency_ms


class LocalCloudStub:
    """Runs fasterrcnn_resnet50_fpn locally to stand in for the cloud endpoint.

    Adds a configurable synthetic round-trip so latency comparisons in the
    benchmark are realistic even without deploying to AWS.
    """

    def __init__(self, synthetic_rtt_ms: float = 80.0, score_thresh: float = 0.4):
        import torch
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn,
            FasterRCNN_ResNet50_FPN_Weights,
        )
        self._torch = torch
        self._weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self._model = fasterrcnn_resnet50_fpn(weights=self._weights)
        self._model.eval()
        self._labels = self._weights.meta["categories"]
        self._thresh = score_thresh
        self._rtt = synthetic_rtt_ms / 1000.0

    def infer_array(self, image: np.ndarray) -> tuple[List[Detection], float]:
        torch = self._torch
        start = time.perf_counter()
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            out = self._model([tensor])[0]
        time.sleep(self._rtt)  # simulate the network the real endpoint would add
        latency_ms = (time.perf_counter() - start) * 1000.0

        dets: List[Detection] = []
        for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
            s = float(score)
            if s < self._thresh:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            idx = int(label)
            name = self._labels[idx] if 0 <= idx < len(self._labels) else str(idx)
            dets.append(Detection(label=name, score=s, box=[x1, y1, x2, y2]))
        return dets, latency_ms
