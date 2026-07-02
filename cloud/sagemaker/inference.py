"""SageMaker inference handler for the Vantage cloud detector.

Deployed as a PyTorch model on a real-time endpoint. The four functions below
are the SageMaker inference contract:

  model_fn    load the model once when the container starts
  input_fn    deserialize the request body (a JPEG) into a tensor
  predict_fn  run inference
  output_fn   serialize detections back to the shared JSON format

The cloud model is fasterrcnn_resnet50_fpn: heavier and more accurate than the
edge SSDLite, which is the entire reason escalation is worth its latency and
cost. Both return the same COCO label space so edge and cloud are comparable.
"""

from __future__ import annotations

import io
import json

import torch
from PIL import Image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
import torchvision.transforms.functional as F

SCORE_THRESH = 0.4
_WEIGHTS = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
_LABELS = _WEIGHTS.meta["categories"]


def model_fn(model_dir):
    model = fasterrcnn_resnet50_fpn(weights=_WEIGHTS)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return model


def input_fn(request_body, content_type="image/jpeg"):
    if content_type != "image/jpeg":
        raise ValueError(f"unsupported content type: {content_type}")
    image = Image.open(io.BytesIO(request_body)).convert("RGB")
    tensor = F.to_tensor(image)
    return tensor


def predict_fn(tensor, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        out = model([tensor.to(device)])[0]
    return out


def output_fn(prediction, accept="application/json"):
    dets = []
    for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        s = float(score)
        if s < SCORE_THRESH:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        idx = int(label)
        name = _LABELS[idx] if 0 <= idx < len(_LABELS) else str(idx)
        dets.append({"label": name, "score": s, "box": [x1, y1, x2, y2]})
    return json.dumps({"detections": dets}), "application/json"
