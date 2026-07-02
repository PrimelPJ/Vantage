"""Export a small object detector to ONNX for edge inference.

Edge model: SSDLite with a MobileNetV3 backbone, pretrained on COCO. It is
small and fast, which is the point: it runs on the robot's own compute with no
network round trip. It trades accuracy for latency, and the whole thesis of
Vantage is deciding, per frame, when that trade is acceptable and when to
escalate to the heavier cloud model.

Run:
  pip install torch torchvision onnx onnxruntime
  python export_onnx.py --out ../models/edge_ssdlite.onnx

Notes:
- torchvision detection models export with dynamic input height/width. We fix
  a 320x320 input here to keep edge latency predictable, which matters for a
  real-time perception loop.
- Outputs are boxes [N,4], labels [N], scores [N] in COCO label space.
"""

from __future__ import annotations

import argparse

import torch
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="edge_ssdlite.onnx")
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights, box_score_thresh=0.2)
    model.eval()

    # torchvision detection models take a list of tensors; for ONNX we trace a
    # single fixed-size image and treat it as batch of one.
    dummy = torch.rand(3, 320, 320)

    torch.onnx.export(
        model,
        [dummy],
        args.out,
        opset_version=args.opset,
        input_names=["image"],
        output_names=["boxes", "labels", "scores"],
        dynamic_axes={
            "boxes": {0: "num_detections"},
            "labels": {0: "num_detections"},
            "scores": {0: "num_detections"},
        },
    )

    # Save the COCO category names next to the model for postprocessing.
    categories = weights.meta["categories"]
    with open(args.out.replace(".onnx", "_labels.txt"), "w") as f:
        f.write("\n".join(categories))

    print(f"exported {args.out}")
    print(f"labels written to {args.out.replace('.onnx', '_labels.txt')}")


if __name__ == "__main__":
    main()
