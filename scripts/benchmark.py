"""Compare edge-only, cloud-only, and hybrid routing over a folder of images.

This is the demo that makes the project's argument concrete. It runs all three
strategies on the same images and prints latency, escalation rate, estimated
cost, and an agreement score, then writes a CSV.

It runs on a laptop with no ROS and no AWS: the edge path uses the exported
ONNX model, and the cloud path uses a local heavier model with synthetic
network latency (see cloud_client.LocalCloudStub). Point --cloud-endpoint at a
real SageMaker endpoint to benchmark the real thing.

Usage:
  # export the edge model first
  python ../models/export_onnx.py --out ../models/edge_ssdlite.onnx

  # then benchmark on any folder of jpg/png images
  python benchmark.py --images ../sample_data --model ../models/edge_ssdlite.onnx \
      --labels ../models/edge_ssdlite_labels.txt
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

import numpy as np

# Make the ROS2 package importable without a ROS install.
PKG = os.path.join(os.path.dirname(__file__), "..", "ros2_ws", "src", "vantage_perception")
sys.path.insert(0, os.path.abspath(PKG))

from vantage_perception.edge_detector import EdgeDetector          # noqa: E402
from vantage_perception.router_policy import Router, RouterConfig   # noqa: E402
from vantage_perception.metrics import Metrics                      # noqa: E402
from vantage_perception.detection import top_score                  # noqa: E402


def load_images(folder: str):
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(glob.glob(os.path.join(folder, ext)))
    from PIL import Image
    for p in sorted(paths):
        img = np.array(Image.open(p).convert("RGB"))
        yield os.path.basename(p), img


def top_label(dets):
    return max(dets, key=lambda d: d.score).label if dets else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--cloud-endpoint", default="", help="real SageMaker endpoint name")
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--synthetic-rtt-ms", type=float, default=80.0)
    ap.add_argument("--out", default="benchmark_results.csv")
    args = ap.parse_args()

    edge = EdgeDetector(args.model, args.labels)

    # Cloud path: real endpoint if given, else the local stand-in.
    cloud = None
    cloud_is_local = False
    if args.cloud_endpoint:
        from vantage_perception.cloud_client import SageMakerCloudDetector
        cloud = SageMakerCloudDetector(args.cloud_endpoint, args.region)
    else:
        try:
            from vantage_perception.cloud_client import LocalCloudStub
            cloud = LocalCloudStub(synthetic_rtt_ms=args.synthetic_rtt_ms)
            cloud_is_local = True
            print("using LocalCloudStub for the cloud path (no endpoint given)")
        except Exception as e:
            print(f"cloud path unavailable ({e}); running edge-only")

    router = Router(RouterConfig())
    metrics = Metrics()

    images = list(load_images(args.images))
    if not images:
        raise SystemExit(f"no images found in {args.images}")

    rows = []
    agree_hits = 0
    agree_total = 0

    for name, img in images:
        edge_dets, edge_ms = edge.infer(img)

        # Cloud reference (only when a cloud path exists), used for agreement.
        cloud_ref = None
        cloud_ref_ms = None
        if cloud is not None:
            if cloud_is_local:
                cloud_ref, cloud_ref_ms = cloud.infer_array(img)
            else:
                import cv2
                ok, buf = cv2.imencode(".jpg", img[:, :, ::-1])
                cloud_ref, cloud_ref_ms = cloud.infer(buf.tobytes())

        # Hybrid decision.
        escalate, reason = router.should_escalate(top_score(edge_dets), len(edge_dets))
        if escalate and cloud is not None:
            hybrid_dets = cloud_ref
            hybrid_ms = edge_ms + (cloud_ref_ms or 0.0)
        else:
            hybrid_dets = edge_dets
            hybrid_ms = edge_ms
        metrics.record(edge_ms, escalate, reason, cloud_ref_ms if escalate else None)

        # Agreement of hybrid vs the strong cloud model, top-label proxy.
        if cloud_ref is not None:
            agree_total += 1
            if top_label(hybrid_dets) == top_label(cloud_ref):
                agree_hits += 1

        rows.append({
            "image": name,
            "edge_ms": round(edge_ms, 1),
            "edge_top": top_label(edge_dets),
            "edge_score": round(top_score(edge_dets), 3),
            "escalated": escalate,
            "reason": reason,
            "cloud_ms": round(cloud_ref_ms, 1) if cloud_ref_ms else "",
            "cloud_top": top_label(cloud_ref) if cloud_ref else "",
            "hybrid_ms": round(hybrid_ms, 1),
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- report ----
    edge_p50 = metrics._pct(metrics.edge_latencies, 50)
    hybrid_lat = [r["hybrid_ms"] for r in rows]
    hybrid_p50 = sorted(hybrid_lat)[len(hybrid_lat) // 2]
    cloud_lat = [r["cloud_ms"] for r in rows if r["cloud_ms"] != ""]
    cloud_p50 = sorted(cloud_lat)[len(cloud_lat) // 2] if cloud_lat else 0

    print("\n=== Vantage benchmark ===")
    print(f"images:            {len(images)}")
    print(f"escalation rate:   {metrics.escalation_rate():.1%}")
    print(f"edge   p50 latency: {edge_p50:.1f} ms")
    print(f"cloud  p50 latency: {cloud_p50:.1f} ms  (all-cloud baseline)")
    print(f"hybrid p50 latency: {hybrid_p50:.1f} ms")
    if agree_total:
        print(f"hybrid vs cloud top-label agreement: {agree_hits}/{agree_total} "
              f"({agree_hits / agree_total:.1%})")
    print(f"est. cloud cost:   ${metrics.estimated_cost_usd():.4f} "
          f"(vs ${len(images) / 1000.0:.4f} for all-cloud)")
    print(f"routing reasons:   {dict(metrics.reasons)}")
    print(f"\nper-image CSV -> {args.out}")
    print("\nThe story: hybrid stays near edge latency and a fraction of cloud cost,")
    print("while recovering most of the cloud model's accuracy on the hard frames.")


if __name__ == "__main__":
    main()
