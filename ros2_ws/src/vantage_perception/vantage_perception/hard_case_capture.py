"""Capture hard cases to S3 to build a retraining dataset.

Any frame the edge model was uncertain about is worth keeping: those are the
examples that will most improve the next model. Uploading them to S3, keyed by
date and confidence, gives you a clean data flywheel. Over time you fine-tune
the edge model on its own hard cases and the escalation rate (and cloud cost)
falls.

Uploads run on a background thread so capture never blocks the perception loop.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import List, Optional

from vantage_perception.detection import Detection


class HardCaseCapture:
    def __init__(self, bucket: str, prefix: str = "hard-cases",
                 region: str = "us-west-2", enabled: bool = True):
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")
        self._enabled = enabled and bool(bucket)
        self._q: "queue.Queue" = queue.Queue(maxsize=100)
        self.count = 0

        if self._enabled:
            import boto3
            self._s3 = boto3.client("s3", region_name=region)
            self._worker = threading.Thread(target=self._drain, daemon=True)
            self._worker.start()

    def submit(self, jpeg_bytes: bytes, device_id: str,
               edge_dets: List[Detection], cloud_dets: Optional[List[Detection]] = None) -> None:
        if not self._enabled:
            return
        item = {
            "jpeg": jpeg_bytes,
            "device_id": device_id,
            "ts": int(time.time() * 1000),
            "edge": [d.to_dict() for d in edge_dets],
            "cloud": [d.to_dict() for d in cloud_dets] if cloud_dets else None,
        }
        try:
            self._q.put_nowait(item)
            self.count += 1
        except queue.Full:
            # Drop under pressure rather than stall perception. This is a
            # best-effort dataset, not a transactional log.
            pass

    def _drain(self) -> None:
        while True:
            item = self._q.get()
            try:
                self._upload(item)
            except Exception:
                pass
            finally:
                self._q.task_done()

    def _upload(self, item: dict) -> None:
        from datetime import datetime, timezone
        dt = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        base = f"{self._prefix}/dt={dt}/{item['device_id']}/{item['ts']}"

        self._s3.put_object(
            Bucket=self._bucket, Key=f"{base}.jpg",
            Body=item["jpeg"], ContentType="image/jpeg",
        )
        meta = {k: v for k, v in item.items() if k != "jpeg"}
        self._s3.put_object(
            Bucket=self._bucket, Key=f"{base}.json",
            Body=json.dumps(meta), ContentType="application/json",
        )
