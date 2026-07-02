"""The core of Vantage: decide, per frame, whether the edge result is good
enough or whether to escalate to the cloud model.

Kept as a pure module (no ROS, no AWS) so it can be unit tested and reused by
both the ROS node and the offline benchmark. The whole project is really an
argument about this one function.

Policy inputs and the reasoning:

- edge_top_score: confidence of the edge model's best detection.
    * high  -> trust the edge, do not pay for the cloud.
    * middle (the ambiguous band) -> escalate: this is exactly where the small
      model is unreliable and the big model earns its cost.
    * very low with an otherwise non-empty scene -> escalate, the edge may have
      missed something.
- budget: a token-bucket rate limit on cloud calls, so a hard scene cannot melt
  your SageMaker bill. Under load, you degrade to edge-only rather than queueing.
- latency_budget_ms: if the control loop cannot wait for a round trip this
  frame, stay on the edge. Safety-critical loops never block on the cloud.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class RouterConfig:
    conf_high: float = 0.70      # above this, edge is trusted outright
    conf_low: float = 0.30       # below this (with detections), edge is doubtful
    cloud_calls_per_sec: float = 5.0   # token-bucket refill rate
    cloud_burst: int = 10        # bucket capacity
    latency_budget_ms: float = 250.0   # if a frame cannot spare this, stay edge


class TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last = time.monotonic()

    def try_take(self) -> bool:
        now = time.monotonic()
        self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
        self._last = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


class Router:
    def __init__(self, config: RouterConfig | None = None):
        self.cfg = config or RouterConfig()
        self._bucket = TokenBucket(self.cfg.cloud_calls_per_sec, self.cfg.cloud_burst)

    def should_escalate(self, edge_top_score: float, edge_num_dets: int,
                        frame_latency_budget_ms: float | None = None) -> tuple[bool, str]:
        """Return (escalate?, reason). Reason is logged for the metrics story."""
        budget = frame_latency_budget_ms if frame_latency_budget_ms is not None \
            else self.cfg.latency_budget_ms

        # 1. Hard latency gate. Never block a tight control loop on the network.
        if budget < self.cfg.latency_budget_ms:
            return False, "latency_budget_too_tight"

        # 2. Confidence gate.
        confident = edge_top_score >= self.cfg.conf_high
        doubtful = (edge_num_dets > 0 and edge_top_score < self.cfg.conf_low) \
            or (self.cfg.conf_low <= edge_top_score < self.cfg.conf_high)

        if confident:
            return False, "edge_confident"
        if not doubtful and edge_num_dets == 0:
            # Empty scene at very low score: nothing to escalate for.
            return False, "edge_empty_scene"

        # 3. Budget gate. Uncertain, but the cloud must be affordable this second.
        if not self._bucket.try_take():
            return False, "cloud_budget_exhausted"

        return True, "escalated_uncertain"
