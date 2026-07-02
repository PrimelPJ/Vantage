"""Rolling metrics for the perception pipeline.

Tracks per-source latency percentiles, how often the router escalated, and a
running cost estimate. These are the numbers you put on a slide: "hybrid gets
within X percent of cloud accuracy at Y percent of the cost and Z ms median
latency."
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

# Rough public pricing used only for an order-of-magnitude cost estimate.
# Edge inference is treated as free (already-owned robot compute).
CLOUD_COST_PER_1K_INVOCATIONS_USD = 1.0  # tune to your instance + throughput


@dataclass
class Metrics:
    edge_latencies: List[float] = field(default_factory=list)
    cloud_latencies: List[float] = field(default_factory=list)
    frames: int = 0
    escalations: int = 0
    reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    hard_cases_captured: int = 0

    def record(self, edge_ms: float, escalated: bool, reason: str,
               cloud_ms: float | None = None) -> None:
        self.frames += 1
        self.edge_latencies.append(edge_ms)
        self.reasons[reason] += 1
        if escalated:
            self.escalations += 1
            if cloud_ms is not None:
                self.cloud_latencies.append(cloud_ms)

    @staticmethod
    def _pct(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        k = int(round((p / 100.0) * (len(s) - 1)))
        return s[k]

    def escalation_rate(self) -> float:
        return self.escalations / self.frames if self.frames else 0.0

    def estimated_cost_usd(self) -> float:
        return self.escalations / 1000.0 * CLOUD_COST_PER_1K_INVOCATIONS_USD

    def summary(self) -> dict:
        return {
            "frames": self.frames,
            "escalation_rate": round(self.escalation_rate(), 3),
            "edge_p50_ms": round(self._pct(self.edge_latencies, 50), 1),
            "edge_p95_ms": round(self._pct(self.edge_latencies, 95), 1),
            "cloud_p50_ms": round(self._pct(self.cloud_latencies, 50), 1),
            "cloud_p95_ms": round(self._pct(self.cloud_latencies, 95), 1),
            "hard_cases_captured": self.hard_cases_captured,
            "estimated_cloud_cost_usd": round(self.estimated_cost_usd(), 4),
            "reasons": dict(self.reasons),
        }
