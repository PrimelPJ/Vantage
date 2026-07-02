"""Shared detection format so edge and cloud results are directly comparable.

Both the edge ONNX detector and the cloud SageMaker endpoint return the same
Detection list. That symmetry is what lets the router treat them as
interchangeable and lets the benchmark measure agreement between them.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List


@dataclass
class Detection:
    label: str
    score: float
    box: List[float]  # [x1, y1, x2, y2] in pixel coords of the original image

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Detection":
        return Detection(label=d["label"], score=float(d["score"]), box=list(d["box"]))


def top_score(dets: List[Detection]) -> float:
    return max((d.score for d in dets), default=0.0)


def summarize(dets: List[Detection]) -> str:
    if not dets:
        return "no detections"
    return ", ".join(f"{d.label}:{d.score:.2f}" for d in sorted(dets, key=lambda x: -x.score)[:5])
