"""Edge object detector backed by ONNX Runtime.

This is the on-robot inference path. No network, predictable latency, lower
accuracy. It is deliberately a plain class (not a ROS node) so the router can
call it in-process and the benchmark can call it without ROS at all.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

try:
    import onnxruntime as ort
    HAVE_ORT = True
except ImportError:
    HAVE_ORT = False

from vantage_perception.detection import Detection


class EdgeDetector:
    def __init__(self, model_path: str, labels_path: Optional[str] = None,
                 input_size: int = 320, score_thresh: float = 0.3):
        if not HAVE_ORT:
            raise RuntimeError("onnxruntime not installed. pip install onnxruntime")
        # CPUExecutionProvider is the honest default for a robot without a GPU.
        # On a Jetson you would add CUDA or TensorRT providers here.
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._size = input_size
        self._thresh = score_thresh
        self._labels = self._load_labels(labels_path)

    @staticmethod
    def _load_labels(path: Optional[str]) -> List[str]:
        if not path:
            return []
        with open(path) as f:
            return [line.strip() for line in f]

    def _preprocess(self, image: np.ndarray):
        """image: HxWx3 uint8 RGB. Returns CHW float32 and the scale factors."""
        h, w = image.shape[:2]
        resized = _resize(image, self._size, self._size)
        chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        scale_x = w / self._size
        scale_y = h / self._size
        return chw, scale_x, scale_y

    def infer(self, image: np.ndarray) -> tuple[List[Detection], float]:
        """Returns (detections, latency_ms)."""
        chw, sx, sy = self._preprocess(image)
        start = time.perf_counter()
        boxes, labels, scores = self._session.run(None, {self._input_name: chw})
        latency_ms = (time.perf_counter() - start) * 1000.0

        dets: List[Detection] = []
        for box, label, score in zip(boxes, labels, scores):
            if score < self._thresh:
                continue
            x1, y1, x2, y2 = box
            name = self._labels[int(label)] if 0 <= int(label) < len(self._labels) else str(int(label))
            dets.append(Detection(
                label=name,
                score=float(score),
                box=[float(x1 * sx), float(y1 * sy), float(x2 * sx), float(y2 * sy)],
            ))
        return dets, latency_ms


def _resize(image: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Tiny dependency-free nearest-neighbor resize so this file needs only numpy.
    Swap in cv2.resize for quality if OpenCV is already a dependency."""
    h, w = image.shape[:2]
    ys = (np.linspace(0, h - 1, out_h)).astype(np.int32)
    xs = (np.linspace(0, w - 1, out_w)).astype(np.int32)
    return image[ys][:, xs]
