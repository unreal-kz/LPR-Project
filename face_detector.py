"""GPU-accelerated face detection using RetinaFace (InsightFace).

Requires:
    pip install insightface opencv-python

InsightFace will automatically download the RetinaFace model the first time it
is used and cache it under ``~/.insightface``.

Example
-------
>>> from face_detector import FaceDetector
>>> fd = FaceDetector(gpu_id=0)
>>> faces = fd.detect(frame)  # list of (x, y, w, h)
"""
from __future__ import annotations

import cv2
import numpy as np
import logging
import torch
from typing import List, Tuple

try:
    from insightface.app import FaceAnalysis
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "InsightFace is required for RetinaFace GPU detection. Install with `pip install insightface`.") from e

LOGGER = logging.getLogger(__name__)


class FaceDetector:
    """RetinaFace GPU detector using InsightFace library."""

    def __init__(self, gpu_id: int = 0, threshold: float = 0.5):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available ‑- RetinaFace requires CUDA device. Configure CUDA first.")

        self.detector = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        # det_size controls internal input resolution; default (640, 640)
        self.detector.prepare(ctx_id=gpu_id, det_size=(640, 640))
        self.threshold = threshold
        LOGGER.info("FaceAnalysis (RetinaFace) model loaded on GPU %d", gpu_id)

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces; returns list of (x, y, w, h)"""
        if frame is None or frame.size == 0:
            return []

        faces = self.detector.get(frame)
        bboxes = [f.bbox.tolist() + [f.det_score] for f in faces if f.det_score >= self.threshold]
        if bboxes is None or len(bboxes) == 0:
            return []

        boxes_xywh = []
        for x1, y1, x2, y2, _conf in bboxes:
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            boxes_xywh.append((x, y, w, h))
        return boxes_xywh
