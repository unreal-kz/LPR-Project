#!/usr/bin/env python3
"""Run YOLOv8 object detection on two RTSP cameras side-by-side.

This small utility re-uses the existing project dependencies (YOLO, FaceDetector)
without touching the single-camera class.  It loads one YOLO model instance and
processes two video streams in parallel inside a unified loop so that both
annotated frames are displayed in a single window (`Two-Camera YOLOv8`).

Usage (adjust RTSP URLs and settings near the bottom):
    python multi_rtsp_yolo.py

Press ESC to quit.
"""
from __future__ import annotations

import cv2
import time
import logging
from threading import Thread, Event
from queue import Queue, Empty
from typing import Tuple, List
from urllib.parse import urlparse

import torch
from ultralytics import YOLO

from face_detector import FaceDetector


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


class CameraWorker(Thread):
    """Read frames from RTSP camera, perform inference, and push annotated frame to queue."""

    def __init__(self,
                 name: str,
                 rtsp_url: str,
                 model: YOLO,
                 conf_threshold: float = 0.75,
                 class_ids: List[int] | None = None,
                 face_detector: FaceDetector | None = None,
                 skip_frames: int = 0,
                 output_q: Queue | None = None):
        super().__init__(name=name, daemon=True)
        self.rtsp_url = rtsp_url
        self.model = model
        self.conf = conf_threshold
        self.class_ids = class_ids or []
        self.face_detector = face_detector
        self.skip_frames = max(0, skip_frames)
        self.output_q: Queue = output_q or Queue(maxsize=5)
        self._stop_event = Event()
        self.cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    def _open(self) -> bool:
        """Try to open RTSP stream with some retry logic."""
        max_retries = 3
        retry_delay = 2
        for i in range(max_retries):
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if self.cap.isOpened():
                LOGGER.info("[%s] RTSP stream opened", self.name)
                return True
            LOGGER.warning("[%s] Retry %d/%d failed to open stream", self.name, i + 1, max_retries)
            time.sleep(retry_delay)
        LOGGER.error("[%s] Could not open RTSP stream", self.name)
        return False

    # ------------------------------------------------------------------
    def run(self):
        if not self._open():
            return
        frame_idx = 0
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            frame_idx += 1
            if not ret:
                LOGGER.warning("[%s] Empty frame; attempting reconnect", self.name)
                if not self._open():
                    break
                continue

            # Skip intermediate frames for performance
            if self.skip_frames and (frame_idx % (self.skip_frames + 1) != 0):
                continue

            # Inference
            results = self.model.predict(
                frame,
                imgsz=640,
                conf=self.conf,
                classes=self.class_ids if self.class_ids else None,

            )
            annotated = results[0].plot()

            # Face overlay
            if self.face_detector:
                faces = self.face_detector.detect(frame)
                for (x, y, w, h) in faces:
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Non-blocking put
            try:
                self.output_q.put_nowait(annotated)
            except:
                # queue full; drop frame to keep real-time
                pass

        # Cleanup
        if self.cap:
            self.cap.release()
        LOGGER.info("[%s] Stopped", self.name)

    # ------------------------------------------------------------------
    def stop(self):
        self._stop_event.set()


# ---------------------------------------------------------------------------

def concat_frames(frames: List[cv2.Mat], labels: List[str] | None = None) -> cv2.Mat:
    """Horizontally concat frames after resizing heights to match and optionally draw labels."""
    if not frames:
        return None
    if labels is None:
        labels = [""] * len(frames)
    # Determine minimum height to keep aspect ratio equalisation
    heights = [f.shape[0] for f in frames]
    min_h = min(heights)
    processed: List[cv2.Mat] = []
    for frame, label in zip(frames, labels):
        f_res = cv2.resize(frame, (int(frame.shape[1] * min_h / frame.shape[0]), min_h))
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(f_res, (0, 0), (tw + 12, th + 12), (0, 0, 0), -1)
            cv2.putText(f_res, label, (6, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        processed.append(f_res)
    return cv2.hconcat(processed)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ----------------------- CONFIGURATION ---------------------------------
    RTSP_1 = "rtsp://admin:admin@192.168.0.10:8554/CH001.sdp"
    RTSP_2 = "rtsp://admin:admin@192.168.0.11:8554/CH001.sdp"

    MODEL_PATH = "yolov8l.pt"
    CONF_THRESHOLD = 0.75
    DETECT_CLASSES = ["person", "backpack", "bottle", "cup", "cell phone", "book"]
    ENABLE_FACE = True
    SKIP_FRAMES = 3  # skip N frames between inference to save compute
    # -----------------------------------------------------------------------

    # Prepare device selection (reuse logic similar to other script)
    device = '0' if torch.cuda.is_available() else 'cpu'
    if device != 'cpu':
        LOGGER.info("Using GPU for inference: %s", torch.cuda.get_device_name(0))
    else:
        LOGGER.info("Using CPU for inference")

    # Load model once for both cameras
    try:
        model = YOLO(MODEL_PATH)  # device will be selected during predict internally
    except Exception as exc:
        LOGGER.error("Failed to load YOLO model: %s", exc)
        raise SystemExit(1)

    # Map class names to IDs
    if DETECT_CLASSES:
        name_to_id = {v: k for k, v in model.names.items()}
        class_ids: List[int] = []
        for c in DETECT_CLASSES:
            if isinstance(c, int):
                class_ids.append(c)
            elif isinstance(c, str):
                if c.isdigit():
                    class_ids.append(int(c))
                elif c in name_to_id:
                    class_ids.append(name_to_id[c])
                else:
                    LOGGER.warning("Unknown class name '%s' ignored", c)
        LOGGER.info("Detecting classes: %s", [model.names[i] for i in class_ids])
    else:
        class_ids = []

    face_detector = FaceDetector() if ENABLE_FACE else None

    # Labels derived from RTSP hostnames
    def label_from_rtsp(url: str) -> str:
        p = urlparse(url)
        return p.hostname or url

    LABELS = [label_from_rtsp(RTSP_1), label_from_rtsp(RTSP_2)]

    # Queues to receive annotated frames
    q1, q2 = Queue(maxsize=5), Queue(maxsize=5)

    # Start camera workers
    cam1 = CameraWorker("Cam1", RTSP_1, model, CONF_THRESHOLD, class_ids, face_detector, SKIP_FRAMES, q1)
    cam2 = CameraWorker("Cam2", RTSP_2, model, CONF_THRESHOLD, class_ids, face_detector, SKIP_FRAMES, q2)
    cam1.start()
    cam2.start()

    # ---------------------------------------------------------------
    # Display window setup
    LOGGER.info("Press ESC to quit window")
    WINDOW_NAME = "Two-Camera YOLOv8"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # allow user to resize or full-screen
    MAX_DISPLAY_WIDTH = 1920  # adjust to match your monitor width (set 0 to disable auto-scaling)


    try:
        while True:
            try:
                f1 = q1.get(timeout=1)
                f2 = q2.get(timeout=1)
            except Empty:
                continue

            combined = concat_frames([f1, f2], labels=LABELS)
            if combined is None:
                continue

            # Auto-scale if wider than monitor
            if combined.shape[1] > MAX_DISPLAY_WIDTH:
                scale = MAX_DISPLAY_WIDTH / combined.shape[1]
                new_size = (MAX_DISPLAY_WIDTH, int(combined.shape[0] * scale))
                display_frame = cv2.resize(combined, new_size)
            else:
                display_frame = combined

            cv2.imshow(WINDOW_NAME, display_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
    finally:
        cam1.stop()
        cam2.stop()
        cam1.join()
        cam2.join()
        cv2.destroyAllWindows()
        LOGGER.info("Shutdown complete")
