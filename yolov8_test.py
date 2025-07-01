import cv2
import torch
from ultralytics import YOLO
from face_detector import FaceDetector
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLOv8RTSPDetector:
    def __init__(self,
                 rtsp_url: str,
                 model_path: str = "yolov8s.pt",
                 device: str | None = None,
                 conf_threshold: float = 0.75,
                 detect_classes: list | None = None,
                 enable_face: bool = False,
                 display: bool = True,
                 skip_frames: int = 0):
        """YOLOv8 RTSP detector

        Args:
            rtsp_url (str): RTSP stream URL.
            model_path (str, optional): Path to YOLOv8 model. Defaults to "yolov8s.pt".
            device (str | None, optional): Device spec ("0" for GPU0, "cpu", None=auto). Defaults to None.
            conf_threshold (float, optional): Confidence threshold for detections. Defaults to 0.5.
            detect_classes (list | None, optional): List of class names or ids to detect. Defaults to ["person"].
        """
        self.rtsp_url = rtsp_url
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.cap = None
        self.enable_face = enable_face
        self.face_detector = FaceDetector() if enable_face else None
        self.display = display
        self._gui_ok = display
        self.skip_frames = max(0, skip_frames)
        self._last_annotated = None  # cache for smoother skipping
        self.model = None
        self.device = self._get_device(device)
        self._requested_classes = detect_classes or ["person"]  # default person
        self.class_ids: list[int] = []  # resolved after model load

    def _get_device(self, device):
        """Get the appropriate device for inference"""
        if device is None:
            if torch.cuda.is_available():
                device = '0'
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                logger.info("Using CPU for inference")
        return device

    # ------------------------------------------------------------------
    # Model and class utilities
    # ------------------------------------------------------------------
    def _load_model(self):
        """Load YOLOv8 model with error handling"""
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Successfully loaded model: {self.model_path}")
            # Resolve class IDs after model is loaded
            self._resolve_class_ids()
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False

    def _open_stream(self):
        """Open RTSP stream with retry logic"""
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if self.cap.isOpened():
                logger.info("RTSP stream opened successfully")
                return True
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to open RTSP stream")
            time.sleep(retry_delay)

        logger.error("Failed to open RTSP stream after multiple attempts")
        return False

    def _resolve_class_ids(self):
        """Resolve requested class names/IDs to numeric IDs based on model.names"""
        if self.model is None:
            return
        name_to_id = {v: k for k, v in self.model.names.items()}
        resolved: list[int] = []
        for c in self._requested_classes:
            if isinstance(c, int):
                resolved.append(c)
            elif isinstance(c, str):
                if c.isdigit():
                    resolved.append(int(c))
                elif c in name_to_id:
                    resolved.append(name_to_id[c])
                else:
                    logger.warning(f"Unknown class name '{c}' ignored")
        if not resolved:
            logger.warning("No valid class IDs resolved; defaulting to detect all classes")
        self.class_ids = resolved
        logger.info(f"Detecting classes: {self.class_ids} -> {[self.model.names[i] for i in self.class_ids] if self.class_ids else 'ALL'}")

    # Public utility methods -------------------------------------------------
    def add_classes(self, classes):
        """Add classes (names or IDs) to detection list and refresh mapping."""
        if isinstance(classes, (str, int)):
            classes = [classes]
        self._requested_classes.extend(classes)
        self._resolve_class_ids()

    def remove_classes(self, classes):
        """Remove classes (names or IDs) from detection list and refresh mapping."""
        if isinstance(classes, (str, int)):
            classes = [classes]
        self._requested_classes = [c for c in self._requested_classes if c not in classes]
        self._resolve_class_ids()

    def list_classes(self):
        """Return current list of requested class identifiers/names."""
        return self._requested_classes

    # ------------------------------------------------------------------
    def detect(self):
        """Main detection loop"""
        if not self._load_model():
            logger.error("Failed to initialize model. Exiting.")
            return

        if not self._open_stream():
            logger.error("Failed to initialize RTSP stream. Exiting.")
            return

        try:
            frame_idx = 0
            while True:
                ret, frame = self.cap.read()
                frame_idx += 1
                # Skip frames if configured
                if self.skip_frames and (frame_idx % (self.skip_frames + 1) != 0):
                    if self._gui_ok and self.display and self._last_annotated is not None:
                        try:
                            cv2.imshow("YOLOv8 Detection", self._last_annotated)
                            if cv2.waitKey(1) & 0xFF == 27:
                                break
                        except cv2.error:
                            self._gui_ok = False
                    continue
                if not ret:
                    logger.warning("Empty frame or stream disconnected. Attempting to reconnect...")
                    if not self._open_stream():
                        break
                    continue

                # Perform inference
                results = self.model.predict(
                    frame,
                    device=self.device,
                    imgsz=640,
                    conf=self.conf_threshold,
                    classes=self.class_ids if self.class_ids else None
                )

                # Draw detections
                annotated_frame = results[0].plot()
                self._last_annotated = annotated_frame

                # Face detection overlay
                if self.face_detector:
                    faces = self.face_detector.detect(frame)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                # Display frame if GUI available
                if self._gui_ok:
                    try:
                        cv2.imshow("YOLOv8 Detection", annotated_frame)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break
                    except cv2.error as gui_err:
                        logger.warning("cv2.imshow not supported in this environment, disabling GUI: %s", gui_err)
                        self._gui_ok = False

        except KeyboardInterrupt:
            logger.info("Detection interrupted by user")
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Resources released")

if __name__ == "__main__":
    # Example usage
    detector = YOLOv8RTSPDetector(
        # rtsp_url="rtsp://admin:admin@192.168.0.10:8554/CH001.sdp",
        rtsp_url="rtsp://admin:admin@192.168.0.11:8554/CH001.sdp",
        conf_threshold=0.75,  # Increased confidence threshold for person detection
        model_path="yolov8l.pt",
        detect_classes=["person","backpack","bottle","cup","cell phone","book"],
        enable_face=True,
        display=True,
        skip_frames=3
    )
    detector.detect()