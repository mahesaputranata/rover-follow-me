"""
TargetProfile — everything we know about the person we are following.
Built once at lock-on. Persists across all tracking frames.
"""
import numpy as np
import cv2
from rover_bridge.vision.core.data_types import BoundingBox, Detection, FaceDetection
from rover_bridge.vision.utils.logger import RoverLogger

logger = RoverLogger("TargetProfile")


class TargetProfile:
    def __init__(self):
        self.face_embedding:        np.ndarray | None = None
        self.upper_body_histogram:  np.ndarray | None = None
        self.lower_body_histogram:  np.ndarray | None = None
        self.shoe_histogram:        np.ndarray | None = None
        self.reference_bbox_area:   int = 0
        self.valid:                 bool = False

    @classmethod
    def calibrate(cls, frame, detections: list[Detection], faces: list[FaceDetection]) -> "TargetProfile | None":
        """
        Called once when user presses 'L'.
        Expects exactly one person clearly visible in the center of frame.
        """
        profile = cls()
        h, w    = frame.shape[:2]
        cx, cy  = w // 2, h // 2

        # Find the person closest to center
        persons = [d for d in detections if d.class_id == 0]
        if not persons:
            logger.warning("No person detected during calibration.")
            return None

        target = min(persons, key=lambda d: abs(d.bbox.center_x - cx) + abs(d.bbox.center_y - cy))
        profile.reference_bbox_area = target.bbox.area

        # Save color histograms
        profile.upper_body_histogram = _color_histogram(frame, target.bbox, zone="upper")
        profile.lower_body_histogram = _color_histogram(frame, target.bbox, zone="lower")
        profile.shoe_histogram       = _color_histogram(frame, target.bbox, zone="feet")

        # Save face embedding if a face is visible
        if faces:
            closest_face = min(faces, key=lambda f: abs(f.bbox.center_x - cx))
            profile.face_embedding = closest_face.embedding
            logger.info("Face embedding saved.")
        else:
            logger.warning("No face found during calibration — color-only tracking.")

        profile.valid = True
        logger.info(f"Profile calibrated. Reference area: {profile.reference_bbox_area}px²")
        return profile


def _color_histogram(frame, bbox: BoundingBox, zone: str) -> np.ndarray:
    """Extracts HSV color histogram from a vertical zone of the bounding box."""
    x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
    h = y2 - y1
    zones = {
        "upper": (y1,          y1 + h // 3),
        "lower": (y1 + h // 3, y1 + 2 * h // 3),
        "feet":  (y1 + 2 * h // 3, y2),
    }
    zy1, zy2 = zones[zone]
    crop = frame[zy1:zy2, x1:x2]
    if crop.size == 0:
        return np.zeros(512)
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()
