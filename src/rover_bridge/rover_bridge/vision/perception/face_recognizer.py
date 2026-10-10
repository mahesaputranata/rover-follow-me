"""
FaceRecognizer — detects faces and computes embeddings.
Uses face_recognition library (dlib under the hood).
Swap this module later for a faster model (e.g. InsightFace) without touching anything else.
"""
import face_recognition
import numpy as np
from rover_bridge.vision.core.data_types import FaceDetection, BoundingBox
from rover_bridge.vision.utils.logger import RoverLogger
import os


class FaceRecognizer:
    def __init__(self, settings):
        self.settings = settings
        self.logger   = RoverLogger("FaceRecognizer")
        self.owner_embeddings = self._load_owner_face()
    
    def _load_owner_face(self):
        path = "/home/ubuntu/rover_ws/src/rover_bridge/rover_bridge/vision/tools/data/owner_face.npy"
        if os.path.exists(path):
            self.logger.info("Owner face loaded from disk")
            return np.load(path)
        self.logger.warning("no owner face")
        return None

    def detect(self, frame) -> list[FaceDetection]:
        """Returns all faces found in frame with embeddings."""
        # rgb = frame[:, :, ::-1]
        rgb = np.ascontiguousarray(frame[:, :, ::-1]) 
        locations  = face_recognition.face_locations(rgb, model="hog")
        encodings  = face_recognition.face_encodings(rgb, locations)
        results = []
        for (top, right, bottom, left), enc in zip(locations, encodings):
            results.append(FaceDetection(
                bbox      = BoundingBox(left, top, right, bottom),
                embedding = np.array(enc),
                confidence = 1.0
            ))
        return results

    def match(self, embedding: np.ndarray, reference: np.ndarray, threshold=0.55) -> float:
        """
        Returns similarity score 0.0–1.0.
        Lower distance = better match. We invert to a score.
        """
        dist = face_recognition.face_distance([reference], embedding)[0]
        if dist > threshold:
            return 0.0
        return float(1.0 - (dist / threshold))
