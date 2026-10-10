"""
Detector — runs YOLOv8 once per frame, returns persons + obstacles.
Also runs a separate door detection model if configured.

Two models:
  model       → YOLOv8n (COCO) for persons + common obstacles
  door_model  → YOLOv8 fine-tuned on doors (optional, loads only if path set)

One inference each per frame. Efficient for RPi later.
"""
import os
from ultralytics import YOLO
from rover_bridge.vision.core.data_types import Detection, BoundingBox
from rover_bridge.vision.utils.logger import RoverLogger

# Synthetic class ID for doors — outside COCO range so no collision
DOOR_CLASS_ID   = 999
DOOR_CLASS_NAME = "door"


class Detector:
    def __init__(self, settings):
        self.settings   = settings
        self.logger     = RoverLogger("Detector")

        # Primary model — persons + obstacles
        self.model      = YOLO(settings.YOLO_MODEL)
        self.logger.info(f"YOLO loaded: {settings.YOLO_MODEL}")

        # Door model — optional, only loads if path is set and file exists
        self.door_model = None
        if settings.DOOR_MODEL and os.path.exists(settings.DOOR_MODEL):
            self.door_model = YOLO(settings.DOOR_MODEL)
            self.logger.info(f"Door model loaded: {settings.DOOR_MODEL}")
        else:
            self.logger.warning(
                "No door model found. "
                "Download one and set DOOR_MODEL in settings.py. "
                "Door detection disabled."
            )

    def detect(self, frame) -> list[Detection]:
        """
        Run primary YOLO on frame.
        Returns persons + obstacles from COCO model.
        """
        results    = self.model(frame, conf=self.settings.YOLO_CONFIDENCE, imgsz=256, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(Detection(
                bbox       = BoundingBox(x1, y1, x2, y2),
                class_id   = cls_id,
                class_name = self.model.names[cls_id],
                confidence = float(box.conf[0])
            ))
        return detections

    def detect_doors(self, frame) -> list[Detection]:
        """
        Run door model on frame separately.
        Returns list of door detections with synthetic class_id=999.
        Returns empty list if door model is not loaded.

        Call this in main.py every N frames — same pattern as face recognition.
        Example:
            if frame_num % 15 == 0:
                cached_doors = detector.detect_doors(frame)
            doors = cached_doors
        """
        if self.door_model is None:
            return []

        results    = self.door_model(frame, conf=self.settings.DOOR_CONFIDENCE, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(Detection(
                bbox       = BoundingBox(x1, y1, x2, y2),
                class_id   = DOOR_CLASS_ID,
                class_name = DOOR_CLASS_NAME,
                confidence = float(box.conf[0])
            ))
        return detections

    def filter_persons(self, detections: list[Detection]) -> list[Detection]:
        return [d for d in detections if d.class_id == self.settings.PERSON_CLASS_ID]

    def filter_obstacles(self, detections: list[Detection]) -> list[Detection]:
        return [d for d in detections if d.class_id in self.settings.OBSTACLE_CLASSES]

    def filter_doors(self, detections: list[Detection]) -> list[Detection]:
        return [d for d in detections if d.class_id == DOOR_CLASS_ID]
