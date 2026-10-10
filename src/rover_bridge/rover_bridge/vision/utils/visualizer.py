"""
Visualizer — draws debug overlay on the frame.
Shows: detections, track bbox, lock mode, confidence, motor command.
Only used during development. Safe to disable on RPi for performance.
"""
import cv2
from rover_bridge.vision.core.data_types import Detection, FaceDetection, TrackResult, MotorCommand

COLORS = {
    "person":   (0, 200, 255),
    "obstacle": (0, 50, 255),
    "face":     (50, 255, 50),
    "target":   (255, 200, 0),
    "door":     (255, 100, 200)
}
MODE_COLORS = {
    "face": (50, 255, 50), "body": (0, 200, 255),
    "lower_body": (0, 165, 255), "feet": (0, 50, 255), "lost": (80, 80, 80)
}


class Visualizer:
    def __init__(self, settings):
        self.settings = settings

    def render(self, frame, detections, faces, doors,
               track: TrackResult | None,
               cmd: MotorCommand | None,
               locked=False):
        h, w = frame.shape[:2]

        for door in doors:
            dc = COLORS["door"]
            cv2.rectangle(frame, (door.bbox.x1, door.bbox.y1),
                        (door.bbox.x2, door.bbox.y2), dc, 2)
            cv2.putText(frame, f"DOOR {door.confidence:.2f}",
                        (door.bbox.x1, door.bbox.y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, dc, 2)
    
        # Draw all detections
        for d in detections:
            color = COLORS["person"] if d.class_id == 0 else COLORS["obstacle"]
            cv2.rectangle(frame, (d.bbox.x1, d.bbox.y1), (d.bbox.x2, d.bbox.y2), color, 1)
            cv2.putText(frame, f"{d.class_name} {d.confidence:.2f}",
                        (d.bbox.x1, d.bbox.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw faces
        for f in faces:
            cv2.rectangle(frame, (f.bbox.x1, f.bbox.y1), (f.bbox.x2, f.bbox.y2), COLORS["face"], 1)

        # Draw target
        if track and track.target_bbox:
            bbox  = track.target_bbox
            color = MODE_COLORS.get(track.lock_mode, (255, 255, 255))
            cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)
            label = f"{track.lock_mode.upper()} | conf:{track.confidence:.2f}"
            cv2.putText(frame, label, (bbox.x1, bbox.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # HUD
        status = "LOCKED" if locked else "SEARCHING — press L to lock"
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (50, 255, 50) if locked else (80, 80, 80), 2)

        if cmd:
            cv2.putText(frame, str(cmd), (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Center crosshair
        cv2.line(frame, (w//2, h//2 - 15), (w//2, h//2 + 15), (100, 100, 100), 1)
        cv2.line(frame, (w//2 - 15, h//2), (w//2 + 15, h//2), (100, 100, 100), 1)

        cv2.imshow("Rover Vision", frame)
