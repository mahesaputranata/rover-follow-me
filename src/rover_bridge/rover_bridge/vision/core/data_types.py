"""
Shared data types used across all modules.
Define once here — import everywhere.
Adding a field here automatically makes it available to all phases.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center_x(self): return (self.x1 + self.x2) // 2
    @property
    def center_y(self): return (self.y1 + self.y2) // 2
    @property
    def area(self): return (self.x2 - self.x1) * (self.y2 - self.y1)
    @property
    def width(self): return self.x2 - self.x1
    @property
    def height(self): return self.y2 - self.y1


@dataclass
class Detection:
    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float


@dataclass
class FaceDetection:
    bbox: BoundingBox
    embedding: Optional[np.ndarray] = None   # 128-d face embedding
    confidence: float = 0.0


@dataclass
class TrackResult:
    """
    Output of the tracker every frame.
    confidence drives all downstream decisions.
    """
    target_bbox: Optional[BoundingBox]
    confidence: float                         # 0.0 – 1.0
    lock_mode: str                            # "face" | "body" | "lower_body" | "feet" | "lost"
    offset_x: float                           # normalized: -1 (left) to +1 (right)
    bbox_area_ratio: float                    # current area / reference area
    offset_y: float = 0.0
    frame_number: int = 0
    lost_streak: int = 0

@dataclass
class ObstacleState:
    blocked: bool = False
    zone: str = "clear"                       # "clear" | "left" | "center" | "right"
    closest_area_ratio: float = 0.0
    is_door: bool = False                     # True = blocker is a door (Phase 2 uses this)


@dataclass
class MotorCommand:
    """
    Normalized motor command — hardware-agnostic.
    MotorBridge translates this to actual signals.
    """
    direction: str                            # "forward"|"backward"|"left"|"right"|"stop"
    speed: float                              # 0.0 – 1.0
    turn_rate: float = 0.0                    # -1.0 (full left) to +1.0 (full right)
    source: str = ""                          # which module issued this command

    def __str__(self):
        return f"[{self.direction}] speed={self.speed:.2f} turn={self.turn_rate:.2f} ({self.source})"
