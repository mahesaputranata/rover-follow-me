"""
ObstacleHandler — evaluates detections and decides obstacle state.
Handles both regular obstacles (COCO) and doors (separate model).

Door behaviour in Phase 1:
  - Door detected + blocking path → STOP (rover pauses, notifies AI assistant)
  - Door detected + not in path   → log only, no action

Phase 2 will add: open/closed check via GPT, segmentation for center passage.
"""
from rover_bridge.vision.core.data_types import Detection, TrackResult, ObstacleState
from rover_bridge.vision.perception.detector import DOOR_CLASS_ID
from rover_bridge.vision.utils.logger import RoverLogger


class ObstacleHandler:
    def __init__(self, settings):
        self.settings = settings
        self.logger   = RoverLogger("ObstacleHandler")

    def evaluate(
        self,
        detections: list[Detection],
        door_detections: list[Detection],
        track: TrackResult
    ) -> ObstacleState:
        s = self.settings

        # ── Check doors first — they take priority ─────────────
        door_state = self._evaluate_doors(door_detections, s)
        if door_state.blocked:
            return door_state

        # ── Check regular obstacles ────────────────────────────
        obstacles = [d for d in detections if d.class_id in s.OBSTACLE_CLASSES]
        if not obstacles:
            return ObstacleState(blocked=False, zone="clear")

        for obs in obstacles:
            area_ratio = obs.bbox.area / (640 * 480)
            if area_ratio < 0.03:
                continue

            cx_norm = obs.bbox.center_x / 640

            if 0.33 < cx_norm < 0.66:
                zone = "center"
            elif cx_norm <= 0.33:
                zone = "left"
            else:
                zone = "right"

            if area_ratio > s.OBSTACLE_STOP_DISTANCE or zone == "center":
                self.logger.info(f"Obstacle — zone: {zone}, area: {area_ratio:.2f}")
                return ObstacleState(blocked=True, zone=zone, closest_area_ratio=area_ratio)

        return ObstacleState(blocked=False, zone="clear")

    def _evaluate_doors(self, door_detections: list[Detection], s) -> ObstacleState:
        """
        Phase 1: treat a door in the path as a blocker — rover stops.
        Phase 2: this method will also check open/closed state via GPT.
        """
        if not door_detections:
            return ObstacleState(blocked=False, zone="clear")

        for door in door_detections:
            area_ratio = door.bbox.area / (640 * 480)

            if area_ratio < s.DOOR_MIN_AREA_RATIO:
                continue                              # door too far away, ignore

            cx_norm = door.bbox.center_x / 640

            if 0.33 < cx_norm < 0.66:
                zone = "center"
            elif cx_norm <= 0.33:
                zone = "left"
            else:
                zone = "right"

            if area_ratio > s.DOOR_BLOCK_THRESHOLD or zone == "center":
                self.logger.info(
                    f"Door detected — zone: {zone}, "
                    f"area: {area_ratio:.2f}, "
                    f"conf: {door.confidence:.2f}"
                )
                # Phase 1: just block. Phase 2: add open/closed logic here.
                return ObstacleState(
                    blocked             = True,
                    zone                = zone,
                    closest_area_ratio  = area_ratio,
                    is_door             = True          # flag for AI assistant + Phase 2
                )

        return ObstacleState(blocked=False, zone="clear")
