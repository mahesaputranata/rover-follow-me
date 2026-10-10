"""
FollowController — translates TrackResult into a MotorCommand.
Pure logic, no hardware knowledge. Easy to unit test.
"""
from rover_bridge.vision.core.data_types import TrackResult, ObstacleState, MotorCommand
from rover_bridge.vision.utils.logger import RoverLogger


class FollowController:
    def __init__(self, settings):
        self.settings = settings
        self.logger   = RoverLogger("FollowController")

    def compute(self, track: TrackResult, obstacle: ObstacleState) -> MotorCommand:
        s = self.settings

        # ── Obstacle takes priority ───────────────────────────
        if obstacle.blocked:
            if obstacle.zone == "center":
                self.logger.info("Obstacle center — STOP")
                return MotorCommand("stop", 0.0, source="obstacle_center")
            elif obstacle.zone == "left":
                return MotorCommand("right", s.SPEED_SLOW, turn_rate=0.6, source="obstacle_left")
            elif obstacle.zone == "right":
                return MotorCommand("left",  s.SPEED_SLOW, turn_rate=-0.6, source="obstacle_right")

        # ── Lost target ───────────────────────────────────────
        if track.lock_mode == "lost" or track.target_bbox is None:
            return MotorCommand("stop", 0.0, source="lost_target")

        # ── Distance control via bbox area ratio ──────────────
        tol = s.BBOX_AREA_TOLERANCE
        if track.bbox_area_ratio > (1.0 + tol):
            # Too close — back up
            return MotorCommand("backward", s.SPEED_SLOW, source="too_close")

        # ── Lateral correction ────────────────────────────────
        turn_rate = 0.0
        if abs(track.offset_x) > s.FRAME_CENTER_TOLERANCE:
            turn_rate = track.offset_x * 0.8          # proportional turn

        # ── Too far or in range — move forward ────────────────
        speed = s.SPEED_NORMAL if track.bbox_area_ratio < (1.0 - tol) else s.SPEED_SLOW

        direction = "forward"
        if abs(turn_rate) > 0.5 and track.bbox_area_ratio > 0.8:
            direction = "right" if turn_rate > 0 else "left"

        return MotorCommand(direction, speed, turn_rate=turn_rate, source="follow")
