
"""
vision_node.py  –  Raspberry Pi  (Ubuntu 22.04 / ROS2 Humble)

Runs the rover vision pipeline in a background thread.
Called by VisionManager in rover_server.py — never run standalone.

Responsibilities:
  - Capture frames from USB camera
  - Run YOLOv8n for person + obstacle detection (every frame)
  - Run face recognition at interval (every N frames)
  - Run door detection at interval (every N frames, FIND_DOOR mode only)
  - Run follow controller to compute MotorCommand
  - Run obstacle handler to gate output
  - Apply hybrid speed modifier from AI assistant
  - Publish final Twist to /cmd_vel

Hybrid mode:
  While in FOLLOW_ME, the AI assistant can still send modifier commands
  without killing vision mode:
    "faster"      → boosts speed_multiplier applied on top of vision speed
    "slower"      → reduces speed_multiplier
    "stop"        → exits vision mode entirely (handled in rover_server.py)
  Vision still controls direction and turning. The rover feels alive.
"""

import sys
import time
import logging
import threading
from std_msgs.msg import Float32MultiArray
import cv2

# ── Add rover_vision to path if needed ────────────────────────────────────────
# Adjust this path to wherever rover_vision/ lives on your RPi
ROVER_VISION_PATH = "/home/ubuntu/rover_ws/src/rover_bridge/rover_bridge/"
if ROVER_VISION_PATH not in sys.path:
    sys.path.insert(0, ROVER_VISION_PATH)
    # /home/ubuntu/rover_ws/src/rover_bridge/rover_bridge/vision/data/models/owner_face.npy

# ── rover_vision imports ───────────────────────────────────────────────────────
from rover_bridge.vision.config.settings import Settings
from rover_bridge.vision.perception.detector import Detector
from rover_bridge.vision.perception.face_recognizer import FaceRecognizer
from rover_bridge.vision.tracking.tracker import Tracker
from rover_bridge.vision.tracking.target_profile import TargetProfile
from rover_bridge.vision.decision.follow_controller import FollowController
from rover_bridge.vision.decision.obstacle_handler import ObstacleHandler
from rover_bridge.vision.core.data_types import MotorCommand

log = logging.getLogger("vision_node")

# Vision modes — must match rover_server.py constants
MODE_FOLLOW_ME = "FOLLOW_ME"
MODE_FIND_DOOR = "FIND_DOOR"

# Debug Purposes
BYPASS_FACE_RECOGNITION = True

# Hybrid speed multiplier limits
SPEED_MULTIPLIER_MIN  = 0.3
SPEED_MULTIPLIER_MAX  = 2.0
SPEED_MULTIPLIER_STEP = 0.2


# ──────────────────────────────────────────────────────────────────────────────
#  Twist converter
#  Maps MotorCommand (direction + speed + turn_rate) → (linear_x, angular_z)
# ──────────────────────────────────────────────────────────────────────────────
def motor_command_to_twist(
    cmd: MotorCommand,
    speed_multiplier: float = 1.0,
) -> tuple[float, float]:
    """
    Convert MotorCommand → (linear_x, angular_z) for bridge_node.py.

    MotorCommand fields:
        cmd.direction   str    "forward"|"backward"|"left"|"right"|"stop"
        cmd.speed       float  0.0 – 1.0
        cmd.turn_rate   float  -1.0 (full left) to +1.0 (full right)

    ROS2 Twist convention:
        linear_x  > 0 = forward,  < 0 = backward
        angular_z > 0 = left turn, < 0 = right turn

    turn_rate sign convention from rover_vision follow_controller:
        offset_x > 0 = person is to the right → turn_rate positive = turn right
        ROS2 angular_z negative = turn right → we negate turn_rate

    speed_multiplier: applied on top of cmd.speed from hybrid AI commands.
    """
    MAX_LINEAR  = 0.3   # m/s — match bridge_node MAX_SPEED
    MAX_ANGULAR = 1.5   # rad/s — tune to your wheel base

    if cmd.direction == "stop":
        return 0.0, 0.0

    # Apply hybrid multiplier — cap at 1.0 normalized before scaling to m/s
    effective_speed = min(cmd.speed * speed_multiplier, 1.0)

    # Linear velocity
    if cmd.direction == "forward":
        linear_x = effective_speed * MAX_LINEAR
    elif cmd.direction == "backward":
        linear_x = -effective_speed * MAX_LINEAR
    elif cmd.direction in ("left", "right"):
        # Turning in place or arc — small forward component
        linear_x = effective_speed * MAX_LINEAR * 0.3
    else:
        linear_x = 0.0

    # Angular velocity — negate turn_rate to match ROS2 convention
    angular_z = -cmd.turn_rate * MAX_ANGULAR

    # For explicit left/right direction, ensure correct angular sign
    if cmd.direction == "left":
        angular_z =  abs(angular_z) if angular_z == 0.0 else angular_z
        if angular_z < 0:
            angular_z = MAX_ANGULAR * 0.6
    elif cmd.direction == "right":
        angular_z = -abs(angular_z) if angular_z == 0.0 else angular_z
        if angular_z > 0:
            angular_z = -MAX_ANGULAR * 0.6

    return linear_x, angular_z


# ──────────────────────────────────────────────────────────────────────────────
#  HybridModifier
#  Thread-safe store for AI assistant modifiers applied on top of vision output.
#  rover_server.py calls apply_modifier() when manual commands arrive in vision
#  mode. VisionLoop reads it every frame via snapshot().
# ──────────────────────────────────────────────────────────────────────────────
class HybridModifier:
    """
    Holds current AI-applied modifiers layered on top of vision output.

    Speed multiplier:
        1.0  = normal vision speed (default)
        1.4  = after one "faster" command
        0.6  = after one "slower" command
        Stacks additively, clamped to [SPEED_MULTIPLIER_MIN, SPEED_MULTIPLIER_MAX].
    """

    def __init__(self):
        self._lock             = threading.Lock()
        self._speed_multiplier = 1.0

    def apply(self, command: str):
        """
        Apply a modifier command from the AI assistant.
        Called from rover_server.py WebSocket handler thread — must be
        thread-safe since VisionLoop reads this from its own thread.
        """
        with self._lock:
            if command == "faster":
                self._speed_multiplier = min(
                    self._speed_multiplier + SPEED_MULTIPLIER_STEP,
                    SPEED_MULTIPLIER_MAX,
                )
                log.info("[HYBRID] Speed multiplier → %.1f", self._speed_multiplier)

            elif command == "slower":
                self._speed_multiplier = max(
                    self._speed_multiplier - SPEED_MULTIPLIER_STEP,
                    SPEED_MULTIPLIER_MIN,
                )
                log.info("[HYBRID] Speed multiplier → %.1f", self._speed_multiplier)

            elif command == "reset speed":
                self._speed_multiplier = 1.0
                log.info("[HYBRID] Speed multiplier reset.")

    @property
    def speed_multiplier(self) -> float:
        with self._lock:
            return self._speed_multiplier

    def reset(self):
        with self._lock:
            self._speed_multiplier = 1.0


# ──────────────────────────────────────────────────────────────────────────────
#  VisionLoop
#  Instantiated by VisionManager. Call run() in a thread, stop() to exit.
# ──────────────────────────────────────────────────────────────────────────────
from debug_streamer import DebugStreamer
class VisionLoop:
    # Class-level cache — loaded once, reused across sessions
    _detector_cache     = None
    _face_recog_cache   = None
    _cache_lock         = threading.Lock()


    @classmethod
    def preload(cls, settings):
        """Call this at server startup to warm up models."""
        with cls._cache_lock:
            if cls._detector_cache is None:
                log.info("[VISION] Preloading YOLO model...")
                cls._detector_cache   = Detector(settings)
                cls._face_recog_cache = FaceRecognizer(settings)
                log.info("[VISION] Models preloaded.")

    def __init__(self, publisher, bbox_pub, mode: str):
        self.publisher  = publisher
        self.bbox_pub   = bbox_pub   
        self.mode       = mode
        self._stop_flag = threading.Event()
        self.hybrid     = HybridModifier()

        settings        = Settings()
        self.settings   = settings

        self.streamer = DebugStreamer(port=5000)
        self.streamer.start()

        # Use preloaded models if available, otherwise load now
        with VisionLoop._cache_lock:
            self.detector   = VisionLoop._detector_cache or Detector(settings)
            self.face_recog = VisionLoop._face_recog_cache or FaceRecognizer(settings)

        self.tracker        = Tracker(settings, face_recognizer=self.face_recog)
        self.follow_ctrl    = FollowController(settings)
        self.obstacle_hdlr  = ObstacleHandler(settings)
        self.target_profile = self._load_target_profile()

        # Frame caches
        self._cached_faces  = []
        self._cached_doors  = []
        self._frame_num     = 0
        # (debug: deactivate person recognition 03/10/26)
        #self._locked        = self.target_profile is not None
        if BYPASS_FACE_RECOGNITION:
            self._locked = True
        else:
            self._locked = self.target_profile is not None
        self._last_good_offset_x = 0.0
        self._last_good_offset_y = 0.0
        self._last_good_area_ratio = 1.0

        log.info("[VISION] VisionLoop ready — mode=%s locked=%s",
                 mode, self._locked)

    # ── lifecycle ──────────────────────────────────────────────────────────────
    # def run(self):
    #     """Main loop — runs until stop() is called."""
    #     cap = cv2.VideoCapture(self.settings.CAMERA_INDEX)
    #     if not cap.isOpened():
    #         log.error("[VISION] Cannot open camera index=%d",
    #                   self.settings.CAMERA_INDEX)
    #         return

    #     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.settings.FRAME_WIDTH)
    #     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.FRAME_HEIGHT)
    #     log.info("[VISION] Camera opened. Loop running in mode=%s", self.mode)

    #     try:
    #         while not self._stop_flag.is_set():
    #             ret, frame = cap.read()
    #             if not ret:
    #                 log.warning("[VISION] Camera read failed — retrying.")
    #                 time.sleep(0.05)
    #                 continue

    #             self._frame_num += 1
    #             self._process_frame(frame)

    #     except Exception:
    #         log.exception("[VISION] Unexpected error in vision loop")
    #     finally:
    #         cap.release()
    #         self.publisher.stop()
    #         self.hybrid.reset()
    #         self.streamer.stop()
    #         log.info("[VISION] Vision loop exited.")
    def run(self):
        from camera_client import get_frame_from_service

        log.info("[VISION] Loop running in mode=%s (pulling from camera_service)", self.mode)
        try:
            while not self._stop_flag.is_set():
                try:
                    frame = get_frame_from_service()
                except RuntimeError:
                    log.warning("[VISION] camera_service unreachable — retrying.")
                    time.sleep(0.2)
                    continue

                if frame is None:
                    time.sleep(0.05)
                    continue

                self._frame_num += 1
                self._process_frame(frame)
        except Exception:
            log.exception("[VISION] Unexpected error in vision loop")
        finally:
            # cap.release() removed — camera_service owns the camera, not us
            # self.publisher.stop()
            self.hybrid.reset()
            self.streamer.stop()
            log.info("[VISION] Vision loop exited.")

    def stop(self):
        """Signal the loop to exit. Called from VisionManager."""
        self._stop_flag.set()

    def apply_modifier(self, command: str):
        """
        Apply a hybrid modifier from the AI assistant.
        Called by rover_server.py when a modifier command arrives while
        vision mode is active. Thread-safe.

        Recognised: "faster", "slower", "reset speed"
        """
        self.hybrid.apply(command)

    # ── frame processing ───────────────────────────────────────────────────────
    def _process_frame(self, frame):
        fn = self._frame_num
        s  = self.settings

        detections = self.detector.detect(frame)

        if BYPASS_FACE_RECOGNITION:
            faces = []
        else:
            if fn % s.FACE_RECOGNITION_INTERVAL == 0:
                self._cached_faces = self.face_recog.detect(frame)
            faces = self._cached_faces

        if self.mode == MODE_FIND_DOOR:
            if fn % s.DOOR_DETECTION_INTERVAL == 0:
                self._cached_doors = self.detector.detect_doors(frame)
        doors = self._cached_doors

        if not self._locked and self.target_profile is not None:
            self._locked = True

        if not self._locked:
            return

        # (for debgug bypass face recognition 03/10/2026: deactivate 'faces' = '[]')
        if BYPASS_FACE_RECOGNITION:
            track_result = self.tracker.update(frame, detections, [], self.target_profile)
        else:
            track_result = self.tracker.update(
                frame, detections, faces, self.target_profile
            )

        if track_result is None or track_result.confidence < s.RECALIBRATE_THRESHOLD:
            conf = track_result.confidence if track_result else 0.0
            log.warning("[VISION] Target lost (conf=%.2f) — stopping.", conf)
            tracking_ok = False
        else:
            tracking_ok = True


        # ── NEW: Publish bbox error for LQR perception bridge
        bbox_msg = Float32MultiArray()
    
        if tracking_ok:
            lost_streak = track_result.lost_streak
        else:
            lost_streak = s.TRACKER_MAX_LOST_FRAMES

        target_valid = 1.0 if lost_streak < s.TRACKER_MAX_LOST_FRAMES else 0.0

        if lost_streak == 0:
            self._last_good_offset_x = track_result.offset_x
            self._last_good_offset_y = track_result.offset_y
            self._last_good_area_ratio = track_result.bbox_area_ratio

        if tracking_ok and lost_streak == 0:
            pub_offset_x = track_result.offset_x
            pub_offset_y = track_result.offset_y
            pub_area = track_result.bbox_area_ratio
        else:
            pub_offset_x = self._last_good_offset_x
            pub_offset_y = self._last_good_offset_y
            pub_area = self._last_good_area_ratio

        bbox_msg.data = [
            float(pub_offset_x),
            float(pub_offset_y),
            float(pub_area),
            float(target_valid),
        ]
        self.bbox_pub.publish(bbox_msg)

        if tracking_ok:
            obstacle_state = self.obstacle_hdlr.evaluate(
                detections, doors, track_result
            )
            motor_cmd = self.follow_ctrl.compute(track_result, obstacle_state)
            self.streamer.push_frame(
                frame,
                detections=detections,
                track_result=track_result,
                motor_cmd=motor_cmd,
                mode=self.mode,
                locked=self._locked,
            )

    
    # ── helpers ────────────────────────────────────────────────────────────────
    def _load_target_profile(self) -> TargetProfile | None:
        import os
        import numpy as np

        face_path = os.path.join(ROVER_VISION_PATH, "vision", "tools", "data", "owner_face.npy")
        if not os.path.exists(face_path):
            log.warning(
                "[VISION] No face profile at %s. Run enroll_face.py first.",
                face_path,
            )
            return None

        try:
            embedding = np.load(face_path)
            profile   = TargetProfile()
            profile.face_embedding = embedding
            profile.valid = True
            log.info("[VISION] Face profile loaded.")
            return profile
        except Exception as e:
            log.error("[VISION] Failed to load face profile: %s", e)
            return None
