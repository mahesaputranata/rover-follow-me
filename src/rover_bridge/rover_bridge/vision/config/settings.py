"""
Central configuration — all tunable values live here.
Change here, affects the whole system.
"""
import os


class Settings:
    # ── Camera ────────────────────────────────────────────────
    CAMERA_INDEX        = 0
    FRAME_WIDTH         = 640
    FRAME_HEIGHT        = 480
    FPS_TARGET          = 30

    # ── Detection (primary YOLO — COCO) ───────────────────────
    YOLO_MODEL          = "yolov8n.pt"
    YOLO_CONFIDENCE     = 0.5
    PERSON_CLASS_ID     = 0                   # COCO class 0 = person
    OBSTACLE_CLASSES    = [56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
                          # chair, couch, potted plant, bed, dining table, etc.

    # ── Door Detection (separate model) ───────────────────────
    # Step 1: run  python tools/download_door_model.py  to get the weights
    # Step 2: that script saves to data/models/door_yolov8n.pt automatically
    # Step 3: set DOOR_MODEL = "" to fully disable door detection
    DOOR_MODEL              = "data/models/door_yolov8n.pt"
    DOOR_CONFIDENCE         = 0.5           # slightly lower — doors are harder
    DOOR_DETECTION_INTERVAL = 15              # run every N frames (saves CPU)
    DOOR_MIN_AREA_RATIO     = 0.04            # ignore tiny/distant doors
    DOOR_BLOCK_THRESHOLD    = 0.20            # area ratio where door = blocking path

    # ── Tracking / Confidence ─────────────────────────────────
    RECALIBRATE_THRESHOLD   = 0.20
    BBOX_AREA_TOLERANCE     = 0.30
    TRACKER_MAX_LOST_FRAMES = 30
    FACE_RECOGNITION_INTERVAL = 1000000 # debug change from 10 -> 1000000

    # ── Confidence Score Weights ──────────────────────────────
    WEIGHT_FACE         = 1.0
    WEIGHT_BODY         = 0.7
    WEIGHT_LOWER_BODY   = 0.4
    WEIGHT_FEET         = 0.2
    WEIGHT_CONTINUITY   = 0.3
    WEIGHT_COLOR        = 0.2

    # ── Follow Controller ─────────────────────────────────────
    FRAME_CENTER_TOLERANCE  = 0.15
    SPEED_NORMAL            = 0.5
    SPEED_SLOW              = 0.25
    OBSTACLE_STOP_DISTANCE  = 0.35
    FIXED_REFERENCE_BBOX_AREA = 32000

    # ── Motor Bridge ──────────────────────────────────────────
    MOTOR_MODE          = "print"            # "print" | "serial" | "socket"
    MOTOR_SERIAL_PORT   = "/dev/ttyUSB0"
    MOTOR_BAUD_RATE     = 115200
    MOTOR_SOCKET_HOST   = "192.168.1.100"
    MOTOR_SOCKET_PORT   = 5005

    # ── AI Assistant Bridge ───────────────────────────────────
    AI_BRIDGE_MODE      = "print"           # "print" | "socket" | "http"
    AI_BRIDGE_HOST      = "192.168.1.50"    # your Windows PC IP
    AI_BRIDGE_PORT      = 6006

    # ── Memory (Phase 2) ──────────────────────────────────────
    MEMORY_FILE         = "data/environment_memory.json"
    SCAN_ANGLES         = [0, 90, 180, 270, 45, 135]
    GPT_MODEL           = "gpt-4o"
    GPT_API_KEY         = os.environ.get("OPENAI_API_KEY", "")

    # ── Logging ───────────────────────────────────────────────
    LOG_LEVEL           = "INFO"
    LOG_TO_FILE         = True
    LOG_FILE            = "logs/rover.log"
