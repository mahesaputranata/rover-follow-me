"""
debug_streamer.py  —  Raspberry Pi side
Drop this file next to vision_node.py in:
    ~/rover_ws/src/rover_bridge/rover_bridge/

A lightweight MJPEG HTTP server that runs in a background thread.
VisionLoop calls streamer.push_frame(frame, detections, track_result)
after every processed frame.

Open in browser on Windows:
    http://10.46.88.217:5000

No extra dependencies beyond what you already have:
    - opencv-python  (already installed)
    - threading, http.server (stdlib)

CPU cost: minimal — JPEG encode is fast, only streams when a client is connected.
"""

import io
import threading
import time
import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────
STREAM_PORT    = 5000
JPEG_QUALITY   = 60     # lower = faster encode, smaller bandwidth
MAX_FPS        = 10     # cap stream FPS to reduce CPU load
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480

# ─────────────────────────────────────────────────────────────────────────────
#  Colors (BGR)
# ─────────────────────────────────────────────────────────────────────────────
COLOR_PERSON   = (0,   255,  0)    # green  — tracked person
COLOR_OBSTACLE = (0,   100, 255)   # orange — obstacle
COLOR_DOOR     = (255, 200,   0)   # cyan   — door
COLOR_UNKNOWN  = (180, 180, 180)   # gray   — other detection
COLOR_TARGET   = (0,    80, 255)   # red    — locked target box
COLOR_TEXT_BG  = (30,   30,  30)   # dark   — text background


# ─────────────────────────────────────────────────────────────────────────────
#  FrameBuffer — thread-safe latest-frame store
# ─────────────────────────────────────────────────────────────────────────────
class FrameBuffer:
    def __init__(self):
        self._lock      = threading.Lock()
        self._jpeg      = None          # latest encoded JPEG bytes
        self._updated   = threading.Event()
        self._last_push = 0.0
        self._min_interval = 1.0 / MAX_FPS

    def push(self, jpeg_bytes: bytes):
        with self._lock:
            self._jpeg = jpeg_bytes
        self._updated.set()

    def get(self, timeout: float = 2.0) -> Optional[bytes]:
        self._updated.wait(timeout=timeout)
        self._updated.clear()
        with self._lock:
            return self._jpeg

    def should_push(self) -> bool:
        now = time.time()
        if now - self._last_push >= self._min_interval:
            self._last_push = now
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  HTTP handler — serves MJPEG stream + simple HTML page
# ─────────────────────────────────────────────────────────────────────────────
class _MJPEGHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # suppress access logs

    def do_GET(self):
        if self.path == "/stream":
            self._serve_stream()
        elif self.path == "/" or self.path == "/index.html":
            self._serve_page()
        else:
            self.send_error(404)

    def _serve_page(self):
        html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Buddy — Vision Debug</title>
  <style>
    body {{ background: #111; color: #eee; font-family: monospace;
            display: flex; flex-direction: column; align-items: center;
            padding: 20px; margin: 0; }}
    h1   {{ color: #4af; margin-bottom: 4px; }}
    p    {{ color: #888; margin: 0 0 16px 0; font-size: 13px; }}
    img  {{ border: 2px solid #333; border-radius: 4px;
            max-width: 100%; }}
    .badge {{ background: #1a3; color: #fff; padding: 2px 8px;
              border-radius: 3px; font-size: 12px; margin-left: 8px; }}
  </style>
</head>
<body>
  <h1>Buddy Vision Debug <span class="badge">LIVE</span></h1>
  <p>MJPEG stream from Raspberry Pi — {FRAME_WIDTH}x{FRAME_HEIGHT} @ {MAX_FPS}fps max</p>
  <img src="/stream" alt="live feed" />
</body>
</html>"""
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_stream(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=--frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        buf: FrameBuffer = self.server.frame_buffer

        while True:
            jpeg = buf.get(timeout=3.0)
            if jpeg is None:
                continue
            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break


# ─────────────────────────────────────────────────────────────────────────────
#  DebugStreamer — main class, used by VisionLoop
# ─────────────────────────────────────────────────────────────────────────────
class DebugStreamer:
    """
    Usage in VisionLoop._process_frame():

        # At top of _process_frame, after reading frame:
        if self.streamer.has_client:
            self.streamer.push_frame(frame, detections, track_result,
                                     motor_cmd=motor_cmd,
                                     mode=self.mode,
                                     locked=self._locked)
    """

    def __init__(self, port: int = STREAM_PORT):
        self.port         = port
        self._buf         = FrameBuffer()
        self._server      = None
        self._thread      = None
        self._running     = False
        self._client_lock = threading.Lock()
        self._has_client  = False

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self):
        self._running = True
        self._server  = HTTPServer(("0.0.0.0", self.port), _MJPEGHandler)
        self._server.frame_buffer = self._buf
        self._thread  = threading.Thread(
            target=self._serve_forever,
            daemon=True,
            name="MJPEGServer",
        )
        self._thread.start()
        print(f"[STREAM] Debug stream ready → http://0.0.0.0:{self.port}")

    def stop(self):
        self._running = False
        if self._server:
            self._server.shutdown()

    def _serve_forever(self):
        self._server.serve_forever()

    # ── frame push ────────────────────────────────────────────────────────────
    def push_frame(
        self,
        frame,
        detections=None,
        track_result=None,
        motor_cmd=None,
        mode: str = "",
        locked: bool = False,
    ):
        """
        Call this from VisionLoop._process_frame() with the raw frame.
        Annotates and encodes to JPEG only if the rate limiter allows.
        Zero cost when no client is connected.
        """
        if not self._buf.should_push():
            return

        try:
            annotated = self._annotate(
                frame, detections, track_result, motor_cmd, mode, locked
            )
            _, jpeg = cv2.imencode(
                ".jpg", annotated,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            self._buf.push(jpeg.tobytes())
        except Exception as e:
            print(f"[STREAM] push_frame error: {e}")

    # ── annotation ────────────────────────────────────────────────────────────
    def _annotate(self, frame, detections, track_result, motor_cmd, mode, locked):
        out = frame.copy()
        h, w = out.shape[:2]

        # ── Detection boxes ───────────────────────────────────────────────────
        if detections:
            for det in detections:
                try:
                    x1, y1, x2, y2 = map(int, det.bbox)
                    label = getattr(det, "label", "?")
                    conf  = getattr(det, "confidence", 0.0)

                    if label == "person":
                        color = COLOR_PERSON
                    elif label == "door":
                        color = COLOR_DOOR
                    else:
                        color = COLOR_OBSTACLE

                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                    self._label(out, f"{label} {conf:.2f}", x1, y1, color)
                except Exception:
                    pass

        # ── Tracked target box ────────────────────────────────────────────────
        if track_result and getattr(track_result, "bbox", None) is not None:
            try:
                x1, y1, x2, y2 = map(int, track_result.bbox)
                conf = getattr(track_result, "confidence", 0.0)
                cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_TARGET, 3)
                self._label(out, f"TARGET {conf:.2f}", x1, y1, COLOR_TARGET)

                # Draw center cross
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.drawMarker(out, (cx, cy), COLOR_TARGET,
                               cv2.MARKER_CROSS, 20, 2)

                # Draw center-to-frame-center line
                fc_x, fc_y = w // 2, h // 2
                cv2.line(out, (fc_x, fc_y), (cx, cy), (200, 200, 0), 1)
                offset_x = cx - fc_x
                self._label(out, f"offset={offset_x:+d}px", fc_x - 40, fc_y + 20,
                            (200, 200, 0))
            except Exception:
                pass

        # ── Frame center line ─────────────────────────────────────────────────
        cv2.line(out, (w // 2, 0), (w // 2, h), (60, 60, 60), 1)

        # ── HUD overlay ───────────────────────────────────────────────────────
        hud_lines = [
            f"MODE: {mode or 'IDLE'}",
            f"LOCKED: {'YES' if locked else 'NO'}",
        ]

        if motor_cmd:
            direction = getattr(motor_cmd, "direction", "?")
            speed     = getattr(motor_cmd, "speed", 0.0)
            turn_rate = getattr(motor_cmd, "turn_rate", 0.0)
            hud_lines += [
                f"DIR: {direction}",
                f"SPEED: {speed:.2f}",
                f"TURN: {turn_rate:+.2f}",
            ]
        else:
            hud_lines.append("MOTOR: stopped")

        # draw HUD box
        line_h   = 20
        pad      = 6
        box_h    = len(hud_lines) * line_h + pad * 2
        box_w    = 200
        cv2.rectangle(out, (pad, pad), (pad + box_w, pad + box_h),
                      COLOR_TEXT_BG, -1)
        cv2.rectangle(out, (pad, pad), (pad + box_w, pad + box_h),
                      (80, 80, 80), 1)

        for i, line in enumerate(hud_lines):
            color = (80, 255, 80) if "YES" in line else (200, 200, 200)
            if "MODE" in line:
                color = (100, 200, 255)
            cv2.putText(out, line,
                        (pad * 2, pad + (i + 1) * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        cv2.LINE_AA)

        # ── Timestamp ─────────────────────────────────────────────────────────
        ts = time.strftime("%H:%M:%S")
        cv2.putText(out, ts, (w - 70, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        return out

    def _label(self, img, text, x, y, color):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y_pos = max(y - 4, th + 4)
        cv2.rectangle(img, (x, y_pos - th - 4), (x + tw + 4, y_pos + 2),
                      COLOR_TEXT_BG, -1)
        cv2.putText(img, text, (x + 2, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)