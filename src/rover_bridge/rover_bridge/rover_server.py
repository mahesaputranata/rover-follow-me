# # ================================================================
# #  UPDATED SECTIONS FOR rover_server.py — Fix 3 Part A
# #  Only the changed sections are shown.
# #  Everything else stays identical.
# # ================================================================

# # ── Add this import at top of rover_server.py ──────────────────
# from std_msgs.msg import Float32MultiArray

# # ── Replace VisionManager.__init__() ───────────────────────────
# class VisionManager:
#     def __init__(self, publisher: TwistPublisher,
#                  bbox_pub, state: RoverState):
#         self.publisher    = publisher
#         self.bbox_pub     = bbox_pub    # NEW — passed to VisionLoop
#         self.state        = state
#         self._thread      = None
#         self._vision_loop = None
#         self._lock        = threading.Lock()

#     def start(self, mode: str) -> str:
#         with self._lock:
#             self._stop_internal()

#             if mode == MODE_IDLE:
#                 self.state.set_mode(MODE_IDLE)
#                 self.publisher.stop()
#                 return "Vision stopped. Rover idle."

#             if mode not in VALID_MODES:
#                 return f"Unknown mode: {mode}"

#             try:
#                 from vision_node import VisionLoop
#             except ImportError as e:
#                 log.error("[VISION] Import failed: %s", e)
#                 return f"Vision node unavailable: {e}"

#             try:
#                 self._vision_loop = VisionLoop(
#                     publisher=self.publisher,
#                     bbox_pub=self.bbox_pub,   # NEW
#                     mode=mode,
#                 )
#             except Exception as e:
#                 log.error("[VISION] VisionLoop init failed: %s", e)
#                 return f"VisionLoop init error: {e}"

#             self._thread = threading.Thread(
#                 target=self._run_with_logging,
#                 daemon=True,
#                 name=f"VisionLoop-{mode}",
#             )
#             self.state.set_mode(mode)
#             self._thread.start()
#             log.info("[VISION] Started mode=%s", mode)
#             return f"Vision mode {mode} started."

#     # rest of VisionManager stays identical...


# # ── Replace main() ──────────────────────────────────────────────
# def main():
#     rclpy.init()

#     state     = RoverState()
#     publisher = TwistPublisher()

#     # Create bbox_error publisher on the same node as TwistPublisher
#     # TwistPublisher is a rclpy.Node so we can add publishers to it
#     bbox_pub = publisher.create_publisher(
#         Float32MultiArray, '/bbox_error', 10
#     )

#     vision_manager = VisionManager(
#         publisher=publisher,
#         bbox_pub=bbox_pub,
#         state=state,
#     )

#     from vision_node import VisionLoop
#     from rover_bridge.vision.config.settings import Settings
#     VisionLoop.preload(Settings())

#     ros_thread = threading.Thread(
#         target=run_ros2, args=(publisher,), daemon=True, name="ros2_spin"
#     )
#     ros_thread.start()
#     log.info("ROS2 spin thread started.")

#     try:
#         asyncio.run(run_ws_server(publisher, state, vision_manager))
#     except KeyboardInterrupt:
#         log.info("Shutting down.")
#     finally:
#         vision_manager.stop()
#         publisher.stop()
#         rclpy.shutdown()
#         log.info("Rover server stopped.")

"""
rover_server.py  –  Raspberry Pi  (Ubuntu 22.04 / ROS2 Humble)

Extends the original WebSocket server with vision mode support.
When the PC assistant sends {"action": "set_mode", "mode": "FOLLOW_ME"},
VisionManager starts a background thread running vision_node.py logic
which publishes Twist directly to /cmd_vel.

Manual movement commands work exactly as before — no changes to that path.

Run order:
    Terminal 1:  ros2 run rover_bridge bridge_node
    Terminal 2:  python rover_server.py

Dependencies (Pi):
    pip install websockets ultralytics opencv-python face_recognition
    (rclpy + geometry_msgs already installed as part of your ROS2 package)
"""
from std_msgs.msg import Float32MultiArray, String

import asyncio
import json
import logging
import threading
import time
import traceback

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Twist

from run_recorder import recorder as run_recorder

# ──────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────
HOST             = "0.0.0.0"
PORT             = 8765
WHEEL_SEPARATION = 0.20   # metres
MAX_SPEED        = 0.3    # m/s
MAX_SPEED_MPS    = 0.5    # m/s at 100%
TURN_STYLE       = "arc"  # "arc" | "tank"

# Vision modes
MODE_IDLE      = "IDLE"
MODE_FOLLOW_ME = "FOLLOW_ME"
MODE_FIND_DOOR = "FIND_DOOR"
VALID_MODES    = {MODE_IDLE, MODE_FOLLOW_ME, MODE_FIND_DOOR}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WS-SERVER] %(levelname)s %(message)s",
)
log = logging.getLogger("rover_server")


def read_battery_percent() -> int | None:
    try:
        raw = open("/sys/class/power_supply/BAT0/capacity").read().strip()
        return int(raw)
    except Exception:
        return None


# ──────────────────────────────────────────────
#  Command → Twist  (unchanged from original)
# ──────────────────────────────────────────────
def command_to_twist(command: dict) -> tuple[float, float, str]:
    action    = command.get("action", "")
    direction = command.get("direction", "stop")
    speed_pct = int(command.get("speed_percent", 0))
    v = (speed_pct / 100.0) * MAX_SPEED_MPS

    if action == "stop" or direction == "stop":
        return 0.0, 0.0, ""

    if action == "move":
        if direction == "forward":
            return v, 0.0, ""
        if direction in ("backward", "back"):
            return -v, 0.0, ""
        return 0.0, 0.0, f"Unknown move direction: {direction}"

    if action == "turn":
        ang = v / (WHEEL_SEPARATION / 2.0)
        lin = v / 2.0 if TURN_STYLE == "arc" else 0.0
        if direction == "left":
            return lin, +ang, ""
        if direction == "right":
            return lin, -ang, ""
        return 0.0, 0.0, f"Unknown turn direction: {direction}"

    if action == "set_speed":
        return None, None, "set_speed_only"

    return 0.0, 0.0, f"Unknown action: {action}"

# ──────────────────────────────────────────────
#  Shared state  (extended with mode)
# ──────────────────────────────────────────────
class RoverState:
    def __init__(self):
        self._lock     = threading.Lock()
        self.direction = "stop"
        self.speed_pct = 0
        self.linear_x  = 0.0
        self.angular_z = 0.0
        self.mode      = MODE_IDLE   # NEW — current rover mode

    def update(self, direction: str, speed_pct: int, lx: float, az: float):
        with self._lock:
            self.direction = direction
            self.speed_pct = speed_pct
            self.linear_x  = lx
            self.angular_z = az

    def set_mode(self, mode: str):
        with self._lock:
            self.mode = mode

    def get_mode(self) -> str:
        with self._lock:
            return self.mode

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "direction": self.direction,
                "speed_pct": self.speed_pct,
                "linear_x":  round(self.linear_x,  4),
                "angular_z": round(self.angular_z,  4),
                "mode":      self.mode,
            }



# Raw Telementry Buffer
class RawTelemetryBuffer:
    def __init__(self, max_lines: int = 20000):
        self._lock = threading.Lock()
        self._lines: list[str] = []
        self._max_lines = max_lines

    def add(self, line: str):
        with self._lock:
            self._lines.append(line)
            if len(self._lines) > self._max_lines:
                self._lines = self._lines[-self._max_lines:]

    def drain(self) -> list[str]:
        with self._lock:
            lines, self._lines = self._lines, []
        return lines


# ──────────────────────────────────────────────
#  ROS2 publisher node  (unchanged from original)
# ──────────────────────────────────────────────
class TwistPublisher(Node):
    def __init__(self):
        super().__init__("rover_ws_bridge")
        qos = QoSProfile(depth=10)
        self._pub = self.create_publisher(Twist, "cmd_vel", qos)
        self._telemetry_buffer = RawTelemetryBuffer()
        self.create_subscription(String, '/esp32/telemetry', lambda msg: self._telemetry_buffer.add(msg.data), 10)
        self.create_subscription(String, '/esp32/warnings', lambda msg: self._telemetry_buffer.add(msg.data), 10)
        self.get_logger().info("TwistPublisher ready — publishing to /cmd_vel")

    def publish(self, linear_x: float, angular_z: float):
        msg = Twist()
        msg.linear.x  = float(linear_x)
        msg.angular.z = float(angular_z)
        self._pub.publish(msg)

    def stop(self):
        self.publish(0.0, 0.0)


# ──────────────────────────────────────────────
#  VisionManager — NEW
#  Owns the vision background thread.
#  Starts/stops based on mode commands from PC.
# ──────────────────────────────────────────────
class VisionManager:
    def __init__(self, publisher: TwistPublisher,
                 bbox_pub, state: RoverState):
        self.publisher    = publisher
        self.bbox_pub     = bbox_pub   
        self.state        = state
        self._thread      = None
        self._vision_loop = None
        self._lock        = threading.Lock()

    # helper
    def _run_with_logging(self):
        try:
            self._vision_loop.run()
        except Exception as e:
            import traceback
            log.error("[VISION] run() crashed: %s\n%s", e, traceback.format_exc())

    def start(self, mode: str) -> str:
        with self._lock:
            self._stop_internal()
            run_recorder.stop()
            if mode == MODE_IDLE:
                self.state.set_mode(MODE_IDLE)
                self.publisher.stop()
                return "Vision stopped. Rover idle."

            if mode not in VALID_MODES:
                return f"Unknown mode: {mode}"

            try:
                from vision_node import VisionLoop
            except ImportError as e:
                log.error("[VISION] Import failed: %s", e)
                return f"Vision node unavailable: {e}"

            try:
                self._vision_loop = VisionLoop(
                    publisher=self.publisher,
                    bbox_pub=self.bbox_pub,   # NEW
                    mode=mode,
                )
            except Exception as e:
                log.error("[VISION] VisionLoop init failed: %s", e)
                return f"VisionLoop init error: {e}"

            self._thread = threading.Thread(
                target=self._run_with_logging,
                daemon=True,
                name=f"VisionLoop-{mode}",
            )
            self.state.set_mode(mode)
            self._thread.start()
            log.info("[VISION] Started mode=%s", mode)
            run_recorder.start(mode)
            return f"Vision mode {mode} started."

    def stop(self) -> str:
        with self._lock:
            self._stop_internal()
            run_recorder.stop()
            self.state.set_mode(MODE_IDLE)
            self.publisher.stop()
            return "Vision stopped. Rover idle."

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def apply_modifier(self, command: str):
        """
        Pass a hybrid modifier to the running vision loop.
        Safe to call even if vision is not running — silently ignored.
        """
        with self._lock:
            if self._vision_loop is not None:
                self._vision_loop.apply_modifier(command)

    # ── internal ───────────────────────────────
    def _stop_internal(self):
        """Must be called with self._lock held."""
        if self._vision_loop is not None:
            self._vision_loop.stop()
            self._vision_loop = None
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                log.warning("[VISION] Thread did not exit cleanly in 3s")
            self._thread = None


# ──────────────────────────────────────────────
#  WebSocket handler  (extended with mode branch)
# ──────────────────────────────────────────────
async def handle_client(
    websocket,
    path,
    publisher:      TwistPublisher,
    state:          RoverState,
    vision_manager: VisionManager,
):
    addr = websocket.remote_address
    log.info("Assistant connected from %s", addr)

    try:
        async for raw in websocket:
            try:
                command = json.loads(raw)
                log.info("CMD  ← %s", command)

                action = command.get("action", "")
        
                if action == "status":
                    snapshot = state.snapshot()
                    ack = {
                        "ok": True,
                        "error": None,
                        "direction": snapshot["direction"],
                        "speed_pct": snapshot["speed_pct"],
                        "battery": read_battery_percent(),
                        "raw_telemetry": publisher._telemetry_buffer.drain(),
                        "timestamp": time.time(),
                    }
                    await websocket.send(json.dumps(ack))
                    continue

                if action == "list_runs":
                    await websocket.send(json.dumps({
                        "ok": True,
                        "runs": run_recorder.list_runs(int(command.get("limit", 20))),
                        "http_port": run_recorder.http_port,
                        "timestamp": time.time()}))
                    continue

                if action == "get_run":
                    info = await asyncio.get_running_loop().run_in_executor(
                        None, run_recorder.export_run, command.get("run_id", "last"))
                    await websocket.send(json.dumps({
                        "ok": "error" not in info, **info, "timestamp": time.time()}))
                    continue

                # ── NEW: mode command branch ───────────────────
                if action == "set_mode":
                    mode = command.get("mode", MODE_IDLE).upper()

                    # Send ack IMMEDIATELY — before vision init blocks
                    ack = {
                        "ok":        True,
                        "mode":      mode,
                        "status":    f"Starting {mode}...",
                        "battery":   read_battery_percent(),
                        "timestamp": time.time(),
            "raw_telemetry": publisher._telemetry_buffer.drain(),
                    }
                    log.info("ACK  → %s", ack)
                    await websocket.send(json.dumps(ack))

                    # Start vision in background — won't block WebSocket
                    loop = asyncio.get_running_loop()

                    def start_vision_and_report():
                        try:
                            result = vision_manager.start(mode)
                            msg = {
                                "ok": True,
                                "mode": mode,
                                "status": "Vision start completed",
                                "result": result,
                                "battery": read_battery_percent(),
                                "raw_telemetry": publisher._telemetry_buffer.drain(),
                                "timestamp": time.time(),
                            }
                        except Exception as e:
                            tb = traceback.format_exc()
                            log.error("[VISION] start failed:\n%s", tb)

                            msg = {
                                "ok": False,
                                "mode": mode,
                                "error": str(e),
                                "traceback": tb,
                                "timestamp": time.time(),
                            }
                        try:
                            future = asyncio.run_coroutine_threadsafe(
                                websocket.send(json.dumps(msg)),
                                loop,
                            )
                            future.result(timeout=5)
                        except Exception:
                            log.exception("[VISION] Failed to send vision result to client")

                    # Start vision in background — won't block WebSocket
                    threading.Thread(
                        target=start_vision_and_report,
                        daemon=True,
                        name=f"VisionStart-{mode}",
                    ).start()
                    continue                    

                # ── NEW: stop also kills vision if running ─────
                if action == "stop":
                    if vision_manager.is_running():
                        vision_manager.stop()
                    publisher.stop()
                    state.update("stop", 0, 0.0, 0.0)
                    ack = {
                        "ok":        True,
                        "error":     None,
                        "direction": "stop",
                        "speed":     0,
                        "mode":      MODE_IDLE,
                        "battery":   read_battery_percent(),
                        "timestamp": time.time(),
            "raw_telemetry": publisher._telemetry_buffer.drain(),
                    }
                    await websocket.send(json.dumps(ack))
                    continue

                # ── Hybrid modifier — faster/slower during vision mode ──
                if vision_manager.is_running():
                    # status polls are safe — never interrupt vision
                    if action in ("status", "ping"):
                        pass  # fall through to normal status handler below
                    elif action == "set_speed" or command.get("direction") in ("faster", "slower"):
                        modifier = "faster" if command.get("direction") == "faster" else "slower"
                        vision_manager.apply_modifier(modifier)
                        await websocket.send(json.dumps({
                            "ok":        True,
                            "mode":      state.get_mode(),
                            "status":    f"Vision speed modifier: {modifier}",
                            "timestamp": time.time(),
                        }))
                        continue
                    else:
                        # Any other manual command → stop vision, take direct control
                        log.info("[VISION] Manual override — stopping vision mode.")
                        vision_manager.stop()

                linear_x, angular_z, err = command_to_twist(command)

                if err == "set_speed_only":
                    await websocket.send(json.dumps({
                        "ok":        True,
                        "sent":      "speed_update_only",
                        "direction": state.snapshot()["direction"],
                        "speed_pct": command.get("speed_percent", 0),
                        "timestamp": time.time(),
                    }))
                    continue

                if err:
                    await websocket.send(json.dumps({"ok": False, "error": err}))
                    continue

                publisher.publish(linear_x, angular_z)
                state.update(
                    direction = command.get("direction", "stop"),
                    speed_pct = int(command.get("speed_percent", 0)),
                    lx        = linear_x,
                    az        = angular_z,
                )

                snap = state.snapshot()
                ack  = {
                    "ok":        True,
                    "error":     None,
                    "direction": snap["direction"],
                    "speed_pct": snap["speed_pct"],
                    "speed":     snap["speed_pct"],
                    "battery":   read_battery_percent(),
                    "linear_x":  snap["linear_x"],
                    "angular_z": snap["angular_z"],
                    "timestamp": time.time(),
            "raw_telemetry": publisher._telemetry_buffer.drain(),
                }
                log.info("ACK  → %s", ack)
                await websocket.send(json.dumps(ack))

            except json.JSONDecodeError as e:
                await websocket.send(json.dumps({"ok": False, "error": f"Bad JSON: {e}"}))

            except Exception as e:
                tb = traceback.format_exc()
                log.error("Handler error:\n%s", tb)
                try:
                    await websocket.send(json.dumps({
                        "ok": False, "error": str(e), "traceback": tb
                    }))
                except Exception:
                    pass

    except Exception:
        pass

    finally:
        log.warning("Assistant %s disconnected — stopping everything.", addr)
        vision_manager.stop()
        publisher.stop()
        state.update("stop", 0, 0.0, 0.0)


# ──────────────────────────────────────────────
#  Thread: ROS2 spin  (unchanged)
# ──────────────────────────────────────────────
def run_ros2(node: TwistPublisher):
    try:
        rclpy.spin(node)
    except Exception:
        log.error("rclpy.spin crashed:\n%s", traceback.format_exc())
    finally:
        node.destroy_node()


# ──────────────────────────────────────────────
#  Async: WebSocket server  (extended with vision_manager)
# ──────────────────────────────────────────────
async def run_ws_server(
    publisher:      TwistPublisher,
    state:          RoverState,
    vision_manager: VisionManager,
):
    import websockets
    async with websockets.serve(
        lambda ws: handle_client(ws, None, publisher, state, vision_manager),
        HOST,
        PORT,
    ):
        log.info("WebSocket server ready on ws://%s:%d", HOST, PORT)
        await asyncio.Future()


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────
def main():
    rclpy.init()

    state     = RoverState()
    publisher = TwistPublisher()

    # Create bbox_error publisher on the same node as TwistPublisher
    # TwistPublisher is a rclpy.Node so we can add publishers to it
    bbox_pub = publisher.create_publisher(
        Float32MultiArray, '/bbox_error', 10
    )

    vision_manager = VisionManager(
        publisher=publisher,
        bbox_pub=bbox_pub,
        state=state,
    )

    from vision_node import VisionLoop
    from rover_bridge.vision.config.settings import Settings
    VisionLoop.preload(Settings())

    ros_thread = threading.Thread(
        target=run_ros2, args=(publisher,), daemon=True, name="ros2_spin"
    )
    ros_thread.start()
    log.info("ROS2 spin thread started.")

    try:
        asyncio.run(run_ws_server(publisher, state, vision_manager))
    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        vision_manager.stop()
        publisher.stop()
        rclpy.shutdown()
        log.info("Rover server stopped.")


if __name__ == "__main__":
    main()
