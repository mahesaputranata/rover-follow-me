"""
MotorBridge — sends MotorCommand to the rover.
Mode is set in settings:
  "print"  → just print (laptop dev mode)
  "serial" → USB serial to Arduino/RPi
  "socket" → UDP to RPi over WiFi

To migrate from laptop to RPi: change MOTOR_MODE in settings. Nothing else changes.
"""
import json
import socket
from rover_bridge.vision.core.data_types import MotorCommand
from rover_bridge.vision.utils.logger import RoverLogger


class MotorBridge:
    def __init__(self, settings):
        self.settings = settings
        self.logger   = RoverLogger("MotorBridge")
        self._serial  = None
        self._socket  = None

        if settings.MOTOR_MODE == "serial":
            import serial
            self._serial = serial.Serial(settings.MOTOR_SERIAL_PORT, settings.MOTOR_BAUD_RATE)
            self.logger.info(f"Serial connected: {settings.MOTOR_SERIAL_PORT}")

        elif settings.MOTOR_MODE == "socket":
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.logger.info(f"Socket ready → {settings.MOTOR_SOCKET_HOST}:{settings.MOTOR_SOCKET_PORT}")

    def send(self, cmd: MotorCommand):
        if self.settings.MOTOR_MODE == "print":
            print(f"[MOTOR] {cmd}")
            return

        payload = json.dumps({
            "direction": cmd.direction,
            "speed":     round(cmd.speed, 3),
            "turn_rate": round(cmd.turn_rate, 3),
        }).encode()

        if self.settings.MOTOR_MODE == "serial" and self._serial:
            self._serial.write(payload + b"\n")

        elif self.settings.MOTOR_MODE == "socket" and self._socket:
            self._socket.sendto(payload, (self.settings.MOTOR_SOCKET_HOST, self.settings.MOTOR_SOCKET_PORT))
