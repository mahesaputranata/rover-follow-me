"""
bridge_node.py  —  Raspberry Pi  (Ubuntu 22.04 / ROS2 Humble)

Bridges ROS2 /cmd_vel  →  ESP32 via UART.

Key change from v1:
  v1 sent:  MOVE {l_pwm} {r_pwm}     (raw PWM, open-loop)
  v2 sends: SET_VEL {v_left} {v_right} (m/s,  ESP32 PID handles the rest)

This activates the ESP32 velocity PID and encoder feedback,
giving the rover closed-loop wheel control without any other changes.

Physical constants to set:
  WHEEL_SEPARATION  —  measure wheel center to wheel center (metres)
  MAX_SPEED_MPS     —  max wheel velocity you want to allow (m/s)
                       should match or be less than ESP32 physical limit
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import serial
import time
import threading
from std_msgs.msg import String

# ----------------------------------------------------------------
#  Hardware constants — update these to match your rover
# ----------------------------------------------------------------
SERIAL_PORT      = '/dev/ttyUSB0'
BAUD_RATE        = 115200
WHEEL_SEPARATION = 0.20    # metres — measure wheel center to center
MAX_SPEED_MPS    = 0.3     # m/s   — max wheel speed sent to ESP32


class RoverBridgeNode(Node):
    def __init__(self):
        super().__init__('rover_bridge')

        # Publishers
        self.odom_pub = self.create_publisher(String, '/esp32/odom', 10)
        self.telemetry_pub = self.create_publisher(String, '/esp32/telemetry', 10)
        self.warnings_pub = self.create_publisher(String, 'esp32/warnings', 10)

        # ── Connect to ESP32 ──────────────────────────────────
        self.get_logger().info(f'Connecting to ESP32 on {SERIAL_PORT}...')
        self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)

        # Ping to confirm connection
        self.ser.write(b'PING\n')
        time.sleep(0.1)
        response = self.ser.readline().decode(errors='ignore').strip()
        self.get_logger().info(f'ESP32 response: {response}')

        if response != 'PONG':
            self.get_logger().warn(
                'Did not get PONG — check serial port and ESP32 firmware'
            )

        # ── Subscribe to /cmd_vel ─────────────────────────────
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # ── Reader thread for ESP32 telemetry ────────────────
        # Reads ODOM and STATUS messages streamed back from ESP32
        self.reader_thread = threading.Thread(
            target=self.serial_reader, daemon=True
        )
        self.reader_thread.start()

        self.get_logger().info(
            'Rover bridge ready (SET_VEL mode) — waiting for /cmd_vel'
        )

    # ----------------------------------------------------------------
    #  cmd_vel_callback
    #  Converts ROS2 Twist → differential drive wheel velocities (m/s)
    #  and sends SET_VEL to ESP32.
    #
    #  Differential drive kinematics:
    #    v_left  = linear_x - (angular_z * WHEEL_SEPARATION / 2)
    #    v_right = linear_x + (angular_z * WHEEL_SEPARATION / 2)
    #
    #  No PWM conversion — ESP32 PID handles velocity tracking.
    # ----------------------------------------------------------------
    def cmd_vel_callback(self, msg: Twist):
        v = msg.linear.x
        w = msg.angular.z

        # Differential drive — same kinematics as before, output in m/s
        v_left  = v - (w * WHEEL_SEPARATION / 2.0)
        v_right = v + (w * WHEEL_SEPARATION / 2.0)

        # Clamp to safe velocity range
        v_left  = max(-MAX_SPEED_MPS, min(MAX_SPEED_MPS, v_left))
        v_right = max(-MAX_SPEED_MPS, min(MAX_SPEED_MPS, v_right))

        # Send SET_VEL — ESP32 PID loop takes it from here
        cmd = f'SET_VEL {v_right:.3f} {v_left:.3f}\n'
        self.ser.write(cmd.encode())
        self.get_logger().info(f'cmd_vel → {cmd.strip()}')

    # ----------------------------------------------------------------
    #  serial_reader
    #  Reads all messages streamed back from ESP32.
    #  Currently logs them — later this is where OdometryNode
    #  will consume ODOM messages for Fix 2.
    #
    #  ESP32 streams at 50 Hz:
    #    ODOM <dA> <dB> <dt_ms>   ← odometry ticks (Fix 2 will use this)
    #    STATUS L=.. R=.. ...     ← general status
    #    WARN WATCHDOG_STOP       ← safety warning
    # ----------------------------------------------------------------
    def serial_reader(self):
        while True:
            try:
                line = self.ser.readline().decode(errors='ignore').strip()
                if not line:
                    continue

                # Log ODOM separately so it's easy to grep
                if line.startswith('ODOM'):
                    self.get_logger().debug(f'ESP32 ODOM: {line}')
                    msg = String()
                    msg.data = line
                    self.odom_pub.publish(msg)
                elif line.startswith('WARN'):
                    self.get_logger().warn(f'ESP32: {line}')
                    msg = String()
                    msg.data = line
                    self.warnings_pub.publish(msg)
                else:
                    self.get_logger().info(f'ESP32: {line}')
                    msg = String()
                    msg.data = line
                    self.telemetry_pub.publish(msg)

            except Exception as e:
                self.get_logger().error(f'Serial read error: {e}')
                break

    # ----------------------------------------------------------------
    #  Shutdown — send STOP to ESP32 before closing serial
    # ----------------------------------------------------------------
    def destroy_node(self):
        self.get_logger().info('Shutting down — sending STOP to ESP32')
        try:
            self.ser.write(b'STOP\n')
            self.ser.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = RoverBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
