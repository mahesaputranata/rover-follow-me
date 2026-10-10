#!/usr/bin/env python3
"""Fake vision: publishes /bbox_error [offset_x, offset_y, area_ratio, valid].
  python3 fake_vision.py open     scripted offsets, ignores the rover
  python3 fake_vision.py closed   person fixed in the world, offset computed from rover yaw
"""
import math, sys
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

RATE = 15.0        # Hz, roughly the tracker rate
HALF_FOV = 0.55    # rad; offset_x = +-1 at the image edge
TRACK = 0.20       # m, only used to print wheel speeds

# open:   (duration_s, label, offset_x, area_ratio, valid)
OPEN = [
    (3.0, "hold-centered", 0.0, 1.0, 1),   # expect cmd ~0
    (3.0, "far-centered",  0.0, 0.5, 1),   # expect forward ~0.2
    (2.0, "person-right",  0.3, 1.0, 1),   # expect right wheel < left
    (2.0, "person-left",  -0.3, 1.0, 1),   # expect left wheel < right
    (2.0, "far-right",     0.3, 0.5, 1),   # expect both ~+0.25, L > R by ~0.08
    (2.0, "far-left",     -0.3, 0.5, 1),   # expect both ~+0.25, R > L by ~0.08
    (2.0, "too-close",     0.0, 1.5, 1),   # expect reverse
    (2.0, "lost",          0.0, 1.0, 0),   # expect stop
]
# closed: (duration_s, label, person_bearing_rad rel. to start heading, area_ratio, valid)
#CLOSED = [
#    (3.0, "ahead",  0.0, 1.0, 1),
#    (6.0, "right", -0.3, 1.0, 1),
#    (6.0, "left",   0.3, 1.0, 1),
#    (3.0, "lost",   0.0, 1.0, 0),
#]
CLOSED = [
    (3.0, "ahead",  0.0, 0.5, 1),
    (6.0, "right", -0.3, 0.5, 1),
    (6.0, "left",   0.3, 0.5, 1),
    (3.0, "lost",   0.0, 1.0, 0),
]

class FakeVision(Node):
    def __init__(self, mode):
        super().__init__("fake_vision")
        self.mode = mode
        self.prof = OPEN if mode == "open" else CLOSED
        self.pub = self.create_publisher(Float32MultiArray, "/bbox_error", 10)
        self.create_subscription(Odometry, "/rover/state", self.on_state, qos_profile_sensor_data)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.psi = None
        self.psi0 = None
        self.cmd = Twist()
        self.t0 = None
        self.tick = 0
        self.done = False
        self.create_timer(1.0 / RATE, self.step)

    def on_state(self, m):
        q = m.pose.pose.orientation
        self.psi = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        if self.psi0 is None:
            self.psi0 = self.psi

    def on_cmd(self, m):
        self.cmd = m

    def send(self, off, area, valid):
        msg = Float32MultiArray()
        msg.data = [float(off), 0.0, float(area), float(valid)]
        self.pub.publish(msg)

    def step(self):
        if self.mode == "closed" and self.psi is None:
            if self.tick % int(RATE) == 0:
                self.get_logger().warn("waiting for /rover/state ...")
            self.tick += 1
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0
        acc = 0.0
        phase = None
        for p in self.prof:
            if t < acc + p[0]:
                phase = p
                break
            acc += p[0]
        if phase is None:
            self.done = True
            return
        _, label, x, area, valid = phase
        if self.mode == "open":
            off = x
        else:
            err = (self.psi0 + x) - self.psi          # + = person to the left
            off = max(-1.0, min(1.0, -err / HALF_FOV))  # + offset = person to the right
        self.send(off, area, valid)
        if self.tick % 3 == 0:
            lin, ang = self.cmd.linear.x, self.cmd.angular.z
            vl, vr = lin - ang * TRACK / 2, lin + ang * TRACK / 2
            extra = "" if self.mode == "open" else f" yaw={self.psi - self.psi0:+.2f}"
            print(f"t={t:5.1f} {label:14s} off={off:+.2f} area={area:.2f} v={valid}"
                  f" | cmd L/R={vl:+.3f}/{vr:+.3f}{extra}", flush=True)
        self.tick += 1


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "open"
    rclpy.init()
    node = FakeVision(mode)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(3):
            node.send(0.0, 1.0, 0)   # tell lqr_node the target is lost
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
