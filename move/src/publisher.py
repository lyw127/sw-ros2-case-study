#!/usr/bin/env python3
"""Starter node for the Lunabotics ROS 2 case study.

Fill in TASKS 1-3 here. See README.md for the full description of each task.

Run it with:

    ros2 run move publisher

As shipped this node starts, spins, and does nothing -- that is intentional. Use it to
confirm your workspace is built and sourced before you write any logic.

Each task is marked with a TASK n.n comment matching the README. Commented-out lines are
deliberate: uncomment and complete them.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import math
from sensor_msgs.msg import Imu

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64


class RobotController(Node):
    """Drives the robot, tracks its position error, and flags obstacles."""

    def __init__(self):
        super().__init__('robot_controller')

        # Publish to /error when the position delta exceeds this (TASK 2.3).
        # This is a starting value -- justify whatever you settle on.
        self.error_thresh = 0.5  # meters. ignore one noisy sample, still catch imu drift

        # ---- TASK 1.2: publisher that drives the robot ---------------------
        # Which topic moves the robot? Find it first (TASK 1.1), then uncomment.
        #
        self.move_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        #
        # Then drive it on a timer:
        self.move_timer = self.create_timer(0.1, self.send_move_cmd)

        # ---- TASK 1.3: the path you chose ----------------------------------
        # Pick a route that gets the robot around the wall, and represent it
        # however you think is best -- a list of waypoints, a sequence of
        # timed velocity commands, a parametric curve, something else.
        #
        # Document HERE why you chose this path and this representation.
        # That reasoning is a large part of what we are evaluating.

        # wall at (6, 0), size 0.5 x 6 -> covers y in [-3, 3], x in [5.75, 6.25]
        # pole at (4, 2.5), north side, so go around the south end
        # robot is 2m long, 1m wide. chassis starts near (0.5, 0)
        # timed commands (forward m/s, turn rad/s, s) -- open loop, world is static
        speed = 0.5
        turn = 0.4
        turn_90 = math.pi / (2.0 * turn)          # exact 90 deg: (pi/2) / rate
        south_m = 3.0 + 1.0 + 0.5                 # wall half-width + half robot length + gap
        past_m = (6.25 + 1.0 + 1.25) - 0.5        # past wall face + half length + margin - start x
        self.path = [
            (0.0, -turn, turn_90),    # -90 deg, face south (-y). +turn would face the pole
            (speed, 0.0, south_m / speed),  # 4.5 m, y ~ -4.5, south of the wall
            (0.0, turn, turn_90),     # +90 deg, face +x again
            (speed, 0.0, past_m / speed),   # 8 m, chassis past x=6.25
            (0.0, 0.0, 1e9),          # hold stop. 1e9 s so we never leave this segment
        ]
        self.path_index = 0
        self.segment_elapsed = 0.0

        # ---- TASK 2.2: subscriber for the robot's 6D pose ------------------
        # One of the two onboard sensors reports 6D data. Find it (TASK 2.1).
        #
        # /imu is 6d (3 accel + 3 gyro) but has no x,y,z. odometry is the real pose from the sim
        self.robot_pos_sub = self.create_subscription(
            Imu, '/imu', self.on_robot_pos, qos_profile_sensor_data)  # best effort, same as the sensor
        self.odom_sub = self.create_subscription(
            Odometry, '/model/vehicle_blue/odometry', self.on_odom, qos_profile_sensor_data)  # ground truth

        # ---- TASK 2.3: where the measured-vs-actual error goes -------------
        self.error_pub = self.create_publisher(Float64, '/error', 10)
        #
        # Hint: ground truth for "actual" is published by the simulator on the
        # robot's odometry topic (nav_msgs/Odometry). Deciding what to compare,
        # and in which frame, is part of the task.

        self.truth_xy = None    # real (x, y) from odometry
        self.est_x = 0.0        # imu guess of x, meters
        self.est_y = 0.0        # imu guess of y, meters
        self.vel_x = 0.0        # integrated speed, m/s
        self.vel_y = 0.0
        self.last_imu_t = None  # need 2 imu stamps before we have a dt
        self.last_ax = None     # last world-frame accel, for trapezoid
        self.last_ay = None

        # ---- TASK 3: lidar in, filtered obstacles out ----------------------
        # The lidar has a single vertical sample, so this cloud is one flat
        # row of points at the sensor's height -- not a 3D volume.
        #
        self.lidar_sub = self.create_subscription(
            PointCloud2, '/lidar/points', self.on_lidar, qos_profile_sensor_data)
        self.obstacle_pub = self.create_publisher(
            PointCloud2, '/obstacle_cloud', 10)  # poles only

        self.get_logger().info('robot_controller started')

    # -----------------------------------------------------------------------
    # TASK 1.2 -- publish a velocity command
    # -----------------------------------------------------------------------
    def send_move_cmd(self):
        """Publish one Twist that moves the robot along self.path.

        TODO: build the Twist and publish it on self.move_pub.
        """
        dt = 0.1   # timer period in __init__
        vx, wz, _ = self.path[self.path_index]

        # The last segment is the stop command, so never advance past it.
        self.segment_elapsed += dt
        if (self.segment_elapsed >= self.path[self.path_index][2]   # leave once its time is up
                and self.path_index < len(self.path) - 1):   # last entry is stop, don't go past it
            self.path_index += 1
            self.segment_elapsed = 0.0
            vx, wz, _ = self.path[self.path_index]

        # differential drive
        msg = Twist()
        msg.linear.x = vx   # linear.x forward speed
        msg.angular.z = wz   # angular.z yaw rate
        self.move_pub.publish(msg)

    # -----------------------------------------------------------------------
    # TASK 2.3 -- compare reported position against ground truth
    # -----------------------------------------------------------------------
    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.truth_xy = (p.x, p.y)  # real x,y in meters

    def on_robot_pos(self, msg):
        """Compare the sensor's idea of where we are against the truth.

        Publish a Float64 on self.error_pub when the delta exceeds
        self.error_thresh.

        TODO: decide what "delta" means here and justify it in a comment.
        """
        # delta = straight-line distance (m) between our guess and odometry, both in the odom frame
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9  # stamp in seconds
        if self.last_imu_t is None:
            self.last_imu_t = t  # first sample, no dt yet
            return
        dt = t - self.last_imu_t
        self.last_imu_t = t
        if dt <= 0.0 or self.truth_xy is None:  # bad stamp, or odom hasn't arrived
            return

        q = msg.orientation
        ax, ay = msg.linear_acceleration.x, msg.linear_acceleration.y  # robot frame. skip z, that's gravity
        # zyx yaw: these two are already sin(yaw), cos(yaw). skip atan2 + sin/cos
        sin_y = 2.0 * (q.w * q.z + q.x * q.y)
        cos_y = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        n = math.hypot(sin_y, cos_y)
        if n > 0.0:
            sin_y /= n
            cos_y /= n
        ax_w = cos_y * ax - sin_y * ay  # rotate accel into odom frame
        ay_w = sin_y * ax + cos_y * ay

        # trapezoid: use avg of last and this sample. imu is only 1 hz so this matters
        if self.last_ax is None:
            self.last_ax, self.last_ay = ax_w, ay_w
            return
        vx_new = self.vel_x + 0.5 * (self.last_ax + ax_w) * dt
        vy_new = self.vel_y + 0.5 * (self.last_ay + ay_w) * dt
        self.est_x += 0.5 * (self.vel_x + vx_new) * dt
        self.est_y += 0.5 * (self.vel_y + vy_new) * dt
        self.vel_x, self.vel_y = vx_new, vy_new
        self.last_ax, self.last_ay = ax_w, ay_w

        dx = self.est_x - self.truth_xy[0]
        dy = self.est_y - self.truth_xy[1]
        delta = math.hypot(dx, dy)  # sqrt(dx^2 + dy^2)
        if delta > self.error_thresh:  # integrating accel twice drifts, so this will fire
            out = Float64()
            out.data = delta
            self.error_pub.publish(out)

    # -----------------------------------------------------------------------
    # TASK 3.3 -- classify a single lidar point
    # -----------------------------------------------------------------------
    def is_obstacle(self, point):
        """Return True if `point` is something we must avoid.

        The barrier is passable -- treat it like dust in the air. The poles are
        not. `point` is an (x, y, z) tuple in the lidar's frame.

        TODO: decide what separates a pole from the barrier and implement it.
        """
        # point is (x, y, z, intensity). wall laser return is ~0, pole is ~2000
        return point[3] > 100.0

    # -----------------------------------------------------------------------
    # TASK 3.2 -- filter the scan and republish what matters
    # -----------------------------------------------------------------------
    def on_lidar(self, msg):
        """Filter incoming points through is_obstacle and republish.

        point_cloud2.read_points(msg, field_names=('x', 'y', 'z')) iterates the
        cloud; point_cloud2.create_cloud_xyz32(msg.header, pts) builds the
        outgoing one.

        TODO: keep only the obstacle points and publish on self.obstacle_pub.
        """
        kept = []
        for p in point_cloud2.read_points(
                msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=True):
            if self.is_obstacle(p):
                kept.append((p[0], p[1], p[2]))  # published cloud is just xyz
        self.obstacle_pub.publish(point_cloud2.create_cloud_xyz32(msg.header, kept))


def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
