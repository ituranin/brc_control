#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
#from std_msgs.msg import Float64
import numpy as np
import cv2
import math


from brc_control.helpers import steering_point_from_topdown, calculate_angle, speed_from_topdown
from brc_control.sensors import VirtualDistanceSensor
from brc_control.stopwatch import StopWatch


IMAGE_WIDTH = 384
IMAGE_HEIGHT = 384
CONTOUR_BOTTOM_OFFSET = 1


# encoding -> (numpy dtype, channels)
ENCODING_MAP = {
    'mono8':  (np.uint8, 1),
    '8UC1':   (np.uint8, 1),
    'rgb8':   (np.uint8, 3),
    'bgr8':   (np.uint8, 3),
    'rgba8':  (np.uint8, 4),
    'mono16': (np.uint16, 1),
    '16UC1':  (np.uint16, 1),
}


def image_to_numpy(msg: Image) -> np.ndarray:
    """Convert sensor_msgs/Image to a numpy array without cv_bridge."""
    if msg.encoding not in ENCODING_MAP:
        raise ValueError(f'Unsupported encoding: {msg.encoding}')

    dtype, channels = ENCODING_MAP[msg.encoding]
    buf = np.frombuffer(msg.data, dtype=dtype)

    if channels == 1:
        img = buf.reshape(msg.height, msg.width)
    else:
        img = buf.reshape(msg.height, msg.width, channels)

    # msg.step accounts for row padding; slice it off if present
    expected_row_bytes = msg.width * channels * np.dtype(dtype).itemsize
    if msg.step != expected_row_bytes:
        row_stride = msg.step // np.dtype(dtype).itemsize
        buf2 = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, row_stride)
        if channels == 1:
            img = buf2[:, :msg.width]
        else:
            img = buf2[:, :msg.width * channels].reshape(msg.height, msg.width, channels)

    return img


def get_main_path_mask(img, point=(IMAGE_WIDTH//2, IMAGE_HEIGHT-CONTOUR_BOTTOM_OFFSET)):
    contours, _ = cv2.findContours((img[..., 0] == 2).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(img.shape).astype(np.uint8)
    for contour in contours:
        result = cv2.pointPolygonTest(contour, point, False)
        if result >= 0:
            cv2.fillPoly(mask, pts=[contour], color=(255,255,255))
            return mask[...,0] > 0
    return np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.uint8)


def steering_angle_to_twist(steering_angle, speed, wheel_base=1.53):
    """
    Convert Ackermann steering angle to geometry_msgs/Twist.

    Args:
        steering_angle (float): Desired steering angle in radians.
                                Positive = left turn.
        speed (float): Desired forward speed in m/s.
        wheel_base (float): Distance between front and rear axle in meters.

    Returns:
        Twist: ROS Twist command.
    """

    cmd = Twist()

    # Forward velocity
    cmd.linear.x = speed

    # Convert steering angle to yaw rate
    if abs(speed) < 1e-6:
        cmd.angular.z = 0.0
    else:
        cmd.angular.z = speed * math.tan(steering_angle) / wheel_base

    return cmd


class PIDController:
    def __init__(self, kp, ki, kd, setpoint = 0):

        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.setpoint = setpoint  # Desired value

        # Internal variables to store PID computations
        self._previous_error = 0.0
        self._integral = 0.0
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0

    def update(self, current_value):
        # Calculate error
        error = self.setpoint - current_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self._integral += error
        i_term = self.ki * self._integral
        
        # Derivative term

        derivative = (error - self._previous_error)
        d_term = self.kd * derivative
       
        # Update previous error for next iteration
        self._previous_error = error
        
        # Control output
        control_output = p_term + i_term + d_term
        self.p_term = p_term
        self.i_term = i_term
        self.d_term = d_term
        return control_output

    def reset_controller(self):
        self._previous_error = 0.0
        self._integral = 0.0

    def set_setpoint(self, setpoint):
        self.setpoint = setpoint

    def get_gains(self):
        return self.kp, self.ki, self.kd 
    
    def set_gains(self, kp=None, ki=None, kd=None):
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd


class ControlNode(Node):
    def __init__(self):
        super().__init__('brc_control')

        self.label_map = None
        self.wheel_fl_vel = None
        self.wheel_fr_vel = None
        self.speed = None
        self.old_angle = 0.0
        self.old_point = None
        self.v_cmd = 0.0
        self.stopwatch = StopWatch()

        self.image_sub = self.create_subscription(
            Image, 'camera/seg/labels_map', self.image_cb, 1)

        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_cb, 1)

        self.odom_sub = self.create_subscription(
            Odometry, 'brc19/odometry', self.odom_cb, 1)

        self.cmd_pub = self.create_publisher(Twist, 'brc19/cmd_vel', 1)

        # 50 Hz control loop
        self.timer = self.create_timer(1.0 / 30.0, self.control_loop)

        #self.steering_pid = PIDController(0.4, 0.00005, 0.0, 0.0)
        self.steering_pid = PIDController(0.325, 0.0, 0.2, 0.0)

        self.velocity_pid = PIDController(0.8, 0.0, 0.0, 0.0)
        
        self.distance_sensor = VirtualDistanceSensor(384)

    def image_cb(self, msg: Image):
        try:
            self.label_map = image_to_numpy(msg)
        except ValueError as e:
            self.get_logger().warn(str(e))

    def joint_cb(self, msg: JointState):
        if 'wheel_fl_joint' in msg.name:
            idx = msg.name.index('wheel_fl_joint')
            self.wheel_fl_vel = msg.velocity[idx]
        if 'wheel_fr_joint' in msg.name:
            idx = msg.name.index('wheel_fr_joint')
            self.wheel_fr_vel = msg.velocity[idx]

    def odom_cb(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.speed = (vx**2 + vy**2) ** 0.5

    def publish_commands(self, steering_angle: float, speed: float):
        twist = steering_angle_to_twist(steering_angle=steering_angle, speed=speed)

        self.cmd_pub.publish(twist)

    def control_loop(self):
        if self.label_map is None:
            return
        if self.wheel_fl_vel is None or self.wheel_fr_vel is None:
            return

        resized = self.label_map

        mask = get_main_path_mask(resized)
        #self.stopwatch.start()
        steering_point, dist, line = steering_point_from_topdown(mask, distance=60)
        #print(self.stopwatch.stop())
        angle = calculate_angle((IMAGE_WIDTH//2, IMAGE_HEIGHT), steering_point)
        angle = math.radians(angle)

        if (abs(angle - self.old_angle) > 0.1):
            #cv2.imwrite('/mnt/SSDDATA/test.png', line)
            print(angle, self.old_angle, dist, steering_point, self.old_point)

        self.old_angle = angle
        self.old_point = steering_point

        pid_out_steer = self.steering_pid.update(angle)

        speed = speed_from_topdown(self.distance_sensor, mask)
        self.velocity_pid.set_setpoint(speed)
        v_pid = self.velocity_pid.update(self.speed)
        v_pid = np.clip(v_pid, -4.5, 1.0)
        self.v_cmd += v_pid * (1.0 / 30.0)
        #print(speed, self.speed, v_pid, self.v_cmd)

        self.publish_commands(steering_angle=pid_out_steer, speed=self.v_cmd)


def main():
    rclpy.init()
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
