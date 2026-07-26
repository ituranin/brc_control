#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist
import numpy as np


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


class ControlNode(Node):
    def __init__(self):
        super().__init__('brc_control')

        self.label_map = None
        self.wheel_fl_vel = None
        self.wheel_fr_vel = None

        self.image_sub = self.create_subscription(
            Image, 'camera/seg/labels_map', self.image_cb, 10)

        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, 'brc19/cmd_vel', 10)

        # 50 Hz control loop
        self.timer = self.create_timer(1.0 / 50.0, self.control_loop)

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

    def control_loop(self):
        if self.label_map is None:
            return
        if self.wheel_fl_vel is None or self.wheel_fr_vel is None:
            return

        # test prints for later control
        velocity_current = (self.wheel_fl_vel + self.wheel_fr_vel) / 2.0
        print(velocity_current)
        print(np.max(self.label_map[...,0] == 2))

        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0

        self.cmd_pub.publish(twist)


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
