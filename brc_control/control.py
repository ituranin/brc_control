#!/usr/bin/env python3
import rclpy
from rcl_interfaces.msg import SetParametersResult
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
from brc_control.controller.pid import PIDController
from brc_control.controller.dyn_pure_pursuit import DynamicPurePursuit
from brc_control.controller.pure_pursuit import PurePursuit
from brc_control.visualization import DebugPublisher


IMAGE_WIDTH = 384
IMAGE_HEIGHT = 384
CONTOUR_BOTTOM_OFFSET = 1
METERS_PER_PIXEL = 30.0 / 384.0  # the top-down image covers 30 m across 384 px


def pixels_to_vehicle_frame(points_px):
    """Image pixels (x right, y down, vehicle at the bottom centre) -> metres in the vehicle
    frame (x forward, y left), ordered as given."""
    return [
        ((IMAGE_HEIGHT - py) * METERS_PER_PIXEL, (IMAGE_WIDTH / 2 - px) * METERS_PER_PIXEL)
        for px, py in points_px
    ]


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
        self.steer_cmd = 0.0
        self.stopwatch = StopWatch()

        controller_name = self.declare_parameter('lateral_controller', 'DynamicPurePursuit').value
        if controller_name == "DynamicPurePursuit":
            self.lat_ctrl = DynamicPurePursuit(wheelbase=1.53)
        elif controller_name == "PurePursuit":
            self.lat_ctrl = PurePursuit(look_ahead_distance=(50.0 * (30.0 / 384.0)), wheelbase=1.53) # look ahead distance calc is pixels to meters
        else:
            raise ValueError(f"Unknown lateral controller: {controller_name}")
        self._declare_tunable_parameters()

        self.image_sub = self.create_subscription(
            Image, 'camera/seg/labels_map', self.image_cb, 1)

        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_cb, 1)

        self.odom_sub = self.create_subscription(
            Odometry, 'brc19/odometry', self.odom_cb, 1)

        self.cmd_pub = self.create_publisher(Twist, 'brc19/cmd_vel', 1)

        # 50 Hz control loop
        self.timer = self.create_timer(1.0 / 30.0, self.control_loop)

        self.velocity_pid = PIDController(0.8, 0.0, 0.0, 0.0)
        
        self.distance_sensor = VirtualDistanceSensor(384)
        self.debug_viz = DebugPublisher(self, IMAGE_WIDTH, METERS_PER_PIXEL)

    def _declare_tunable_parameters(self):
        """Expose the controller's numeric/bool attributes as ROS parameters (defaults = the
        controller's own defaults), so they can be set from a YAML file or live at runtime."""
        self.tunable = [
            name for name, value in vars(self.lat_ctrl).items()
            if not name.startswith(('_', 'last_')) and isinstance(value, (bool, int, float))
        ]
        for name in self.tunable:
            self.declare_parameter(name, getattr(self.lat_ctrl, name))
        # The LUT is a flat list [curvature0, look_ahead0, curvature1, look_ahead1, ...]
        if hasattr(self.lat_ctrl, 'look_ahead_lut'):
            self.declare_parameter(
                'look_ahead_lut', [float(v) for node in self.lat_ctrl.look_ahead_lut for v in node])
        self.vel = self.declare_parameter('velocity', 3.0).value

        # Apply values from the YAML file, then keep listening for live changes
        self._apply_parameters(self.get_parameters(self._tunable_names()))
        self.add_on_set_parameters_callback(self._on_parameters)

    def _tunable_names(self):
        extra = ['velocity'] + (['look_ahead_lut'] if hasattr(self.lat_ctrl, 'look_ahead_lut') else [])
        return self.tunable + extra

    def _apply_parameters(self, params):
        for p in params:
            if p.name == 'velocity':
                self.vel = p.value
            elif p.name == 'look_ahead_lut':
                v = list(p.value)
                self.lat_ctrl.look_ahead_lut = list(zip(v[0::2], v[1::2]))
            elif p.name in self.tunable:
                setattr(self.lat_ctrl, p.name, p.value)

    def _on_parameters(self, params):
        for p in params:
            if p.name == 'look_ahead_lut' and (len(p.value) < 4 or len(p.value) % 2):
                return SetParametersResult(
                    successful=False, reason='look_ahead_lut needs >= 2 (curvature, look_ahead) pairs')
        self._apply_parameters(params)
        return SetParametersResult(successful=True)

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
        cmd = Twist()
        
        cmd.linear.x = speed

        cmd.angular.z = steering_angle

        self.cmd_pub.publish(cmd)

    def control_loop(self):
        if self.label_map is None:
            return
        if self.wheel_fl_vel is None or self.wheel_fr_vel is None:
            return
        if self.speed is None:  # no odometry received yet
            return

        resized = self.label_map

        mask = get_main_path_mask(resized)
        #self.stopwatch.start()
        steering_point, center_points_px, line = steering_point_from_topdown(mask, distance=70)
        #print(self.stopwatch.stop())
        angle = calculate_angle((IMAGE_WIDTH//2, IMAGE_HEIGHT), steering_point)
        angle = math.radians(angle)
        #cv2.imwrite('/mnt/SSDDATA/test.png', line)
        if (abs(angle - self.old_angle) > 0.1):
            #cv2.imwrite('/mnt/SSDDATA/test.png', line)
            print(angle, self.old_angle, steering_point, self.old_point)

        self.old_angle = angle
        self.old_point = steering_point

        #offset = (offset_point[0] - (IMAGE_WIDTH//2)) * (30.0 / 384.0)
        #print(offset)

        # Select control strategy
        ## Longitudinal control (velocity/acceleration)
        vel = self.vel

        ## Lateral control (steering), published convention: positive = left
        points_m = pixels_to_vehicle_frame(center_points_px)
        if self.lat_ctrl.name == "PurePursuit":
            steering = -self.lat_ctrl.step(angle)  # angle is positive to the right of the image
        elif self.lat_ctrl.name == "DynamicPurePursuit":
            steering = self.lat_ctrl.step(points_m)
        else:
            raise ValueError(f"Unknown lateral controller: {self.lat_ctrl.name}")

        # Target point: chosen by the controller if it exposes one, else the single steering point
        target = getattr(self.lat_ctrl, 'target_point', None)
        if target is None and self.lat_ctrl.name == "PurePursuit":
            target = pixels_to_vehicle_frame([steering_point])[0]
        self.debug_viz.publish(mask, points_m, target, steering)
        if self.lat_ctrl.name == "DynamicPurePursuit":
            self.debug_viz.publish_signals(
                steering,
                self.lat_ctrl.last_curvature,
                self.lat_ctrl.last_look_ahead,
                self.lat_ctrl.last_requested_look_ahead,
            )
        else:
            self.debug_viz.publish_signals(steering)

        speed = speed_from_topdown(self.distance_sensor, mask)
        self.velocity_pid.set_setpoint(speed)
        v_pid = self.velocity_pid.update(self.speed)
        v_pid = np.clip(v_pid, -4.5, 1.0)
        self.v_cmd += v_pid * (1.0 / 30.0)
        #print(speed, self.speed, v_pid, self.v_cmd)
        #print(angle, pid_out_steer)

        self.publish_commands(steering_angle=steering, speed=vel)


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
