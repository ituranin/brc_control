from __future__ import annotations

import math


class PurePursuit:
    """Simple pure pursuit with a fixed look-ahead distance.

    The target angle is measured in the vehicle frame (0 = straight ahead).
    """

    def __init__(
        self,
        look_ahead_distance: float = 2.5,  # [m]
        wheelbase: float = 1.53,  # [m]
        min_steering_angle: float = -0.75,  # [rad]
        max_steering_angle: float = 0.75,  # [rad]
        debug: bool = False,  # print intermediate values on every step
    ):
        self.wheelbase = wheelbase
        self.look_ahead_distance = look_ahead_distance
        self.min_steering_angle = min_steering_angle
        self.max_steering_angle = max_steering_angle
        self.debug = debug


    def step(self, angle: float) -> float:
        """angle: heading error to the target point [rad].
        Returns the steering angle [rad]."""
        # Pure pursuit law: delta = atan(2 * L * sin(alpha) / l_d)
        steering = math.atan2(2.0 * self.wheelbase * math.sin(angle), self.look_ahead_distance)
        if self.debug:
            print(f"angle: {angle:.4f} look_ahead_distance: {self.look_ahead_distance:.4f} steering: {steering:.4f}")
        return max(self.min_steering_angle, min(self.max_steering_angle, steering))
