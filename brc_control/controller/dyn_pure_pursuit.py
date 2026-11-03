from __future__ import annotations

import math


class DynamicPurePursuit:
    """Dynamic pure pursuit on a given target trajectory.

    Per step:
      1. steer with the current look-ahead distance (LAD): pure pursuit towards the path point
         at that distance;
      2. evaluate the path curvature at that same distance;
      3. map it through the LUT (tighter curve -> shorter LAD) and move the LAD towards that
         value with an asymmetric rate limit -> the LAD of the next step.
    On straights the LAD therefore grows step by step, when a curve shows up it shrinks quickly.
    The only steering state kept between steps is the LAD (initialised with `look_ahead_distance`).

    Optional longitudinal control (`dynamic_velocity=True`): the LAD used for steering is mapped
    through `velocity_lut` (long LAD -> fast, short LAD -> slow), clamped to
    [min_velocity, max_velocity] and rate limited like the LAD (slow acceleration, fast braking).
    `step()` returns (steering, velocity); velocity is None when `dynamic_velocity` is off, so the
    caller can keep using a static velocity.
    """

    def __init__(
        self,
        wheelbase: float = 2.7,
        look_ahead_distance: float = 3.0,  # initial / fallback look-ahead [m]
        look_ahead_rise_limit: float | None = 0.02,  # max look-ahead increase per step [m]; None disables rate limiting
        look_ahead_descent_limit: float | None = 0.5,  # max look-ahead decrease per step [m]
        # (abs curvature [1/m], look-ahead distance [m]) nodes
        look_ahead_lut: list[tuple[float, float]] | None = None,
        min_steering_angle: float = -0.75,  # [rad]
        max_steering_angle: float = 0.75,  # [rad]
        debug: bool = False,  # print intermediate values on every step
        dynamic_velocity: bool = False,  # also return a target velocity derived from the LAD
        # (look-ahead distance [m], velocity [m/s]) nodes
        velocity_lut: list[tuple[float, float]] | None = None,
        min_velocity: float = 1.0,  # [m/s]
        max_velocity: float = 10.0,  # [m/s]
        velocity_rise_limit: float | None = 0.03,  # max velocity increase per step [m/s]; None disables rate limiting
        velocity_descent_limit: float | None = 0.1,  # max velocity decrease per step [m/s]
    ):
        self.name = "DynamicPurePursuit"
        self.wheelbase = wheelbase
        self.look_ahead_distance = look_ahead_distance
        self.look_ahead_rise_limit = look_ahead_rise_limit
        self.look_ahead_descent_limit = look_ahead_descent_limit
        self.look_ahead_lut = (
            [(0.1, 4.0), (0.01, 8.0), (0.001, 12.0)] if look_ahead_lut is None else look_ahead_lut
        )
        self.min_steering_angle = min_steering_angle
        self.max_steering_angle = max_steering_angle
        self.debug = debug
        self.dynamic_velocity = dynamic_velocity
        self.velocity_lut = [(3.0, 2.0), (6.0, 3.0), (12.0, 5.0)] if velocity_lut is None else velocity_lut
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.velocity_rise_limit = velocity_rise_limit
        self.velocity_descent_limit = velocity_descent_limit
        self._velocity: float | None = None  # state: last target velocity
        self._look_ahead: float | None = None  # state: LAD used in the next step
        # Last step's results, exposed for visualisation
        self.target_point: tuple[float, float] | None = None
        self.last_look_ahead: float | None = None  # LAD that was used for the steering
        self.last_curvature: float | None = None  # signed curvature at that LAD
        self.last_requested_look_ahead: float | None = None  # LUT output before rate limiting
        self.last_velocity: float | None = None  # target velocity returned by the last step
        self.last_requested_velocity: float | None = None  # velocity LUT output before rate limiting

    def reset(self) -> None:
        self._look_ahead = None
        self.target_point = None
        self.last_look_ahead = None
        self.last_curvature = None
        self.last_requested_look_ahead = None
        self._velocity = None
        self.last_velocity = None
        self.last_requested_velocity = None

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _lut_eval(nodes: list[tuple[float, float]], value: float) -> float:
        """Piecewise linear interpolation, clamped at both ends."""
        nodes = sorted(nodes)
        if value <= nodes[0][0]:
            return nodes[0][1]
        if value >= nodes[-1][0]:
            return nodes[-1][1]
        for (x0, y0), (x1, y1) in zip(nodes, nodes[1:]):
            if value <= x1:
                return y0 + (value - x0) / (x1 - x0) * (y1 - y0)
        return nodes[-1][1]

    @staticmethod
    def _calc_curvature(p, p_prev, p_next) -> float:
        """Curvature at point p from a non-uniform central difference."""
        x, y = p
        x_p, y_p = p_prev
        x_n, y_n = p_next
        h_p = math.hypot(x - x_p, y - y_p)
        h_n = math.hypot(x_n - x, y_n - y)
        if h_p < 1e-9 or h_n < 1e-9:
            return 0.0
        denom = h_p * h_n * (h_p + h_n)
        dx = (h_p**2 * x_n + (h_n**2 - h_p**2) * x - h_n**2 * x_p) / denom
        dy = (h_p**2 * y_n + (h_n**2 - h_p**2) * y - h_n**2 * y_p) / denom
        ddx = 2 * (h_p * x_n - (h_p + h_n) * x + h_n * x_p) / denom
        ddy = 2 * (h_p * y_n - (h_p + h_n) * y + h_n * y_p) / denom
        speed_sq = dx * dx + dy * dy
        if speed_sq == 0:
            return 0.0
        return (dx * ddy - dy * ddx) / speed_sq**1.5

    @staticmethod
    def _calc_curvature_backward(p, p_prev, p_prev_prev) -> float:
        """Curvature at the last point p from a backward difference (no next point available)."""
        x, y = p
        x_p, y_p = p_prev
        x_pp, y_pp = p_prev_prev
        h1 = math.hypot(x_p - x_pp, y_p - y_pp)
        h2 = math.hypot(x - x_p, y - y_p)
        if h1 < 1e-9 or h2 < 1e-9:
            return 0.0
        a = h2 / (h1 * (h1 + h2))
        b = (h1 + h2) / (h1 * h2)
        c = (h1 + 2 * h2) / (h2 * (h1 + h2))
        dx = a * x_pp - b * x_p + c * x
        dy = a * y_pp - b * y_p + c * y
        ddx = 2 * (x_pp / (h1 * (h1 + h2)) - x_p / (h1 * h2) + x / ((h1 + h2) * h2))
        ddy = 2 * (y_pp / (h1 * (h1 + h2)) - y_p / (h1 * h2) + y / ((h1 + h2) * h2))
        speed_sq = dx * dx + dy * dy
        if speed_sq == 0:
            return 0.0
        return (dx * ddy - dy * ddx) / speed_sq**1.5

    @classmethod
    def _point_curvatures(cls, points) -> list[float | None]:
        """Curvature per point; None where it cannot be computed (first point, duplicates)."""
        # Interior points use the central formula; the last point has no successor, so it uses the backward one
        n = len(points)
        curvatures: list[float | None] = [None] * n
        for i in range(1, n):
            prev = points[i - 1]
            dist_prev = math.hypot(points[i][0] - prev[0], points[i][1] - prev[1])
            if dist_prev < 1e-9:
                continue
            if i < n - 1:
                nxt = points[i + 1]
                if math.hypot(nxt[0] - points[i][0], nxt[1] - points[i][1]) < 1e-9:
                    continue
                curvatures[i] = cls._calc_curvature(points[i], prev, nxt)
            elif i >= 2:
                pp = points[i - 2]
                if math.hypot(prev[0] - pp[0], prev[1] - pp[1]) < 1e-9:
                    continue
                curvatures[i] = cls._calc_curvature_backward(points[i], prev, pp)
        return curvatures

    def _curvature_at(self, points, distance: float) -> float | None:
        """Curvature at the given distance from the vehicle, linearly interpolated between the
        two neighbouring path points; clamped to the first/last valid point if the distance lies
        outside the path."""
        curvatures = self._point_curvatures(points)
        valid = [
            (math.hypot(*pt), c)
            for pt, c in zip(points, curvatures)
            if c is not None and math.isfinite(c)
        ]
        if not valid:
            return None
        if distance <= valid[0][0]:
            return valid[0][1]
        for (d0, c0), (d1, c1) in zip(valid, valid[1:]):
            if d0 <= distance <= d1:
                if d1 - d0 < 1e-9:
                    return c0
                return c0 + (distance - d0) / (d1 - d0) * (c1 - c0)
        return valid[-1][1]

    @staticmethod
    def _rate_limit(current: float, requested: float, rise_limit, descent_limit) -> float:
        """Asymmetric per-step rate limiter; a None limit disables limiting."""
        if rise_limit is None or descent_limit is None:
            return requested
        delta = requested - current
        return current + max(-descent_limit, min(rise_limit, delta))

    def _next_velocity(self, look_ahead: float) -> float:
        """Target velocity for the LAD that is used for steering in this step."""
        requested = self._lut_eval(self.velocity_lut, look_ahead)
        requested = max(self.min_velocity, min(self.max_velocity, requested))
        self.last_requested_velocity = requested
        current = self.min_velocity if self._velocity is None else self._velocity  # start slow
        self._velocity = self._rate_limit(
            current, requested, self.velocity_rise_limit, self.velocity_descent_limit)
        return self._velocity

    def _next_look_ahead(self, points, current: float) -> tuple[float, float | None, float | None]:
        """Returns (look-ahead for the next step, curvature at `current`, LUT output).
        Without a usable curvature (or LUT) the look-ahead is held."""
        curvature = self._curvature_at(points, current) if self.look_ahead_lut else None
        if curvature is None:
            return current, None, None
        requested = self._lut_eval(self.look_ahead_lut, abs(curvature))  # tighter curve -> shorter
        return (self._rate_limit(current, requested, self.look_ahead_rise_limit, self.look_ahead_descent_limit),
                curvature, requested)

    @staticmethod
    def _point_at_distance(points, distance: float) -> tuple[float, float]:
        """First point along the path at the given Euclidean distance from the vehicle (linear
        interpolation between neighbours). Falls back to the last point if the path is shorter."""
        if math.hypot(*points[0]) >= distance:
            return points[0]
        for (ax, ay), (bx, by) in zip(points, points[1:]):
            if math.hypot(bx, by) < distance:
                continue
            # Solve |a + t * (b - a)| = distance for t in [0, 1]
            dx, dy = bx - ax, by - ay
            qa = dx * dx + dy * dy
            qb = 2.0 * (ax * dx + ay * dy)
            qc = ax * ax + ay * ay - distance * distance
            t = (-qb + math.sqrt(max(0.0, qb * qb - 4.0 * qa * qc))) / (2.0 * qa)
            return ax + t * dx, ay + t * dy
        return points[-1]

    def step(self, points) -> tuple[float, float | None]:
        """points: [(x, y), ...] target trajectory in the vehicle frame (x forward, y left),
        ordered from the vehicle outward. Returns (steering angle [rad], target velocity [m/s]);
        the velocity is None if `dynamic_velocity` is off (or not available yet)."""
        points = [pt for pt in points if pt[0] > 0.0]  # ignore points behind the vehicle
        if not points:
            self.target_point = None
            return 0.0, self._velocity if self.dynamic_velocity else None  # hold the last velocity

        if self._look_ahead is None:
            self._look_ahead = self.look_ahead_distance
        look_ahead = self._look_ahead

        # 1. Classic pure pursuit with the current look-ahead
        target_x, target_y = self._point_at_distance(points, look_ahead)
        self.target_point = (target_x, target_y)
        self.last_look_ahead = look_ahead
        # Pure pursuit law: delta = atan(2 * L * sin(alpha) / l_d)
        heading_error = math.atan2(target_y, target_x)
        steering = math.atan2(2.0 * self.wheelbase * math.sin(heading_error), look_ahead)

        # 2. + 3. Curvature at that look-ahead -> look-ahead for the next step
        self._look_ahead, curvature, requested = self._next_look_ahead(points, look_ahead)
        self.last_curvature = curvature
        self.last_requested_look_ahead = requested

        # 4. Optional: target velocity from the LAD used in this step
        velocity = self._next_velocity(look_ahead) if self.dynamic_velocity and self.velocity_lut else None
        self.last_velocity = velocity
        if self.debug:
            curvature_str = "n/a" if curvature is None else f"{curvature:.4f}"
            print(f"curvature: {curvature_str} look_ahead: {look_ahead:.4f} "
                  f"target: ({target_x:.4f}, {target_y:.4f}) "
                  f"heading_error: {heading_error:.4f} steering: {steering:.4f} velocity: {velocity}")
        return max(self.min_steering_angle, min(self.max_steering_angle, steering)), velocity
