import numpy as np
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray

STEERING_LINE_LENGTH = 3.0  # [m]


def _point(x, y, z=0.0):
    return Point(x=float(x), y=float(y), z=float(z))


class DebugPublisher:
    """Publishes debug topics in the vehicle frame (x forward, y left, origin = image bottom centre).

    - debug/track   (OccupancyGrid): the drivable-path mask
    - debug/markers (MarkerArray): path points, chosen target point, steering line
    - debug/steering (Float64): steering angle [rad] that is published to cmd_vel (positive = left)
    - debug/curvature, debug/look_ahead, debug/look_ahead_requested (Float64): dynamic pure pursuit signals
    - debug/velocity_requested, debug/velocity_cmd (Float64): velocity LUT output and the commanded velocity [m/s]

    In rviz set the Fixed Frame to `frame_id` (default "brc_vehicle").
    """

    def __init__(self, node, image_size_px: int, meters_per_pixel: float, frame_id: str = 'brc_vehicle'):
        self.node = node
        self.frame_id = frame_id
        self.size = image_size_px
        self.mpp = meters_per_pixel
        self.grid_pub = node.create_publisher(OccupancyGrid, 'debug/track', 1)
        self.marker_pub = node.create_publisher(MarkerArray, 'debug/markers', 1)
        self.signal_pubs = {
            name: node.create_publisher(Float64, f'debug/{name}', 1)
            for name in ('steering', 'curvature', 'look_ahead', 'look_ahead_requested',
                         'velocity_requested', 'velocity_cmd')
        }

    def publish_signals(self, steering, curvature=None, look_ahead=None, look_ahead_requested=None,
                        velocity_requested=None, velocity_cmd=None):
        """Scalar signals for rqt_plot / PlotJuggler. None is published as NaN (plots as a gap)."""
        values = {
            'steering': steering,
            'curvature': curvature,
            'look_ahead': look_ahead,
            'look_ahead_requested': look_ahead_requested,
            'velocity_requested': velocity_requested,
            'velocity_cmd': velocity_cmd,
        }
        for name, value in values.items():
            self.signal_pubs[name].publish(Float64(data=float('nan') if value is None else float(value)))

    def _marker(self, marker_id, ns, marker_type, rgb, scale):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.node.get_clock().now().to_msg()
        m.ns = ns
        m.id = marker_id
        m.type = marker_type
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color.r, m.color.g, m.color.b = rgb
        m.color.a = 1.0
        return m

    def publish(self, mask, points_m, target, steering):
        """mask: bool image (rows = forward, cols = right->left flipped as in the camera image).
        points_m: [(x, y)] path points in metres. target: (x, y) or None. steering: [rad], positive = left."""
        stamp = self.node.get_clock().now().to_msg()

        grid = OccupancyGrid()
        grid.header.frame_id = self.frame_id
        grid.header.stamp = stamp
        grid.info.resolution = float(self.mpp)
        grid.info.width = self.size  # cells along x (forward)
        grid.info.height = self.size  # cells along y (left)
        grid.info.origin.position.x = 0.0
        grid.info.origin.position.y = -self.size * self.mpp / 2.0
        grid.info.origin.orientation.w = 1.0
        # grid[j, i]: i along forward, j along left -> image rotated by 90 degrees
        cells = np.asarray(mask, dtype=bool)[::-1, ::-1].T
        grid.data = (cells.astype(np.int8) * 100).ravel().tolist()
        self.grid_pub.publish(grid)

        markers = MarkerArray()

        path = self._marker(0, 'path_points', Marker.SPHERE_LIST, (0.1, 0.4, 1.0), (0.25, 0.25, 0.25))
        path.points = [_point(x, y, 0.05) for x, y in points_m]
        markers.markers.append(path)

        line = self._marker(1, 'path_line', Marker.LINE_STRIP, (0.1, 0.4, 1.0), (0.05, 0.0, 0.0))
        line.points = [_point(x, y, 0.05) for x, y in points_m]
        markers.markers.append(line)

        tgt = self._marker(2, 'target', Marker.SPHERE, (1.0, 0.1, 0.1), (0.6, 0.6, 0.6))
        if target is None:
            tgt.action = Marker.DELETE
        else:
            tgt.pose.position = _point(target[0], target[1], 0.1)
        markers.markers.append(tgt)

        steer = self._marker(3, 'steering', Marker.LINE_STRIP, (0.1, 1.0, 0.1), (0.1, 0.0, 0.0))
        steer.points = [
            _point(0.0, 0.0, 0.1),
            _point(STEERING_LINE_LENGTH * np.cos(steering), STEERING_LINE_LENGTH * np.sin(steering), 0.1),
        ]
        markers.markers.append(steer)

        self.marker_pub.publish(markers)
