import numpy as np
import cv2

from brc_control.sensors import VirtualDistanceSensor

def steering_point_from_polyfit(center_points_t, image_height, distance, degree=2):
    if len(center_points_t) < degree + 1:
        return None  # not enough points to fit — let caller hold last-good angle

    pts = np.array(center_points_t)
    xs, ys = pts[:, 0], pts[:, 1]

    coeffs = np.polyfit(ys, xs, degree)
    poly = np.poly1d(coeffs)

    target_y = image_height - distance  # "distance" pixels ahead, roughly
    target_x = poly(target_y)

    return (float(target_x), float(target_y))

def calculate_angle(origin, point):
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]

    return np.degrees(np.arctan2(dx, -dy))

def steering_point_from_topdown(track, distance_threshold=20, distance=32):
    trackU = ((track).astype(np.uint8)*255)
    dist = cv2.distanceTransform(trackU, cv2.DIST_L2, 3)
    local_maxima_test = (cv2.compare(dist, cv2.dilate(dist, None), cv2.CMP_EQ) > 0)
    center_line = (trackU > 0) & local_maxima_test
    xs, ys = np.where(center_line > 0)
    center_points_t = list(zip(ys, xs))
    point_dist = 0.0
    point = steering_point_from_polyfit(center_points_t, 384, distance, 3)
    return point, point_dist, center_line

def calculate_max_target_speed(distance, max_deceleration, min_speed):
    initial_speed = min_speed / 3.6  # Convert minimum speed from km/h to m/s
    max_speed_squared = (initial_speed ** 2) - (2 * -max_deceleration * distance)
    
    if max_speed_squared < 0:
        return 0
    else:
        max_speed = max_speed_squared ** 0.5
        return max_speed

def speed_from_topdown(sensor: VirtualDistanceSensor, img):
    dists = sensor.getDistances(image=img)
    speed = calculate_max_target_speed(max(dists[8], 0.0), 4.5, 0.0)
    return speed#max(speeds) * 3.6