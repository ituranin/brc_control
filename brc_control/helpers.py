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

def follow_centerline_grid(mask,
                           start_xy,
                           window_size=20,
                           grid_step=10,
                           max_steps=20):

    h, w = mask.shape

    center = np.array(start_xy, dtype=float)

    # initial direction upward
    direction = np.array([0,-1], dtype=float)

    path = [tuple(center.astype(int))]


    # grid offsets around center
    offsets = []

    for dy in range(-window_size, window_size+1, grid_step):
        for dx in range(-window_size, window_size+1, grid_step):
            offsets.append([dx,dy])

    offsets = np.array(offsets)


    for _ in range(max_steps):

        best_score = -1
        best_point = None


        for offset in offsets:

            candidate = center + offset

            x = int(candidate[0])
            y = int(candidate[1])


            if x < 0 or x >= w or y < 0 or y >= h:
                continue


            # only look forward
            movement = candidate-center

            if np.dot(movement, direction) <= 0:
                continue


            # score region around candidate
            x0 = max(0, x-window_size//2)
            x1 = min(w, x+window_size//2)

            y0 = max(0, y-window_size//2)
            y1 = min(h, y+window_size//2)


            region = mask[y0:y1, x0:x1]


            score = np.sum(region)


            # prefer stronger centerline pixels
            if score > best_score:
                best_score = score
                best_point = candidate



        if best_point is None:
            break


        # update direction
        movement = best_point-center

        if np.linalg.norm(movement) > 0:
            direction = movement / np.linalg.norm(movement)


        center = best_point

        path.append(tuple(center.astype(int)))


    return path

def steering_point_from_topdown(track, distance_threshold=20, distance=32):
    trackU = ((track).astype(np.uint8)*255)
    dist = cv2.distanceTransform(trackU, cv2.DIST_L2, 3)
    local_maxima_test = (cv2.compare(dist, cv2.dilate(dist, None), cv2.CMP_EQ) > 0)
    center_line = (trackU > 0) & local_maxima_test
    #xs, ys = np.where(center_line > 0)
    #center_points_t = list(zip(ys, xs))
    center_points_t = follow_centerline_grid(center_line, (384//2, 384-10), max_steps=5)
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