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

def steering_point_from_parametric_fit(center_points_t, image_height, distance, degree=3):
    if len(center_points_t) < degree + 1:
        return None

    pts = np.array(center_points_t, dtype=np.float64)

    # order points along the path — arc-length parameterization requires
    # them in path order, not fitting order. If your points already come
    # in path order (e.g. from follow_centerline_grid), skip re-sorting.
    xs, ys = pts[:, 0], pts[:, 1]

    # cumulative arc length as the parameter t
    deltas = np.diff(pts, axis=0)
    seg_lengths = np.sqrt((deltas ** 2).sum(axis=1))
    t = np.concatenate([[0], np.cumsum(seg_lengths)])

    if t[-1] == 0:
        return None  # all points coincide

    # normalize t to [0, 1] for numerical stability in polyfit
    t_norm = t / t[-1]

    coeffs_x = np.polyfit(t_norm, xs, degree)
    coeffs_y = np.polyfit(t_norm, ys, degree)
    poly_x = np.poly1d(coeffs_x)
    poly_y = np.poly1d(coeffs_y)

    # find target_t such that poly_y(target_t) == target_y
    # (poly_y isn't invertible in closed form for degree > 1, so sample
    # densely and pick the closest match)
    target_y = image_height - distance
    t_samples = np.linspace(0, 1, 200)
    y_samples = poly_y(t_samples)
    closest_idx = np.argmin(np.abs(y_samples - target_y))
    target_t = t_samples[closest_idx]

    target_x = poly_x(target_t)
    target_y_actual = poly_y(target_t)  # may differ slightly from target_y

    return (float(target_x), float(target_y_actual))

def calculate_angle(origin, point):
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]

    return np.degrees(np.arctan2(dx, -dy))

def follow_centerline_grid(mask, start_xy, step=15, search_width=50, grid_step=10, max_steps=100):
    # made with chatgpt based on an idea i had
    h, w = mask.shape

    center = np.array(start_xy, dtype=float)
    direction = np.array([0.0, -1.0])

    path = [center.copy()]

    for _ in range(max_steps):

        # perpendicular direction
        perp = np.array([-direction[1], direction[0]])

        # predicted point forward
        prediction = center + direction * step

        best_score = -1
        best_point = None

        # search only sideways
        for offset in np.arange(-search_width, search_width+1, grid_step):

            candidate = prediction + perp * offset

            x, y = candidate.astype(int)

            if x < 0 or x >= w or y < 0 or y >= h:
                continue

            r = 10
            x0 = max(0, x-r)
            x1 = min(w, x+r)
            y0 = max(0, y-r)
            y1 = min(h, y+r)

            score = np.count_nonzero(mask[y0:y1, x0:x1])

            if score > best_score:
                best_score = score
                best_point = candidate


        if best_point is None:
            break


        # center only on detected pixels
        x, y = best_point.astype(int)

        r = 10
        roi = mask[max(0,y-r):min(h,y+r),
                   max(0,x-r):min(w,x+r)]

        ys, xs = np.where(roi)

        if len(xs):
            new_point = np.array([
                max(0,x-r)+xs.mean(),
                max(0,y-r)+ys.mean()
            ])
        else:
            new_point = best_point


        # update direction
        movement = new_point - center

        if np.linalg.norm(movement) > 1e-6:
            direction = movement / np.linalg.norm(movement)

        center = new_point
        path.append(center.copy())

    path = path[1:]

    return np.array(path)

def steering_point_from_topdown(track, distance=32, debug=False):
    trackU = ((track).astype(np.uint8)*255)
    dist = cv2.distanceTransform(trackU, cv2.DIST_L2, 3)
    local_maxima_test = (cv2.compare(dist, cv2.dilate(dist, None), cv2.CMP_EQ) > 0)
    center_line = (trackU > 0) & local_maxima_test
    center_points_t = follow_centerline_grid(center_line, (384//2, 384-10), max_steps=10, search_width=40, step=20, grid_step=10)
    point_dist = 0.0
    degree = 3
    point = steering_point_from_parametric_fit(center_points_t, 384, distance, degree)
    
    if debug:
        center_line = np.zeros((384,384), dtype=np.uint8)#center_line.astype(np.uint8) * 255

        if len(center_points_t) >= degree + 1:
            pts = np.array(center_points_t, dtype=np.float64)
            xs, ys = pts[:, 0], pts[:, 1]

            deltas = np.diff(pts, axis=0)
            seg_lengths = np.sqrt((deltas ** 2).sum(axis=1))
            t = np.concatenate([[0], np.cumsum(seg_lengths)])

            if t[-1] > 0:
                t_norm = t / t[-1]
                coeffs_x = np.polyfit(t_norm, xs, degree)
                coeffs_y = np.polyfit(t_norm, ys, degree)
                poly_x = np.poly1d(coeffs_x)
                poly_y = np.poly1d(coeffs_y)

                t_samples = np.linspace(0, 1, 100)
                x_samples = poly_x(t_samples)
                y_samples = poly_y(t_samples)

                curve_pts = np.stack([x_samples, y_samples], axis=1)
                curve_pts = np.clip(curve_pts, [0, 0], [center_line.shape[1]-1, center_line.shape[0]-1])
                curve_pts = curve_pts.astype(np.int32).reshape(-1, 1, 2)

                cv2.polylines(center_line, [curve_pts], isClosed=False, color=200, thickness=1)

        for steering_point in center_points_t:
            cv2.circle(center_line, (int(steering_point[0]), int(steering_point[1])), radius=2, color=150, thickness=-1)

        cv2.circle(center_line, (int(point[0]), int(point[1])), radius=2, color=75, thickness=-1)
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
    dist = dists[8] - 10.0
    speed = calculate_max_target_speed(max(dist, 0.0), 4.5, 10.0)
    return speed#max(speeds) * 3.6