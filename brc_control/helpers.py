import numpy as np
import cv2

def filter_outliers_along_path_simplified(points, distance_threshold=5.0):
    if not points:
        return []

    points_array = np.array(points)
    path_points = [np.asarray((int(218/2), int(218)))]#[points_array[0]]#[np.asarray((int(218/2), int(0)))]

    while True:
        distances = np.linalg.norm(points_array - path_points[-1], axis=1)
        min_distance_index = np.argmin(distances)

        if distances[min_distance_index] > distance_threshold:
            return path_points

        path_points.append(points_array[min_distance_index])
        points_array = np.delete(points_array, min_distance_index, axis=0)

        if not points_array.shape[0]:
            return path_points

def get_nearest_coordinates_at_distance(path_points, distance):
    if not path_points or distance <= 0:
        return None

    path_array = np.array(path_points)
    cumulative_distances = np.cumsum(np.linalg.norm(np.diff(path_array, axis=0), axis=1))

    if len(cumulative_distances) == 0 or len(path_points) == 0:
        return (0,0), 0

    if distance >= cumulative_distances[-1]:
        return path_points[-1], cumulative_distances[-1]

    index = np.searchsorted(cumulative_distances, distance)
    if index == 0:
        return path_points[0], cumulative_distances[0]

    nearest_point = path_array[index - 1] if distance - cumulative_distances[index - 1] < cumulative_distances[index] - distance else path_array[index]
    dist = cumulative_distances[index - 1] if distance - cumulative_distances[index - 1] < cumulative_distances[index] - distance else cumulative_distances[index]
    
    return tuple(nearest_point), dist

def calculate_angle(origin, point):
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]

    return np.degrees(np.arctan2(dx, -dy))

def steering_point_from_topdown(track, distance_threshold=20, distance=32):
    trackU = ((track).astype(np.uint8)*255)
    dist = cv2.distanceTransform(trackU, cv2.DIST_L2, 3)
    local_maxima_test = (cv2.compare(dist, cv2.dilate(dist, None), cv2.CMP_EQ) > 0)
    center_line = (trackU > 0) & local_maxima_test
    center_points = cv2.findNonZero(center_line.astype(np.uint8))
    center_points_t = [tuple(point[0]) for point in center_points]
    center_points_t = filter_outliers_along_path_simplified(center_points_t, distance_threshold=distance_threshold)
    point, point_dist = get_nearest_coordinates_at_distance(center_points_t, distance=distance)
    return point, point_dist