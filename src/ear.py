import math


def euclidean_distance(p1, p2):
    """Straight-line distance between two (x, y) points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def eye_aspect_ratio(eye_points):
    """
    Calculates EAR from 6 eye landmark points, ordered as:
    P1 = left corner, P2/P3 = top lid, P4 = right corner, P5/P6 = bottom lid
    (this matches the order LEFT_EYE / RIGHT_EYE are defined in face_landmarks.py)

    EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
    """
    A = euclidean_distance(eye_points[1], eye_points[5])
    B = euclidean_distance(eye_points[2], eye_points[4])
    C = euclidean_distance(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)