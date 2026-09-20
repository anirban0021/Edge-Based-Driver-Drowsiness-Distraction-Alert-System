from ear import euclidean_distance  # reuse the distance helper from Day 6


def mouth_opening_ratio(mouth_points):
    """
    mouth_points is the dict from face_landmarks.get_keypoints()["mouth"]:
    {"top_lip": (x,y), "bottom_lip": (x,y), "left_corner": (x,y), "right_corner": (x,y)}

    MOR = vertical lip gap / horizontal mouth width
    """
    vertical = euclidean_distance(mouth_points["top_lip"], mouth_points["bottom_lip"])
    horizontal = euclidean_distance(mouth_points["left_corner"], mouth_points["right_corner"])
    return vertical / horizontal