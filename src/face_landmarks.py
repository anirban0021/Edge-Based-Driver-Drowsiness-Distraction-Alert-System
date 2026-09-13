import mediapipe as mp

# Landmark indices we care about (out of MediaPipe's 468 total points)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = {"top_lip": 13, "bottom_lip": 14, "left_corner": 61, "right_corner": 291}
NOSE_TIP = 1
CHIN = 152


class FaceLandmarkDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame):
        """Returns MediaPipe's raw landmark result, or None if no face found."""
        rgb_frame = frame[:, :, ::-1]  # OpenCV uses BGR, MediaPipe needs RGB
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0]

    def get_keypoints(self, landmarks, frame_shape):
        """Converts normalized landmark coordinates to pixel coordinates
        for just the indices we need (eyes, mouth, nose, chin)."""
        h, w = frame_shape[:2]

        def to_px(idx):
            lm = landmarks.landmark[idx]
            return int(lm.x * w), int(lm.y * h)

        return {
            "left_eye": [to_px(i) for i in LEFT_EYE],
            "right_eye": [to_px(i) for i in RIGHT_EYE],
            "mouth": {name: to_px(idx) for name, idx in MOUTH.items()},
            "nose_tip": to_px(NOSE_TIP),
            "chin": to_px(CHIN),
        }