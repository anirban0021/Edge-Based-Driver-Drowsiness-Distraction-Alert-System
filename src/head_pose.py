import numpy as np
import cv2

# Generic 3D face model (arbitrary units, not real mm measurements — only
# relative proportions matter for solvePnP to work).
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye - outer corner
    (225.0, 170.0, -135.0),   # Right eye - outer corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype="double")


class HeadPoseEstimator:
    def __init__(self):
        self._camera_matrix = None
        self._dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

    def _get_camera_matrix(self, frame_shape):
        if self._camera_matrix is None:
            h, w = frame_shape[:2]
            focal_length = w
            center = (w / 2, h / 2)
            self._camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
        return self._camera_matrix

    def estimate(self, points, frame_shape):
        """
        points: the dict returned by FaceLandmarkDetector.get_keypoints()
        Returns (pitch, yaw, roll) in degrees, plus rvec/tvec for drawing.
        """
        image_points = np.array([
            points["nose_tip"],
            points["chin"],
            points["left_eye"][3],   # outer corner of left eye
            points["right_eye"][0],  # outer corner of right eye
            points["mouth"]["left_corner"],
            points["mouth"]["right_corner"],
        ], dtype="double")

        camera_matrix = self._get_camera_matrix(frame_shape)

        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS, image_points, camera_matrix, self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return None

        rmat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = self._rotation_matrix_to_euler(rmat)

        return {
            "pitch": pitch, "yaw": yaw, "roll": roll,
            "rvec": rvec, "tvec": tvec, "camera_matrix": camera_matrix,
            "dist_coeffs": self._dist_coeffs,
        }

    @staticmethod
    def _rotation_matrix_to_euler(R):
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return np.degrees(x), np.degrees(y), np.degrees(z)  # pitch, yaw, roll