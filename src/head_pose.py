import numpy as np
import cv2
import time

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
    def __init__(self, smoothing=0.7):
        self._camera_matrix = None
        self._dist_coeffs = np.zeros((4, 1))  # assume no lens distortion
        self._smoothing = smoothing
        self._smoothed_yaw = None
        self._smoothed_pitch = None
        self._smoothed_roll = None

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
        Returns (pitch, yaw, roll) in degrees (smoothed), plus rvec/tvec for drawing.
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
            flags=cv2.SOLVEPNP_EPNP
        )
        if not success:
            return None

        rmat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = self._rotation_matrix_to_euler(rmat)

        # Exponential smoothing to reduce single-frame jitter/spikes
        if self._smoothed_yaw is None:
            self._smoothed_yaw, self._smoothed_pitch, self._smoothed_roll = yaw, pitch, roll
        else:
            a = self._smoothing
            self._smoothed_yaw = a * self._smoothed_yaw + (1 - a) * yaw
            self._smoothed_pitch = a * self._smoothed_pitch + (1 - a) * pitch
            self._smoothed_roll = a * self._smoothed_roll + (1 - a) * roll

        return {
            "pitch": self._smoothed_pitch, "yaw": self._smoothed_yaw, "roll": self._smoothed_roll,
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


def calibrate_head_pose(stream, detector, pose_estimator, duration=3.0):
    """
    Runs for `duration` seconds, telling the user to look straight at the
    camera, then returns the average (yaw, pitch) as their personal neutral
    baseline — corrects for webcam position and face-shape offset.
    """
    print(f"Calibrating head pose... look straight at the camera for {duration:.0f} seconds.")

    yaw_readings = []
    pitch_readings = []
    start_time = time.time()

    while time.time() - start_time < duration:
        frame = stream.read()
        if frame is None:
            continue

        landmarks = detector.process(frame)
        if landmarks is not None:
            points = detector.get_keypoints(landmarks, frame.shape)
            pose = pose_estimator.estimate(points, frame.shape)
            if pose is not None:
                yaw_readings.append(pose["yaw"])
                pitch_readings.append(pose["pitch"])

        remaining = duration - (time.time() - start_time)
        cv2.putText(frame, f"Calibrating... look straight ({remaining:.1f}s)",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Head Pose Calibration", frame)
        cv2.waitKey(1)

    cv2.destroyWindow("Head Pose Calibration")

    if not yaw_readings:
        raise RuntimeError("No face detected during head pose calibration. Try again.")

    baseline_yaw = sum(yaw_readings) / len(yaw_readings)
    baseline_pitch = sum(pitch_readings) / len(pitch_readings)
    return baseline_yaw, baseline_pitch