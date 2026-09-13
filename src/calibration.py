import cv2
import time


def calibrate_ear(stream, detector, ear_func, duration=5.0):
    """
    Runs for `duration` seconds, telling the user to keep their eyes open
    and look at the camera, then returns the average EAR measured during
    that window as the person's personal open-eye baseline.
    """
    print(f"Calibrating... keep your eyes open and look at the camera for {duration:.0f} seconds.")

    ear_readings = []
    start_time = time.time()

    while time.time() - start_time < duration:
        frame = stream.read()
        if frame is None:
            continue

        landmarks = detector.process(frame)
        if landmarks is not None:
            points = detector.get_keypoints(landmarks, frame.shape)
            left_ear = ear_func(points["left_eye"])
            right_ear = ear_func(points["right_eye"])
            avg_ear = (left_ear + right_ear) / 2.0
            ear_readings.append(avg_ear)

        remaining = duration - (time.time() - start_time)
        cv2.putText(frame, f"Calibrating... keep eyes open ({remaining:.1f}s)",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Calibration", frame)
        cv2.waitKey(1)

    cv2.destroyWindow("Calibration")

    if not ear_readings:
        raise RuntimeError("No face detected during calibration. Try again with better lighting/positioning.")

    baseline_ear = sum(ear_readings) / len(ear_readings)
    return baseline_ear


class SessionConfig:
    """Holds calibrated values for the current session."""
    def __init__(self, baseline_ear, closed_ratio=0.75):
        self.baseline_ear = baseline_ear
        self.closed_threshold = baseline_ear * closed_ratio

    def is_closed(self, current_ear):
        return current_ear < self.closed_threshold