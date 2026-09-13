import cv2
import time
from video_stream import VideoStream
from face_landmarks import FaceLandmarkDetector
from ear import eye_aspect_ratio
from calibration import calibrate_ear, SessionConfig

stream = VideoStream(0)
detector = FaceLandmarkDetector()

# --- Calibration step ---
baseline = calibrate_ear(stream, detector, eye_aspect_ratio, duration=5.0)
config = SessionConfig(baseline, closed_ratio=0.75)

print(f"Calibration complete.")
print(f"Baseline (open) EAR: {baseline:.3f}")
print(f"Closed-eye threshold: {config.closed_threshold:.3f}")

# --- Live loop using the calibrated threshold ---
prev_time = time.time()

while True:
    frame = stream.read()
    if frame is None:
        break

    landmarks = detector.process(frame)

    if landmarks is not None:
        points = detector.get_keypoints(landmarks, frame.shape)
        left_ear = eye_aspect_ratio(points["left_eye"])
        right_ear = eye_aspect_ratio(points["right_eye"])
        avg_ear = (left_ear + right_ear) / 2.0

        is_closed = config.is_closed(avg_ear)
        status = "CLOSED" if is_closed else "OPEN"
        color = (0, 0, 255) if is_closed else (0, 255, 0)

        for (x, y) in points["left_eye"] + points["right_eye"]:
            cv2.circle(frame, (x, y), 2, color, -1)

        cv2.putText(frame, f"EAR: {avg_ear:.3f}  [{status}]", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Threshold: {config.closed_threshold:.3f}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    else:
        cv2.putText(frame, "No face detected", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Calibrated EAR - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()