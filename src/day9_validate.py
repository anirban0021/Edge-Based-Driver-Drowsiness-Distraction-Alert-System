import cv2
import csv
import time
from video_stream import VideoStream
from face_landmarks import FaceLandmarkDetector
from ear import eye_aspect_ratio
from calibration import calibrate_ear, SessionConfig
from perclos import PerclosTracker

stream = VideoStream(0)
detector = FaceLandmarkDetector()

baseline = calibrate_ear(stream, detector, eye_aspect_ratio, duration=5.0)
config = SessionConfig(baseline, closed_ratio=0.75)
tracker = PerclosTracker(window_size=90, perclos_threshold=20.0, continuous_threshold_sec=2.0)

print(f"Calibration complete. Baseline EAR: {baseline:.3f}, Closed threshold: {config.closed_threshold:.3f}")
print("Recording starts now. Follow the on-screen instructions.")

log_rows = []
start_time = time.time()
TEST_DURATION = 30.0  # seconds

while True:
    elapsed = time.time() - start_time
    if elapsed > TEST_DURATION:
        break

    frame = stream.read()
    if frame is None:
        continue

    landmarks = detector.process(frame)

    if landmarks is not None:
        points = detector.get_keypoints(landmarks, frame.shape)
        left_ear = eye_aspect_ratio(points["left_eye"])
        right_ear = eye_aspect_ratio(points["right_eye"])
        avg_ear = (left_ear + right_ear) / 2.0
        is_closed = config.is_closed(avg_ear)
        result = tracker.update(is_closed)

        log_rows.append({
            "elapsed_sec": round(elapsed, 2),
            "ear": round(avg_ear, 4),
            "is_closed": is_closed,
            "perclos": round(result["perclos"], 2),
            "continuous_closed_sec": round(result["continuous_closed_time"], 2),
            "microsleep_flag": result["microsleep"],
        })

        color = (0, 0, 255) if result["microsleep"] else (0, 255, 0)
        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"PERCLOS: {result['perclos']:.1f}%", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if elapsed < 10:
        instruction = "0-10s: Look at camera normally, eyes OPEN"
    elif elapsed < 20:
        instruction = "10-20s: Blink NORMALLY (quick, natural blinks)"
    else:
        instruction = "20-30s: CLOSE your eyes and hold 2+ seconds, twice"

    cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Time: {elapsed:.1f}s / {TEST_DURATION:.0f}s", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Day 9 Validation - auto-closes at 30s", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()

with open("logs/day9_validation.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
    writer.writeheader()
    writer.writerows(log_rows)

print("Saved results to logs/day9_validation.csv")