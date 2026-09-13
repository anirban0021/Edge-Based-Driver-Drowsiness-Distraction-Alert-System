import cv2
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
        result = tracker.update(is_closed)

        color = (0, 0, 255) if result["microsleep"] else (0, 255, 0)

        for (x, y) in points["left_eye"] + points["right_eye"]:
            cv2.circle(frame, (x, y), 2, color, -1)

        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"PERCLOS: {result['perclos']:.1f}%", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Eyes closed for: {result['continuous_closed_time']:.1f}s", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if result["microsleep"]:
            cv2.putText(frame, "MICROSLEEP DETECTED", (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            print(f"[ALERT] Microsleep flagged | PERCLOS={result['perclos']:.1f}% | continuous={result['continuous_closed_time']:.1f}s")
    else:
        cv2.putText(frame, "No face detected", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("PERCLOS - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()