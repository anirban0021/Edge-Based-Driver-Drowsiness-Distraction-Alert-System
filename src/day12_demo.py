import cv2
import time
from video_stream import VideoStream
from face_landmarks import FaceLandmarkDetector
from head_pose import HeadPoseEstimator, calibrate_head_pose
from distraction import DistractionTracker

stream = VideoStream(0)
detector = FaceLandmarkDetector()
pose_estimator = HeadPoseEstimator()

# --- Calibration step: measure this user's neutral "looking straight" pose ---
baseline_yaw, baseline_pitch = calibrate_head_pose(stream, detector, pose_estimator, duration=3.0)
print(f"Calibration complete. Baseline yaw: {baseline_yaw:.1f}, Baseline pitch: {baseline_pitch:.1f}")

distraction_tracker = DistractionTracker(yaw_threshold=25.0, pitch_threshold=18.0, sustain_duration=1.0)

signals = {}
prev_time = time.time()

while True:
    frame = stream.read()
    if frame is None:
        break

    landmarks = detector.process(frame)

    if landmarks is not None:
        points = detector.get_keypoints(landmarks, frame.shape)

        debug_points = [points["nose_tip"], points["chin"],
                         points["left_eye"][3], points["right_eye"][0],
                         points["mouth"]["left_corner"], points["mouth"]["right_corner"]]
        for (x, y) in debug_points:
            cv2.circle(frame, (int(x), int(y)), 4, (255, 0, 255), -1)

        pose = pose_estimator.estimate(points, frame.shape)

        if pose is not None:
            # Use deviation from this user's calibrated neutral pose, not raw angles
            relative_yaw = pose["yaw"] - baseline_yaw
            relative_pitch = pose["pitch"] - baseline_pitch

            result = distraction_tracker.update(relative_yaw, relative_pitch)
            signals["yaw_pitch"] = (relative_yaw, relative_pitch)
            signals["distracted"] = result["distracted"]

            color = (0, 0, 255) if result["distracted"] else (0, 255, 0)

            cv2.putText(frame, f"Yaw: {relative_yaw:.1f}  Pitch: {relative_pitch:.1f}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Away duration: {result['away_duration']:.1f}s",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if result["distracted"]:
                cv2.putText(frame, "DISTRACTED - LOOKING AWAY", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "No face detected", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Distraction Detection - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()