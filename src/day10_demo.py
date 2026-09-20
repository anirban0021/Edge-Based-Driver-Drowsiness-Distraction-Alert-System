import cv2
import time
from video_stream import VideoStream
from face_landmarks import FaceLandmarkDetector
from mor import mouth_opening_ratio
from yawn_tracker import YawnTracker

stream = VideoStream(0)
detector = FaceLandmarkDetector()
tracker = YawnTracker(mor_threshold=0.6, min_yawn_duration=1.5)

prev_time = time.time()

while True:
    frame = stream.read()
    if frame is None:
        break

    landmarks = detector.process(frame)

    if landmarks is not None:
        points = detector.get_keypoints(landmarks, frame.shape)
        mor = mouth_opening_ratio(points["mouth"])
        result = tracker.update(mor)

        color = (0, 0, 255) if result["yawning"] else (0, 255, 0)

        for (x, y) in points["mouth"].values():
            cv2.circle(frame, (x, y), 3, color, -1)

        cv2.putText(frame, f"MOR: {mor:.3f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Yawns: {result['total_yawns']}  ({result['yawns_per_minute']}/min)",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if result["yawning"]:
            cv2.putText(frame, "YAWN DETECTED", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "No face detected", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Yawn Detection - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()