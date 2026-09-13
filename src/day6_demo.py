import cv2
import time
from video_stream import VideoStream
from face_landmarks import FaceLandmarkDetector
from ear import eye_aspect_ratio

stream = VideoStream(0)
detector = FaceLandmarkDetector()
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

        # Draw eye landmarks for visual reference
        for (x, y) in points["left_eye"] + points["right_eye"]:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "No face detected", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("EAR - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()