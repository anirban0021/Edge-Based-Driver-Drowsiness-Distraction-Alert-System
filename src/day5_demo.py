import cv2
import time
from video_stream import VideoStream
from face_landmarks import FaceLandmarkDetector

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

        for (x, y) in points["left_eye"] + points["right_eye"]:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)      # eyes = green
        for (x, y) in points["mouth"].values():
            cv2.circle(frame, (x, y), 2, (0, 165, 255), -1)    # mouth = orange
        cv2.circle(frame, points["nose_tip"], 3, (255, 0, 0), -1)   # nose = blue
        cv2.circle(frame, points["chin"], 3, (255, 0, 255), -1)     # chin = pink
    else:
        cv2.putText(frame, "No face detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Mesh - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()