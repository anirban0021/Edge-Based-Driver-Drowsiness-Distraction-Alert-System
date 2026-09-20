import cv2
import time
import numpy as np
from video_stream import VideoStream
from face_landmarks import FaceLandmarkDetector
from head_pose import HeadPoseEstimator

stream = VideoStream(0)
detector = FaceLandmarkDetector()
pose_estimator = HeadPoseEstimator()

prev_time = time.time()

while True:
    frame = stream.read()
    if frame is None:
        break

    landmarks = detector.process(frame)

    if landmarks is not None:
        points = detector.get_keypoints(landmarks, frame.shape)
        pose = pose_estimator.estimate(points, frame.shape)

        if pose is not None:
            cv2.putText(frame, f"Pitch: {pose['pitch']:.1f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Yaw:   {pose['yaw']:.1f}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Roll:  {pose['roll']:.1f}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Draw a direction line from the nose tip showing where the head points
            nose_end_3d = np.array([(0.0, 0.0, 1000.0)])
            nose_end_2d, _ = cv2.projectPoints(
                nose_end_3d, pose["rvec"], pose["tvec"],
                pose["camera_matrix"], pose["dist_coeffs"]
            )
            p1 = (int(points["nose_tip"][0]), int(points["nose_tip"][1]))
            p2 = (int(nose_end_2d[0][0][0]), int(nose_end_2d[0][0][1]))
            cv2.line(frame, p1, p2, (255, 0, 0), 3)
    else:
        cv2.putText(frame, "No face detected", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Head Pose - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()