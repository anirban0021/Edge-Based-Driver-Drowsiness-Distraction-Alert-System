import cv2
import time


class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {src}. "
                f"Try a different index (0, 1, 2) or check nothing else is using the webcam."
            )

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


if __name__ == "__main__":
    try:
        stream = VideoStream(0)
    except RuntimeError as e:
        print(e)
        exit(1)

    prev_time = time.time()

    while True:
        frame = stream.read()
        if frame is None:
            print("Failed to grab frame")
            break

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time

        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        cv2.imshow("Video Stream - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()