from collections import deque
import time


class YawnTracker:
    """
    Counts a yawn only when MOR stays above threshold continuously for at
    least `min_yawn_duration` seconds. Also tracks a rolling yawns-per-minute rate.
    """

    def __init__(self, mor_threshold=0.6, min_yawn_duration=1.5, yawn_window_sec=60):
        self.mor_threshold = mor_threshold
        self.min_yawn_duration = min_yawn_duration
        self.yawn_window_sec = yawn_window_sec

        self._mouth_open_since = None
        self._counted_this_open = False
        self.total_yawns = 0
        self._yawn_timestamps = deque()

    def update(self, mor):
        now = time.time()
        is_open = mor > self.mor_threshold
        yawning_now = False
        open_duration = 0.0

        if is_open:
            if self._mouth_open_since is None:
                self._mouth_open_since = now
                self._counted_this_open = False

            open_duration = now - self._mouth_open_since

            if open_duration >= self.min_yawn_duration:
                yawning_now = True
                if not self._counted_this_open:
                    self.total_yawns += 1
                    self._counted_this_open = True
                    self._yawn_timestamps.append(now)
        else:
            self._mouth_open_since = None
            self._counted_this_open = False

        # Drop yawn timestamps older than the rolling window
        while self._yawn_timestamps and now - self._yawn_timestamps[0] > self.yawn_window_sec:
            self._yawn_timestamps.popleft()

        yawns_per_minute = len(self._yawn_timestamps) * (60.0 / self.yawn_window_sec)

        return {
            "mouth_open_duration": round(open_duration, 2),
            "yawning": yawning_now,
            "total_yawns": self.total_yawns,
            "yawns_per_minute": round(yawns_per_minute, 1),
        }