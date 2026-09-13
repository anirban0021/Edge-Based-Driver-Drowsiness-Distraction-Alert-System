from collections import deque
import time


class PerclosTracker:
    """
    Tracks eye-closed state over a rolling window of frames and flags
    microsleep risk based on either:
      - PERCLOS (% of frames closed in the window) exceeding a threshold, OR
      - Eyes being continuously closed for longer than a time threshold.
    """

    def __init__(self, window_size=90, perclos_threshold=20.0, continuous_threshold_sec=2.0):
        self.window = deque(maxlen=window_size)
        self.perclos_threshold = perclos_threshold
        self.continuous_threshold_sec = continuous_threshold_sec
        self._closed_since = None  # timestamp when eyes first became closed, or None

    def update(self, is_closed):
        """Call once per frame with the current open/closed boolean."""
        self.window.append(is_closed)

        now = time.time()
        if is_closed:
            if self._closed_since is None:
                self._closed_since = now
        else:
            self._closed_since = None

        return self._evaluate(now)

    def _evaluate(self, now):
        perclos = self.perclos()
        continuous_closed_time = (now - self._closed_since) if self._closed_since else 0.0

        microsleep = (
            perclos > self.perclos_threshold
            or continuous_closed_time > self.continuous_threshold_sec
        )

        return {
            "perclos": perclos,
            "continuous_closed_time": continuous_closed_time,
            "microsleep": microsleep,
        }

    def perclos(self):
        """Percentage of frames in the window where eyes were closed."""
        if not self.window:
            return 0.0
        return (sum(self.window) / len(self.window)) * 100.0