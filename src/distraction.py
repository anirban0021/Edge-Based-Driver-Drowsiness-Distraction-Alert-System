import time


class DistractionTracker:
    """
    Flags 'looking away' distraction when |yaw| or |pitch| exceeds a threshold
    continuously for at least `sustain_duration` seconds. A brief glance that
    doesn't hold past that duration is ignored.
    """

    def __init__(self, yaw_threshold=25.0, pitch_threshold=25.0, sustain_duration=1.0):
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.sustain_duration = sustain_duration
        self._away_since = None

    def update(self, yaw, pitch):
        now = time.time()
        looking_away_now = abs(yaw) > self.yaw_threshold or abs(pitch) > self.pitch_threshold

        if looking_away_now:
            if self._away_since is None:
                self._away_since = now
        else:
            self._away_since = None

        away_duration = (now - self._away_since) if self._away_since else 0.0
        distracted = away_duration >= self.sustain_duration

        return {
            "looking_away_now": looking_away_now,
            "away_duration": round(away_duration, 2),
            "distracted": distracted,
        }