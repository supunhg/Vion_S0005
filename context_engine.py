from datetime import datetime
from config import (BASE_THRESHOLD, DAY_NIGHT_SWITCH,
                    NIGHT_DELTA, MULTIFACE_DELTA, UNKNOWN_DELTA)

class ContextEngine:
    def __init__(self):
        self.last_unknown_time = None

    def evaluate(self, faces_in_frame: int, unknown_present: bool) -> dict:
        hour = datetime.now().hour
        threshold = BASE_THRESHOLD

        # time-of-day
        if hour >= DAY_NIGHT_SWITCH or hour < 6:
            threshold += NIGHT_DELTA
            tod = "night"
        else:
            tod = "day"

        # multiface
        if faces_in_frame > 1:
            threshold += MULTIFACE_DELTA

        # unknown spotted
        if unknown_present:
            threshold += UNKNOWN_DELTA
            self.last_unknown_time = datetime.now()

        # simple risk level
        if threshold >= BASE_THRESHOLD + 0.10:
            risk = "high"
        elif threshold > BASE_THRESHOLD:
            risk = "elevated"
        else:
            risk = "low"

        return {"threshold": threshold, "time_of_day": tod, "risk": risk}
