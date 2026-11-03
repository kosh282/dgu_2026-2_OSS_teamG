"""Drowsiness monitoring from eye/mouth signals.

Combines three classic indicators over a rolling time window:

- **PERCLOS** — the fraction of time the eyes are closed. The single most
  validated drowsiness measure; sustained values above ~15% indicate fatigue.
- **Microsleep** — a single eye closure longer than ~1 s.
- **Yawning** — a sustained high Mouth Aspect Ratio.

Everything is time-based (not frame-based) so it behaves the same regardless
of frame rate.
"""

from __future__ import annotations

from collections import deque


class DrowsinessMonitor:
    """Track PERCLOS, microsleeps and yawns, and raise a drowsiness level.

    Args:
        ear_threshold: EAR below this means the eye is closed.
        mar_threshold: MAR above this means the mouth is wide open (yawn).
        window_s: Rolling window for PERCLOS and yawn-rate.
        perclos_alert: PERCLOS above this flags drowsiness.
        microsleep_s: Continuous closure beyond this flags a microsleep.
        yawn_min_s: A mouth-open episode must last this long to count as a yawn.
    """

    def __init__(
        self,
        ear_threshold: float = 0.21,
        mar_threshold: float = 0.6,
        window_s: float = 60.0,
        perclos_alert: float = 0.15,
        microsleep_s: float = 1.0,
        yawn_min_s: float = 0.4,
    ) -> None:
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.window_s = window_s
        self.perclos_alert = perclos_alert
        self.microsleep_s = microsleep_s
        self.yawn_min_s = yawn_min_s

        self._samples: deque[tuple[float, bool]] = deque()  # (t, eye_closed)
        self._closure_start: float | None = None
        self._yawn_start: float | None = None
        self.yawns = 0
        self.microsleeps = 0

    def update(self, ear: float, mar: float, now_s: float) -> dict:
        eye_closed = ear < self.ear_threshold

        # --- PERCLOS over the rolling window ---
        self._samples.append((now_s, eye_closed))
        cutoff = now_s - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        closed = sum(1 for _, c in self._samples if c)
        perclos = closed / len(self._samples) if self._samples else 0.0

        # --- Continuous closure / microsleep ---
        if eye_closed:
            if self._closure_start is None:
                self._closure_start = now_s
            closure_s = now_s - self._closure_start
        else:
            if self._closure_start is not None and \
                    (now_s - self._closure_start) >= self.microsleep_s:
                self.microsleeps += 1
            self._closure_start = None
            closure_s = 0.0
        microsleep = closure_s >= self.microsleep_s

        # --- Yawn detection (sustained mouth-open episode) ---
        yawning = False
        if mar > self.mar_threshold:
            if self._yawn_start is None:
                self._yawn_start = now_s
            if (now_s - self._yawn_start) >= self.yawn_min_s:
                yawning = True
        else:
            if self._yawn_start is not None and \
                    (now_s - self._yawn_start) >= self.yawn_min_s:
                self.yawns += 1
            self._yawn_start = None

        # --- Overall level ---
        if microsleep or perclos >= self.perclos_alert * 2:
            level = "ALERT"
        elif perclos >= self.perclos_alert or yawning:
            level = "drowsy"
        else:
            level = "awake"

        return {
            "perclos": round(perclos, 3),
            "closure_s": round(closure_s, 2),
            "microsleep": microsleep,
            "microsleeps": self.microsleeps,
            "yawning": yawning,
            "yawns": self.yawns,
            "level": level,
        }
