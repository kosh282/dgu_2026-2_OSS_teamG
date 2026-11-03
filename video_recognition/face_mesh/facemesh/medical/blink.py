"""Blink analytics as a clinical-style biomarker.

Ordinary blink *counting* answers "how many"; clinical interest is in the
*shape* of blinking:

- **rate** (blinks/min) — reduced in Parkinson's disease, elevated in some
  anxiety and tic disorders, and it drops sharply during screen use.
- **duration** — how long the eye stays closed per blink.
- **completeness** — whether the lid fully closes. **Incomplete blinks** leave
  the ocular surface exposed and are strongly associated with dry-eye disease.
- **interblink interval** — the gap between blinks, and its variability.
- **per-eye asymmetry** — one eye blinking less completely than the other is a
  sign of unilateral weakness (facial palsy: incomplete closure = lagophthalmos).

Every blink is measured from the Eye Aspect Ratio (EAR) waveform, so this
module needs no new landmarks and no new model.

SCREENING / EDUCATIONAL USE ONLY — not a diagnostic device.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..metrics import EYE_LANDMARKS_LEFT, EYE_LANDMARKS_RIGHT, eye_aspect_ratio

# A blink whose EAR never falls below this fraction of the open baseline is
# considered incomplete (the lid did not fully meet).
COMPLETE_CLOSURE_RATIO = 0.35

# Plausible physiological bounds; anything outside is an artifact, not a blink.
MIN_BLINK_S = 0.05
MAX_BLINK_S = 0.60


@dataclass
class Blink:
    """One detected blink."""

    start_s: float
    end_s: float
    min_ear: float
    baseline_ear: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def closure_ratio(self) -> float:
        """How far the eye closed: 0 = fully shut, 1 = never moved."""
        if self.baseline_ear <= 0:
            return 1.0
        return float(self.min_ear / self.baseline_ear)

    @property
    def complete(self) -> bool:
        return self.closure_ratio <= COMPLETE_CLOSURE_RATIO

    def to_dict(self) -> dict:
        return {
            "start_s": round(self.start_s, 2),
            "duration_s": round(self.duration_s, 3),
            "min_ear": round(self.min_ear, 4),
            "closure_ratio": round(self.closure_ratio, 3),
            "complete": self.complete,
        }


class EyeBlinkDetector:
    """Detect blinks in one eye's EAR waveform and describe each one.

    The open-eye baseline adapts (rolling median of open frames), so the
    detector works across people and lighting without a fixed EAR threshold.
    """

    def __init__(self, threshold_ratio: float = 0.75, baseline_window: int = 150) -> None:
        self.threshold_ratio = threshold_ratio
        self._open_ears: deque[float] = deque(maxlen=baseline_window)
        self._in_blink = False
        self._start: float | None = None
        self._min_ear = 1.0
        self.blinks: list[Blink] = []

    @property
    def baseline(self) -> float:
        return float(np.median(self._open_ears)) if self._open_ears else 0.0

    def update(self, ear: float, now_s: float) -> Blink | None:
        """Feed one EAR sample. Returns a :class:`Blink` on the frame it ends."""
        base = self.baseline
        if base <= 0:                       # still learning the open baseline
            self._open_ears.append(ear)
            return None

        closing = ear < base * self.threshold_ratio
        finished: Blink | None = None

        if closing:
            if not self._in_blink:
                self._in_blink = True
                self._start = now_s
                self._min_ear = ear
            else:
                self._min_ear = min(self._min_ear, ear)
        else:
            if self._in_blink and self._start is not None:
                dur = now_s - self._start
                if MIN_BLINK_S <= dur <= MAX_BLINK_S:
                    finished = Blink(self._start, now_s, self._min_ear, base)
                    self.blinks.append(finished)
                self._in_blink = False
                self._start = None
                self._min_ear = 1.0
            self._open_ears.append(ear)     # only open frames shape the baseline

        return finished

    @property
    def is_closed(self) -> bool:
        return self._in_blink


class BlinkAnalytics:
    """Per-eye blink biomarkers over a rolling window.

    Args:
        window_s: Rolling window for rate and interval statistics.
    """

    def __init__(self, window_s: float = 60.0) -> None:
        self.window_s = window_s
        self.left = EyeBlinkDetector()
        self.right = EyeBlinkDetector()
        self._events: list[tuple[float, Blink, Blink | None]] = []

    def _recent(self, now_s: float) -> list[tuple[float, Blink, Blink | None]]:
        cutoff = now_s - self.window_s
        self._events = [e for e in self._events if e[0] >= cutoff]
        return self._events

    def update(self, landmarks, width: int, height: int, now_s: float) -> dict:
        ear_l = eye_aspect_ratio(landmarks, EYE_LANDMARKS_LEFT, width, height)
        ear_r = eye_aspect_ratio(landmarks, EYE_LANDMARKS_RIGHT, width, height)
        bl = self.left.update(ear_l, now_s)
        br = self.right.update(ear_r, now_s)

        # A blink event is registered when either eye completes one.
        if bl is not None or br is not None:
            self._events.append((now_s, bl or br, br if bl is not None else None))

        recent = self._recent(now_s)
        elapsed = min(now_s, self.window_s) or 1e-9
        rate = len(recent) * 60.0 / elapsed

        durations = [b.duration_s for _, b, _ in recent]
        completeness = [b.closure_ratio for _, b, _ in recent]
        incomplete = sum(1 for c in completeness if c > COMPLETE_CLOSURE_RATIO)

        starts = [t for t, _, _ in recent]
        intervals = np.diff(starts) if len(starts) > 1 else np.array([])

        out = {
            "ear_left": round(ear_l, 4),
            "ear_right": round(ear_r, 4),
            "blinks_left": len(self.left.blinks),
            "blinks_right": len(self.right.blinks),
            "rate_per_min": round(rate, 1),
            "window_blinks": len(recent),
            "mean_duration_s": round(float(np.mean(durations)), 3) if durations else 0.0,
            "mean_closure_ratio": round(float(np.mean(completeness)), 3) if completeness else 0.0,
            "incomplete_blinks": incomplete,
            "incomplete_fraction": round(incomplete / len(recent), 3) if recent else 0.0,
            "mean_interblink_s": round(float(np.mean(intervals)), 2) if len(intervals) else 0.0,
            "interblink_sd_s": round(float(np.std(intervals)), 2) if len(intervals) > 1 else 0.0,
        }

        # Per-eye closure asymmetry: unilateral weakness shows up here.
        lc = [b.closure_ratio for b in self.left.blinks[-20:]]
        rc = [b.closure_ratio for b in self.right.blinks[-20:]]
        if lc and rc:
            out["closure_asymmetry"] = round(abs(float(np.mean(lc)) - float(np.mean(rc))), 3)
            out["weaker_eye"] = "left" if np.mean(lc) > np.mean(rc) else "right"
        else:
            out["closure_asymmetry"] = 0.0
            out["weaker_eye"] = ""

        # Flags. Normal spontaneous rate is roughly 10-25/min at rest.
        out["low_rate"] = bool(len(recent) >= 3 and rate < 8.0)
        out["high_incomplete"] = bool(len(recent) >= 5 and out["incomplete_fraction"] > 0.4)
        return out

    def all_blinks(self) -> dict:
        return {
            "left": [b.to_dict() for b in self.left.blinks],
            "right": [b.to_dict() for b in self.right.blinks],
        }
