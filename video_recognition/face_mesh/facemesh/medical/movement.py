"""Facial movement / expressivity analysis (hypomimia-style).

Quantifies how much the face *moves* over a rolling window, after removing
rigid head motion. Reduced facial movement ("masked face", hypomimia) is a
hallmark of Parkinson's disease; facial expression amplitude and tremor of
small muscle groups are the features used in the literature.

Three signals, all in the normalized face frame:

- **amplitude** — how far expressive landmarks range over the window.
- **jitter** — mean frame-to-frame displacement (small, fast muscle motion).
- **tremor** — spectral power in the 4-6 Hz Parkinsonian band, as a fraction of
  total motion power. A scalar jitter value cannot distinguish tremor from
  ordinary movement; the spectrum can.

**Context matters.** A resting face is legitimately still, so passive
observation cannot distinguish "calm" from "hypomimic". Expressivity is only
meaningful relative to a *task*: use :meth:`MovementAnalyzer.set_task` to label
the current prompt ("smile", "rest"), and read
:meth:`MovementAnalyzer.task_summary` for per-task results. Metrics recorded
without a task are reported with ``task=None`` and should not be interpreted as
clinical findings.

SCREENING / EDUCATIONAL USE ONLY — not a diagnostic device.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from .geometry import normalize

# Expressive landmarks: brows, eyelids, mouth, nasolabial region.
EXPRESSIVE_IDX = (
    # brows
    46, 52, 53, 55, 63, 65, 66, 70, 105, 107,
    276, 282, 283, 285, 293, 295, 296, 300, 334, 336,
    # eyelids
    159, 145, 386, 374,
    # mouth
    61, 291, 13, 14, 78, 308, 40, 270, 91, 321, 84, 314,
    # nasolabial / cheeks
    50, 280, 205, 425,
)

# Parkinsonian rest-tremor band (Hz).
TREMOR_LO, TREMOR_HI = 4.0, 6.0


def tremor_power(series: np.ndarray, fps: float) -> tuple[float, float]:
    """Fraction of motion power in the 4-6 Hz band, and its peak frequency.

    Args:
        series: 1D displacement magnitudes over time.
        fps: sample rate.

    Returns ``(ratio, peak_hz)``; ratio is 0-1.
    """
    n = len(series)
    if n < 32 or fps <= 0:
        return 0.0, 0.0
    x = (series - series.mean()) * np.hanning(n)
    spec = np.abs(np.fft.rfft(x, n=max(n, 256))) ** 2
    freqs = np.fft.rfftfreq(max(n, 256), d=1.0 / fps)

    valid = (freqs > 0.5) & (freqs <= min(fps / 2, 15.0))
    if not valid.any() or spec[valid].sum() <= 0:
        return 0.0, 0.0
    band = valid & (freqs >= TREMOR_LO) & (freqs <= TREMOR_HI)
    if not band.any():
        return 0.0, 0.0
    ratio = float(spec[band].sum() / spec[valid].sum())
    peak_hz = float(freqs[band][np.argmax(spec[band])])
    return ratio, peak_hz


class MovementAnalyzer:
    """Rolling-window facial expressivity metrics for one face.

    Args:
        window_s: Length of the analysis window in seconds.
        low_amplitude: Amplitude below this flags reduced expressivity.
    """

    def __init__(self, window_s: float = 5.0, low_amplitude: float = 0.012) -> None:
        self.window_s = window_s
        self.low_amplitude = low_amplitude
        self._samples: deque[tuple[float, np.ndarray]] = deque()
        self._prev: np.ndarray | None = None
        self._jitter: deque[tuple[float, float]] = deque()
        self.task: str | None = None
        self._by_task: dict[str, list[dict]] = defaultdict(list)

    # --- task context ----------------------------------------------------
    def set_task(self, task: str | None) -> None:
        """Label the current prompt (e.g. ``"smile"``). ``None`` clears it."""
        if task != self.task:
            self.reset_window()
        self.task = task

    def reset_window(self) -> None:
        self._samples.clear()
        self._jitter.clear()
        self._prev = None

    def task_summary(self) -> dict:
        """Per-task peak expressivity and mean tremor across the session."""
        out = {}
        for task, records in self._by_task.items():
            if not records:
                continue
            out[task] = {
                "peak_expressivity": round(max(r["expressivity"] for r in records), 3),
                "peak_amplitude": round(max(r["amplitude"] for r in records), 5),
                "mean_tremor_ratio": round(
                    float(np.mean([r["tremor_ratio"] for r in records])), 3),
                "frames": len(records),
            }
        return out

    # --- per-frame update -------------------------------------------------
    def update(self, landmarks, width: int, height: int, now_s: float) -> dict:
        N = normalize(landmarks, width, height)
        pts = N[list(EXPRESSIVE_IDX)]

        if self._prev is not None:
            j = float(np.linalg.norm(pts - self._prev, axis=1).mean())
            self._jitter.append((now_s, j))
        self._prev = pts

        self._samples.append((now_s, pts))
        cutoff = now_s - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        while self._jitter and self._jitter[0][0] < cutoff:
            self._jitter.popleft()

        if len(self._samples) >= 2:
            stack = np.stack([s[1] for s in self._samples])       # (T, K, 2)
            amplitude = float(stack.std(axis=0).mean())
        else:
            amplitude = 0.0

        jitter = float(np.mean([j for _, j in self._jitter])) if self._jitter else 0.0

        # Tremor spectrum from the SIGNED centroid position series.
        # NOTE: it must not be computed from the jitter magnitude - rectifying a
        # sinusoid doubles its frequency and would move a 5 Hz tremor to 10 Hz,
        # outside the band we are looking for.
        tremor_ratio, tremor_hz = 0.0, 0.0
        if len(self._samples) >= 32:
            ts = np.array([t for t, _ in self._samples])
            stack = np.stack([s[1] for s in self._samples])       # (T, K, 2)
            centroid = stack.mean(axis=1)                          # (T, 2)
            span = ts[-1] - ts[0]
            fps = (len(ts) - 1) / span if span > 1e-6 else 0.0
            # Evaluate both axes; tremor may be dominantly vertical or horizontal.
            rx, hx = tremor_power(centroid[:, 0], fps)
            ry, hy = tremor_power(centroid[:, 1], fps)
            tremor_ratio, tremor_hz = (rx, hx) if rx >= ry else (ry, hy)

        expressivity = float(np.clip(amplitude / (self.low_amplitude * 3), 0.0, 1.0))
        reduced = bool(len(self._samples) > 5 and amplitude < self.low_amplitude)

        result = {
            "amplitude": round(amplitude, 5),
            "jitter": round(jitter, 5),
            "expressivity": round(expressivity, 3),
            "reduced_expressivity": reduced,
            "tremor_ratio": round(tremor_ratio, 3),
            "tremor_hz": round(tremor_hz, 2),
            "tremor_detected": bool(tremor_ratio > 0.35),
            "task": self.task,
            # Without a task label, expressivity is not clinically interpretable.
            "interpretable": self.task is not None,
        }
        if self.task is not None:
            self._by_task[self.task].append(result)
        return result
