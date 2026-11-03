"""Pupil and gaze dynamics.

Eye-movement measures are among the most sensitive behavioural markers in
neurology: saccade velocity, fixation stability and nystagmus all change in
Parkinson's disease, progressive supranuclear palsy, concussion and vestibular
disorders. This module derives them from the iris landmarks the mesh already
provides — no eye tracker required.

What is measured, per eye and combined:

- **gaze position** — normalized iris centre within the eye opening.
- **saccades** — fast gaze shifts, with peak angular velocity and amplitude.
- **fixation stability** — spread of gaze during non-saccadic periods (BCEA-like).
- **nystagmus screen** — a dominant 2-10 Hz oscillation in horizontal gaze.
- **pupil proxy** — relative iris diameter, in inter-ocular units. This is a
  *proxy*, not a millimetre pupil size; the iris ring is what MediaPipe gives.

.. warning::
   Iris landmarks are low-resolution. Absolute pupil size is **not** recoverable
   here; only relative change is meaningful, and only under constant lighting
   and distance. Saccade velocities are limited by frame rate: at 30 fps most
   real saccades (30-500 deg/s, 20-80 ms) are undersampled, so treat velocity
   as a *relative* indicator, never an absolute clinical value.

SCREENING / EDUCATIONAL USE ONLY — not a diagnostic device.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..metrics import (
    LEFT_EYE_CORNERS,
    LEFT_EYE_LIDS,
    LEFT_IRIS,
    RIGHT_EYE_CORNERS,
    RIGHT_EYE_LIDS,
    RIGHT_IRIS,
)

# Iris centre landmarks (the rings are 469-472 / 474-477).
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468

# A gaze shift faster than this (normalized units/s) counts as a saccade.
SACCADE_SPEED = 1.2
# Nystagmus band (Hz).
NYSTAGMUS_LO, NYSTAGMUS_HI = 2.0, 10.0


def _iris_center(landmarks, ring) -> np.ndarray:
    xs = [landmarks[i].x for i in ring]
    ys = [landmarks[i].y for i in ring]
    return np.array([sum(xs) / len(xs), sum(ys) / len(ys)])


def iris_diameter(landmarks, ring, width: int, height: int, iod: float) -> float:
    """Iris diameter in inter-ocular units (a *relative* pupil-size proxy)."""
    pts = np.array([[landmarks[i].x * width, landmarks[i].y * height] for i in ring])
    center = pts.mean(axis=0)
    radius = np.linalg.norm(pts - center, axis=1).mean()
    return float(2 * radius / iod) if iod > 0 else 0.0


def eye_gaze(landmarks, ring, corners, lids) -> tuple[float, float]:
    """Normalized iris position within one eye: ``(h, v)``, 0.5 = centred."""
    cx, cy = _iris_center(landmarks, ring)
    x0, x1 = landmarks[corners[0]].x, landmarks[corners[1]].x
    yt, yb = landmarks[lids[0]].y, landmarks[lids[1]].y
    lo, hi = min(x0, x1), max(x0, x1)
    h = (cx - lo) / (hi - lo) if hi > lo else 0.5
    v = (cy - yt) / (yb - yt) if yb > yt else 0.5
    return float(h), float(v)


def detect_oscillation(series: np.ndarray, fps: float,
                       lo_hz: float = NYSTAGMUS_LO,
                       hi_hz: float = NYSTAGMUS_HI) -> tuple[float, float]:
    """Fraction of power in ``[lo_hz, hi_hz]`` and its peak frequency."""
    n = len(series)
    if n < 32 or fps <= 0:
        return 0.0, 0.0
    x = (series - series.mean()) * np.hanning(n)
    spec = np.abs(np.fft.rfft(x, n=max(n, 256))) ** 2
    freqs = np.fft.rfftfreq(max(n, 256), d=1.0 / fps)
    valid = (freqs > 0.3) & (freqs <= min(fps / 2, 20.0))
    if not valid.any() or spec[valid].sum() <= 0:
        return 0.0, 0.0
    band = valid & (freqs >= lo_hz) & (freqs <= hi_hz)
    if not band.any():
        return 0.0, 0.0
    ratio = float(spec[band].sum() / spec[valid].sum())
    peak = float(freqs[band][np.argmax(spec[band])])
    return ratio, peak


class GazeDynamics:
    """Saccades, fixation stability, nystagmus screen and pupil proxy.

    Args:
        window_s: Rolling window for statistics.
        saccade_speed: Gaze speed (normalized units/s) above which a shift
            counts as a saccade.
    """

    def __init__(self, window_s: float = 5.0, saccade_speed: float = SACCADE_SPEED) -> None:
        self.window_s = window_s
        self.saccade_speed = saccade_speed
        self._samples: deque[tuple[float, float, float, float]] = deque()  # t,h,v,speed
        self._prev: tuple[float, float, float] | None = None
        self._in_saccade = False
        self._sacc_start: tuple[float, float, float] | None = None
        self._peak_speed = 0.0
        self.saccades: list[dict] = []

    def _push(self, now_s: float, h: float, v: float, speed: float) -> None:
        self._samples.append((now_s, h, v, speed))
        cutoff = now_s - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def update(self, landmarks, width: int, height: int, now_s: float) -> dict:
        hl, vl = eye_gaze(landmarks, LEFT_IRIS, LEFT_EYE_CORNERS, LEFT_EYE_LIDS)
        hr, vr = eye_gaze(landmarks, RIGHT_IRIS, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS)
        h, v = (hl + hr) / 2.0, (vl + vr) / 2.0

        # Pupil proxy (relative iris diameter).
        pr = np.array([landmarks[33].x * width, landmarks[33].y * height])
        pl = np.array([landmarks[263].x * width, landmarks[263].y * height])
        iod = float(np.linalg.norm(pr - pl))
        d_l = iris_diameter(landmarks, LEFT_IRIS, width, height, iod)
        d_r = iris_diameter(landmarks, RIGHT_IRIS, width, height, iod)

        # Instantaneous gaze speed.
        speed = 0.0
        if self._prev is not None:
            pt, ph, pv = self._prev
            dt = now_s - pt
            if dt > 1e-6:
                speed = float(np.hypot(h - ph, v - pv) / dt)
        self._prev = (now_s, h, v)
        self._push(now_s, h, v, speed)

        # Saccade state machine: rising edge starts, falling edge closes.
        finished = None
        if speed > self.saccade_speed:
            if not self._in_saccade:
                self._in_saccade = True
                self._sacc_start = (now_s, h, v)
                self._peak_speed = speed
            else:
                self._peak_speed = max(self._peak_speed, speed)
        elif self._in_saccade and self._sacc_start is not None:
            t0, h0, v0 = self._sacc_start
            amp = float(np.hypot(h - h0, v - v0))
            finished = {
                "start_s": round(t0, 3),
                "duration_s": round(now_s - t0, 3),
                "amplitude": round(amp, 4),
                "peak_speed": round(self._peak_speed, 3),
            }
            self.saccades.append(finished)
            self._in_saccade = False
            self._sacc_start = None
            self._peak_speed = 0.0

        # Fixation stability: spread of gaze during non-saccadic samples.
        fix = [(hh, vv) for _, hh, vv, sp in self._samples if sp <= self.saccade_speed]
        if len(fix) >= 5:
            arr = np.array(fix)
            stability = float(np.hypot(arr[:, 0].std(), arr[:, 1].std()))
        else:
            stability = 0.0

        # Nystagmus screen on the horizontal gaze series.
        nys_ratio, nys_hz = 0.0, 0.0
        if len(self._samples) >= 32:
            ts = np.array([s[0] for s in self._samples])
            hs = np.array([s[1] for s in self._samples])
            span = ts[-1] - ts[0]
            fps = (len(ts) - 1) / span if span > 1e-6 else 0.0
            nys_ratio, nys_hz = detect_oscillation(hs, fps)

        recent = [s for s in self.saccades if s["start_s"] >= now_s - self.window_s]
        return {
            "gaze_h": round(h, 3),
            "gaze_v": round(v, 3),
            "speed": round(speed, 3),
            "in_saccade": self._in_saccade,
            "saccade_count": len(self.saccades),
            "saccades_in_window": len(recent),
            "mean_saccade_speed": round(
                float(np.mean([s["peak_speed"] for s in recent])), 3) if recent else 0.0,
            "mean_saccade_amplitude": round(
                float(np.mean([s["amplitude"] for s in recent])), 4) if recent else 0.0,
            "fixation_stability": round(stability, 4),
            "nystagmus_ratio": round(nys_ratio, 3),
            "nystagmus_hz": round(nys_hz, 2),
            "nystagmus_suspected": bool(nys_ratio > 0.5 and len(self._samples) >= 32),
            "pupil_proxy_left": round(d_l, 4),
            "pupil_proxy_right": round(d_r, 4),
            "pupil_asymmetry": round(abs(d_l - d_r), 4),
            "last_saccade": finished,
        }
