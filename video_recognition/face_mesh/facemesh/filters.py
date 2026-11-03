"""Signal smoothing — the 1-Euro filter (Casiez et al., 2012).

A low-pass filter with an adaptive cutoff: it smooths hard when the signal is
slow (killing jitter) and loosens when the signal moves fast (avoiding lag).
Used here to de-jitter head-pose angles, but it works on any scalar stream.
"""

from __future__ import annotations

import math

import numpy as np


class _LowPass:
    """First-order exponential low-pass filter with externally-set alpha."""

    def __init__(self) -> None:
        self.y: float | None = None  # last raw input
        self.s: float | None = None  # last filtered output

    def __call__(self, x: float, alpha: float) -> float:
        s = x if self.s is None else alpha * x + (1.0 - alpha) * self.s
        self.y, self.s = x, s
        return s


class OneEuroFilter:
    """Adaptive low-pass filter for a single scalar signal.

    Args:
        freq: Initial sampling frequency (Hz); refined from timestamps.
        min_cutoff: Minimum cutoff frequency (Hz). Lower = smoother at rest.
        beta: Speed coefficient. Higher = less lag on fast motion.
        d_cutoff: Cutoff for the derivative low-pass.
    """

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0,
                 beta: float = 0.05, d_cutoff: float = 1.0) -> None:
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_t: float | None = None

    def _alpha(self, cutoff: float) -> float:
        te = 1.0 / self.freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, timestamp: float | None = None) -> float:
        if self._last_t is not None and timestamp is not None and timestamp > self._last_t:
            self.freq = 1.0 / (timestamp - self._last_t)
        self._last_t = timestamp

        prev = self._x.s
        dx = 0.0 if prev is None else (x - prev) * self.freq
        edx = self._dx(dx, self._alpha(self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x(x, self._alpha(cutoff))


class PoseSmoother:
    """Smooth a (pitch, yaw, roll) triple with three 1-Euro filters."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05) -> None:
        self._f = [OneEuroFilter(min_cutoff=min_cutoff, beta=beta) for _ in range(3)]

    def update(self, pitch: float, yaw: float, roll: float,
               timestamp: float | None = None) -> tuple[float, float, float]:
        return (
            self._f[0](pitch, timestamp),
            self._f[1](yaw, timestamp),
            self._f[2](roll, timestamp),
        )


class LandmarkSmoother:
    """Vectorized 1-Euro smoothing of whole landmark arrays, per face.

    Applies the adaptive filter to every ``(x, y, z)`` coordinate at once
    (NumPy, not 1434 Python filter objects), keyed by a face/track ID so each
    tracked person is smoothed independently. This removes the frame-to-frame
    shimmer of the mesh while staying responsive to real motion.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05,
                 d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._state: dict[int, dict] = {}  # face_id -> {x_s, dx_s, t}

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * freq)

    def apply(self, face_id: int, coords: np.ndarray, timestamp: float) -> np.ndarray:
        """Return a smoothed copy of ``coords`` (shape ``(N, 3)``)."""
        coords = np.asarray(coords, dtype=float)
        st = self._state.get(face_id)
        if st is None:
            self._state[face_id] = {"x_s": coords.copy(),
                                    "dx_s": np.zeros_like(coords), "t": timestamp}
            return coords

        dt = timestamp - st["t"]
        freq = 1.0 / dt if dt > 1e-6 else 30.0
        st["t"] = timestamp

        dx = (coords - st["x_s"]) * freq
        a_d = self._alpha(self.d_cutoff, freq)
        st["dx_s"] = a_d * dx + (1.0 - a_d) * st["dx_s"]
        cutoff = self.min_cutoff + self.beta * np.abs(st["dx_s"])
        a = self._alpha(cutoff, freq)
        st["x_s"] = a * coords + (1.0 - a) * st["x_s"]
        return st["x_s"].copy()

    def drop(self, face_id: int) -> None:
        self._state.pop(face_id, None)
