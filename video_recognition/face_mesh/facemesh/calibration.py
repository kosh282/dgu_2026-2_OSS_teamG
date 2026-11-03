"""Gaze calibration — map raw iris ratios to screen coordinates.

Raw gaze from :func:`metrics.gaze_direction` is a normalized ``(h, v)`` pair
that only loosely tracks where a person looks. Calibration fits a mapping from
``(h, v)`` to screen ``(x, y)`` using known fixation points (typically a 3x3
grid). Two models are available:

- ``degree=1`` — affine (6 parameters), robust with few points.
- ``degree=2`` — quadratic (12 parameters), corrects mild nonlinearity.

Fit quality is reported as a leave-one-out RMS error in pixels, so callers can
show the user how trustworthy the calibration is.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _design(h: np.ndarray, v: np.ndarray, degree: int) -> np.ndarray:
    """Design matrix rows for the chosen polynomial degree."""
    ones = np.ones_like(h)
    if degree == 1:
        return np.column_stack([ones, h, v])
    if degree == 2:
        return np.column_stack([ones, h, v, h * h, h * v, v * v])
    raise ValueError("degree must be 1 (affine) or 2 (quadratic)")


class GazeCalibrator:
    """Collect (raw_gaze -> screen) samples and fit a polynomial mapping.

    Args:
        degree: 1 for affine, 2 for quadratic.
    """

    def __init__(self, degree: int = 1) -> None:
        if degree not in (1, 2):
            raise ValueError("degree must be 1 or 2")
        self.degree = degree
        self._h: list[float] = []
        self._v: list[float] = []
        self._x: list[float] = []
        self._y: list[float] = []
        self.coef: np.ndarray | None = None       # (n_terms, 2)
        self.rms_error: float | None = None        # leave-one-out RMS (px)

    def add_sample(self, raw_h: float, raw_v: float, screen_x: float, screen_y: float) -> None:
        self._h.append(raw_h)
        self._v.append(raw_v)
        self._x.append(screen_x)
        self._y.append(screen_y)

    @property
    def n_samples(self) -> int:
        return len(self._h)

    @property
    def is_fitted(self) -> bool:
        return self.coef is not None

    def _fit_coef(self, h, v, x, y) -> np.ndarray:
        A = _design(h, v, self.degree)
        targets = np.column_stack([x, y])
        coef, *_ = np.linalg.lstsq(A, targets, rcond=None)
        return coef

    def fit(self) -> float:
        """Fit the mapping and return the leave-one-out RMS error (pixels)."""
        min_pts = 3 if self.degree == 1 else 6
        if self.n_samples < min_pts:
            raise ValueError(f"need at least {min_pts} points for degree {self.degree}")
        h = np.asarray(self._h)
        v = np.asarray(self._v)
        x = np.asarray(self._x)
        y = np.asarray(self._y)

        self.coef = self._fit_coef(h, v, x, y)

        # Leave-one-out cross-validation for an honest error estimate.
        n = self.n_samples
        if n > min_pts:
            errs = []
            idx = np.arange(n)
            for i in idx:
                m = idx != i
                coef = self._fit_coef(h[m], v[m], x[m], y[m])
                pred = _design(h[i:i+1], v[i:i+1], self.degree) @ coef
                errs.append(np.linalg.norm(pred[0] - np.array([x[i], y[i]])))
            self.rms_error = float(np.sqrt(np.mean(np.square(errs))))
        else:  # not enough points to hold one out; fall back to train RMS
            pred = _design(h, v, self.degree) @ self.coef
            res = pred - np.column_stack([x, y])
            self.rms_error = float(np.sqrt(np.mean(np.sum(res * res, axis=1))))
        return self.rms_error

    def apply(self, raw_h: float, raw_v: float) -> tuple[float, float]:
        if self.coef is None:
            raise RuntimeError("calibrator is not fitted; call fit() first")
        row = _design(np.array([raw_h]), np.array([raw_v]), self.degree)
        x, y = (row @ self.coef)[0]
        return float(x), float(y)

    def save(self, path: str | Path) -> None:
        if self.coef is None:
            raise RuntimeError("nothing to save; calibrator is not fitted")
        Path(path).write_text(json.dumps({
            "degree": self.degree,
            "coef": self.coef.tolist(),
            "rms_error": self.rms_error,
        }))

    @classmethod
    def load(cls, path: str | Path) -> "GazeCalibrator":
        data = json.loads(Path(path).read_text())
        obj = cls(degree=data.get("degree", 1))
        obj.coef = np.asarray(data["coef"], dtype=float)
        obj.rms_error = data.get("rms_error")
        return obj


def grid_targets(width: int, height: int, margin: float = 0.12) -> list[tuple[int, int]]:
    """A 3x3 grid of screen fixation points inset from the edges."""
    xs = [margin, 0.5, 1 - margin]
    ys = [margin, 0.5, 1 - margin]
    return [(int(x * width), int(y * height)) for y in ys for x in xs]
