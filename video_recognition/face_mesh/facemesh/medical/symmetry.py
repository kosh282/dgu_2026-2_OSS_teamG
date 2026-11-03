"""Facial symmetry / asymmetry index.

Compares mirrored landmark pairs across the face's mid-sagittal plane. This is
the geometric core of facial-palsy and post-stroke facial assessment: a healthy
face is near-symmetric, while palsy shows a large, region-localized deviation
(usually mouth and eye on the affected side).

Three refinements over a naive score:

- **3D by default.** Deviation is measured in a 3D face-local frame, so a lip
  that droops *backwards* is measured rather than projected away.
- **Baselines.** Everyone has mild natural asymmetry, so an absolute score is
  weakly informative. :class:`SymmetryBaseline` records a person's own resting
  asymmetry and reports *change from their baseline*.
- **Dynamic scoring.** Palsy often hides at rest and appears on movement, so
  :class:`DynamicSymmetry` tracks the worst (peak) asymmetry over a window,
  which is what a prompted movement should be scored on.

All scores are pose/scale invariant (see :mod:`geometry`).

SCREENING / EDUCATIONAL USE ONLY — not a diagnostic device.
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .geometry import normalize, normalize_3d

# Mirrored landmark pairs (right_index, left_index) per region.
EYE_PAIRS = list(zip(
    (33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246, 7),
    (263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466, 249),
))
BROW_PAIRS = list(zip(
    (46, 52, 53, 55, 63, 65, 66, 70, 105, 107),
    (276, 282, 283, 285, 293, 295, 296, 300, 334, 336),
))
MOUTH_PAIRS = [
    (61, 291), (146, 375), (91, 321), (181, 405), (84, 314),
    (40, 270), (39, 269), (37, 267), (185, 409), (78, 308),
    (95, 324), (88, 318), (87, 317), (80, 310),
]
CHEEK_PAIRS = [(50, 280), (101, 330), (36, 266), (205, 425), (207, 427), (123, 352)]

REGIONS = {
    "eye": EYE_PAIRS,
    "brow": BROW_PAIRS,
    "mouth": MOUTH_PAIRS,
    "cheek": CHEEK_PAIRS,
}

# Deviation (inter-ocular units) treated as "large" when mapping to a 0-1 score.
FULL_SCALE_DEVIATION = 0.15


def _pair_deviation(N: np.ndarray, pairs) -> float:
    """Mean mirrored-pair deviation, in inter-ocular units.

    Reflect each left point across the mid-sagittal plane (negate x) and
    measure its distance to the paired right point. Works for 2D or 3D ``N``.
    """
    if not pairs:
        return 0.0
    right = N[[p[0] for p in pairs]]
    left = N[[p[1] for p in pairs]]
    mirror = np.ones(N.shape[1])
    mirror[0] = -1.0                      # reflect the x (left-right) axis only
    return float(np.linalg.norm(right - left * mirror, axis=1).mean())


def symmetry_metrics(landmarks, width: int, height: int, use_3d: bool = True) -> dict:
    """Asymmetry per region plus an overall symmetry score.

    Args:
        use_3d: Measure in the 3D face frame (recommended). Set ``False`` to
            reproduce a purely image-plane score.

    Returns per-region deviations (``*_asym``, inter-ocular units, 0 =
    symmetric), an overall ``asymmetry``, a ``symmetry`` score in ``[0, 1]``
    (1 = perfectly symmetric), and ``worst_region``.
    """
    N = normalize_3d(landmarks, width, height) if use_3d else normalize(landmarks, width, height)
    per_region = {name: _pair_deviation(N, pairs) for name, pairs in REGIONS.items()}
    overall = float(np.mean(list(per_region.values())))
    symmetry = float(np.clip(1.0 - overall / FULL_SCALE_DEVIATION, 0.0, 1.0))
    worst = max(per_region, key=per_region.get)

    out = {f"{k}_asym": round(v, 4) for k, v in per_region.items()}
    out["asymmetry"] = round(overall, 4)
    out["symmetry"] = round(symmetry, 3)
    out["worst_region"] = worst
    return out


class SymmetryBaseline:
    """Record a person's resting asymmetry, then report deviation from it.

    Absolute asymmetry conflates natural facial asymmetry (common, benign) with
    acquired asymmetry (the clinically interesting kind). Capturing a neutral
    baseline and reporting *change* is a far more defensible signal.
    """

    def __init__(self, subject: str | None = None) -> None:
        self._samples: list[dict] = []
        self.baseline: dict | None = None
        self.subject = subject
        self.created: str | None = None
        self._loaded_n = 0

    def add_sample(self, metrics: dict) -> None:
        self._samples.append(metrics)

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    def fit(self, min_samples: int = 10) -> dict:
        """Median per-region asymmetry over the collected neutral samples."""
        if self.n_samples < min_samples:
            raise ValueError(f"need >= {min_samples} samples, have {self.n_samples}")
        keys = [f"{r}_asym" for r in REGIONS] + ["asymmetry"]
        self.baseline = {
            k: float(np.median([s[k] for s in self._samples])) for k in keys
        }
        return self.baseline

    @property
    def is_fitted(self) -> bool:
        return self.baseline is not None

    def compare(self, metrics: dict) -> dict:
        """Deviation of ``metrics`` from the fitted baseline.

        Returns ``*_delta`` per region (positive = more asymmetric than this
        person's normal), an overall ``asymmetry_delta``, and a
        ``relative_symmetry`` score in ``[0, 1]``.
        """
        if self.baseline is None:
            raise RuntimeError("baseline is not fitted; call fit() first")
        out = {}
        for r in REGIONS:
            k = f"{r}_asym"
            out[f"{r}_delta"] = round(metrics[k] - self.baseline[k], 4)
        delta = metrics["asymmetry"] - self.baseline["asymmetry"]
        out["asymmetry_delta"] = round(delta, 4)
        out["relative_symmetry"] = round(
            float(np.clip(1.0 - max(delta, 0.0) / FULL_SCALE_DEVIATION, 0.0, 1.0)), 3)
        out["worst_delta_region"] = max(REGIONS, key=lambda r: out[f"{r}_delta"])
        return out

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "subject": self.subject,
            "created": self.created,
            "n_samples": self.n_samples or self._loaded_n,
            "baseline": self.baseline,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SymmetryBaseline":
        obj = cls(subject=data.get("subject"))
        obj.baseline = data.get("baseline")
        obj.created = data.get("created")
        obj._loaded_n = int(data.get("n_samples", 0))
        return obj

    def save(self, path: str | Path) -> Path:
        """Persist the fitted baseline to JSON so later sessions can compare."""
        if self.baseline is None:
            raise RuntimeError("nothing to save; baseline is not fitted")
        self.created = self.created or datetime.now(timezone.utc).isoformat(timespec="seconds")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SymmetryBaseline":
        """Load a baseline saved by :meth:`save`."""
        data = json.loads(Path(path).read_text())
        if "baseline" not in data or data["baseline"] is None:
            raise ValueError(f"{path} does not contain a fitted baseline")
        return cls.from_dict(data)


class DynamicSymmetry:
    """Track peak (worst) asymmetry over a rolling window.

    Palsy is often invisible at rest and obvious on movement, so the meaningful
    number during a prompted movement ("smile", "raise brows") is the *peak*
    asymmetry reached, not the instantaneous or mean value.
    """

    def __init__(self, window_s: float = 3.0) -> None:
        self.window_s = window_s
        self._samples: deque[tuple[float, dict]] = deque()

    def update(self, metrics: dict, now_s: float) -> dict:
        self._samples.append((now_s, metrics))
        cutoff = now_s - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        asyms = [m["asymmetry"] for _, m in self._samples]
        peak_idx = int(np.argmax(asyms))
        peak = self._samples[peak_idx][1]
        return {
            "peak_asymmetry": round(max(asyms), 4),
            "peak_symmetry": peak["symmetry"],
            "peak_worst_region": peak["worst_region"],
            "mean_asymmetry": round(float(np.mean(asyms)), 4),
            "samples": len(self._samples),
        }

    def reset(self) -> None:
        self._samples.clear()
