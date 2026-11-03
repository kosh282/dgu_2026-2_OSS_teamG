"""Compare sessions over time.

"Is this face more asymmetric than it was last week?" is a far more defensible
question than "is this face asymmetric?", because it controls for the person's
own anatomy. This module loads exported sessions, extracts per-session summaries
with confidence intervals, and reports change with an explicit statement of
whether that change is distinguishable from noise.

    python scripts/compare_sessions.py week1.jsonl week2.jsonl week3.jsonl

.. warning::
   A "significant" change here means the confidence intervals do not overlap.
   It does **not** mean the change is clinically meaningful, nor that it is
   caused by anything in particular. Lighting, camera distance, time of day,
   caffeine and mood all move these numbers. Compare like with like.

SCREENING / EDUCATIONAL USE ONLY — not a diagnostic device.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .stats import Interval, block_bootstrap_ci

#: Metrics tracked across sessions: label -> (path in the medical dict, better direction)
TRACKED = {
    "symmetry": (("symmetry", "symmetry"), "higher"),
    "asymmetry": (("symmetry", "asymmetry"), "lower"),
    "expressivity": (("movement", "expressivity"), "higher"),
    "tremor_ratio": (("movement", "tremor_ratio"), "lower"),
    "blink_rate": (("blink", "rate_per_min"), "neutral"),
    "incomplete_blinks": (("blink", "incomplete_fraction"), "lower"),
    "heart_rate": (("heart_rate", "bpm"), "neutral"),
    "fixation_stability": (("gaze_dynamics", "fixation_stability"), "lower"),
}


def _dig(d: dict, path) -> float | None:
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur if isinstance(cur, (int, float)) and not isinstance(cur, bool) else None


@dataclass
class SessionSummary:
    """One session, reduced to per-metric intervals."""

    name: str
    frames: int
    usable_frames: int
    recorded: str | None
    metrics: dict[str, Interval]

    @property
    def usable_fraction(self) -> float:
        return self.usable_frames / self.frames if self.frames else 0.0

    def to_dict(self) -> dict:
        return {
            "session": self.name,
            "frames": self.frames,
            "usable_frames": self.usable_frames,
            "usable_fraction": round(self.usable_fraction, 3),
            "recorded": self.recorded,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }


def summarize_session(path: str | Path, n_boot: int = 1000) -> SessionSummary:
    """Reduce an exported ``.jsonl`` session to per-metric intervals.

    Only quality-passing frames of the primary face contribute. Confidence
    intervals use the moving-block bootstrap, since consecutive frames are
    autocorrelated.
    """
    path = Path(path)
    series: dict[str, list[float]] = {k: [] for k in TRACKED}
    frames = usable = 0

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        frames += 1
        faces = rec.get("faces") or []
        if not faces:
            continue
        med = faces[0].get("medical")
        if not med:
            continue
        q = med.get("quality")
        if q is not None and not q.get("ok", True):
            continue           # respect the quality gate
        usable += 1
        for label, (dig_path, _) in TRACKED.items():
            v = _dig(med, dig_path)
            if v is not None:
                series[label].append(v)

    metrics: dict[str, Interval] = {}
    for label, values in series.items():
        if len(values) >= 3:
            metrics[label] = block_bootstrap_ci(values, n_boot=n_boot)

    return SessionSummary(
        name=path.stem, frames=frames, usable_frames=usable,
        recorded=datetime.fromtimestamp(path.stat().st_mtime,
                                        tz=timezone.utc).isoformat(timespec="seconds"),
        metrics=metrics,
    )


def compare(a: SessionSummary, b: SessionSummary) -> dict:
    """Change from session ``a`` to session ``b``, per metric.

    ``separated`` is True when the confidence intervals do not overlap — weak
    evidence that the change exceeds sampling noise. It is not a hypothesis
    test and it is not a clinical finding.
    """
    out: dict[str, dict] = {}
    for label in TRACKED:
        ia, ib = a.metrics.get(label), b.metrics.get(label)
        if ia is None or ib is None:
            continue
        delta = ib.estimate - ia.estimate
        separated = (ib.low > ia.high) or (ib.high < ia.low)
        direction = TRACKED[label][1]
        if direction == "neutral" or delta == 0:
            verdict = "changed" if separated else "stable"
        else:
            improved = (delta > 0) if direction == "higher" else (delta < 0)
            verdict = ("improved" if improved else "worsened") if separated else "stable"

        out[label] = {
            "before": ia.to_dict(),
            "after": ib.to_dict(),
            "delta": round(delta, 5),
            "pct_change": (round(100.0 * delta / ia.estimate, 1)
                           if ia.estimate not in (0.0,) else None),
            "intervals_separated": bool(separated),
            "verdict": verdict,
        }
    return out


def trend(summaries: list[SessionSummary], label: str) -> dict | None:
    """Least-squares slope of a metric across ordered sessions."""
    pts = [(i, s.metrics[label].estimate)
           for i, s in enumerate(summaries) if label in s.metrics]
    if len(pts) < 3:
        return None
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    return {
        "metric": label,
        "sessions": len(pts),
        "slope_per_session": round(float(slope), 5),
        "r_squared": round(r2, 3),
        "first": round(float(y[0]), 5),
        "last": round(float(y[-1]), 5),
    }


def compare_many(paths, n_boot: int = 1000) -> dict:
    """Summarize several sessions, compare consecutive pairs, and fit trends."""
    summaries = [summarize_session(p, n_boot=n_boot) for p in paths]
    result: dict = {
        "sessions": [s.to_dict() for s in summaries],
        "comparisons": [],
        "trends": {},
    }
    for a, b in zip(summaries, summaries[1:]):
        result["comparisons"].append({
            "from": a.name, "to": b.name, "metrics": compare(a, b),
        })
    for label in TRACKED:
        t = trend(summaries, label)
        if t is not None:
            result["trends"][label] = t
    return result
