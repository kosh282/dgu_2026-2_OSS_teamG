"""Pupillary light reflex (PLR).

Shine a light in one eye and both pupils constrict, then slowly redilate. The
timing and amplitude of that response are a genuine neurological measure:
latency, constriction velocity and the direct/consensual difference are used in
concussion assessment, brainstem evaluation and autonomic testing.

Here the "light" is the screen: the window flashes white, and the iris diameter
is tracked before, during and after. From that trace we extract:

- **baseline diameter** — mean before the flash.
- **minimum diameter** and **constriction amplitude** (% change).
- **latency** — time from flash onset to the start of constriction.
- **peak constriction velocity**.
- **redilation** — recovery toward baseline (75% recovery time).
- **anisocoria of response** — left/right difference in amplitude.

.. warning::
   **This is a toy reproduction, not pupillometry.** Real pupillometers use
   infrared illumination at high frame rate with a calibrated, controlled light
   stimulus. Here:

   - MediaPipe returns an **iris ring**, not the pupil. The pupil is inside the
     iris, and its boundary is not resolved at all in visible light on a dark
     iris. The measured "diameter" is therefore a *proxy that may barely move*.
   - A webcam at 30 fps samples a reflex whose latency is ~200-300 ms; you get
     roughly 6-9 samples across the entire constriction phase.
   - Screen brightness is uncalibrated and varies by display, ambient light and
     distance.

   Treat every number here as relative and exploratory. Do not compare across
   sessions, devices or people.

SCREENING / EDUCATIONAL USE ONLY — not a diagnostic device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _smooth(x: np.ndarray, fps: float, seconds: float) -> np.ndarray:
    """Moving average; averages away per-frame landmark jitter."""
    win = max(1, int(round(fps * seconds)))
    if win <= 1 or len(x) < win:
        return x
    kern = np.ones(win) / win
    pad = win // 2
    return np.convolve(np.pad(x, pad, mode="edge"), kern, mode="same")[pad:pad + len(x)]

# Phases of the protocol, in seconds.
DEFAULT_BASELINE_S = 3.0
DEFAULT_FLASH_S = 0.5
DEFAULT_RECOVERY_S = 5.0

# A constriction of at least this fraction of baseline counts as a response.
MIN_RESPONSE = 0.03

#: The constriction must exceed this multiple of the *expected extreme* of the
#: baseline noise before it can be called a response.
#:
#: Comparing the single minimum sample to the baseline standard deviation is not
#: enough: over N noise samples the minimum naturally lies ~sqrt(2 ln N) sigma
#: below the mean, so pure jitter would always "pass" a plain 3-sigma test. The
#: trace is therefore smoothed first, and the noise floor scaled by that factor.
MIN_SNR = 1.5

#: Smoothing window (seconds) applied before extremum detection. Real
#: constriction lasts hundreds of ms; landmark jitter is per-frame.
SMOOTH_S = 0.10

#: Physiological latency window (seconds). A real PLR latency is ~0.2-0.3 s;
#: anything outside this range is a measurement artifact, not a reflex.
LATENCY_MIN_S, LATENCY_MAX_S = 0.10, 0.60

#: Direct and consensual responses are near-simultaneous. A larger inter-eye
#: latency difference means the measurement failed, not that the eyes differ.
MAX_LATENCY_DIFFERENCE_S = 0.10


@dataclass
class PLRTrace:
    """Time series of iris diameter through one flash trial."""

    times: list[float] = field(default_factory=list)
    left: list[float] = field(default_factory=list)
    right: list[float] = field(default_factory=list)
    flash_onset: float | None = None

    def add(self, t: float, d_left: float, d_right: float) -> None:
        self.times.append(t)
        self.left.append(d_left)
        self.right.append(d_right)


def _analyze_eye(t: np.ndarray, d: np.ndarray, onset: float,
                 baseline_end: float) -> dict:
    """Extract PLR features for one eye from its diameter trace.

    Enforces (issue #5) that the constriction is distinguishable from the
    baseline's own noise, and that any latency lies in a physiological window.
    Otherwise the eye is reported as ``responded=False`` with a reason, rather
    than as a confident-looking reflex measured from landmark jitter.
    """
    pre = d[t < baseline_end]
    if len(pre) < 3:
        return {"valid": False, "reason": "not enough baseline samples"}
    baseline = float(np.median(pre))
    if baseline <= 0:
        return {"valid": False, "reason": "no iris measured"}

    # Smooth before looking for extrema: real constriction lasts ~100s of ms,
    # landmark jitter does not.
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    fps = 1.0 / dt if dt > 0 else 30.0
    d = _smooth(np.asarray(d, dtype=float), fps, SMOOTH_S)
    pre = d[t < baseline_end]
    baseline_noise = float(np.std(pre)) if len(pre) > 1 else 0.0

    post_mask = t >= onset
    if post_mask.sum() < 5:
        return {"valid": False, "reason": "not enough post-flash samples"}
    tp, dp = t[post_mask], d[post_mask]

    min_i = int(np.argmin(dp))
    d_min = float(dp[min_i])
    t_min = float(tp[min_i])
    constriction = baseline - d_min
    amplitude = constriction / baseline                # fraction of baseline

    # SNR gate. Normalize by the EXPECTED EXTREME of the noise over n samples,
    # sqrt(2 ln n) * sigma, otherwise the minimum of pure noise always passes.
    n_post = max(len(dp), 2)
    expected_extreme = baseline_noise * math.sqrt(2.0 * math.log(n_post))
    snr = constriction / expected_extreme if expected_extreme > 0 else float("inf")
    reasons: list[str] = []
    if amplitude < MIN_RESPONSE:
        reasons.append("constriction below threshold")
    if snr < MIN_SNR:
        reasons.append("constriction within baseline noise")

    # Latency: first sample crossing a threshold set above the noise floor, not
    # an arbitrary fraction of baseline (which any jitter will cross eventually).
    thresh = baseline - max(baseline * MIN_RESPONSE / 2, expected_extreme / 2)
    below = np.where(dp < thresh)[0]
    latency = float(tp[below[0]] - onset) if len(below) else float("nan")
    if latency == latency and not (LATENCY_MIN_S <= latency <= LATENCY_MAX_S):
        reasons.append(f"latency {latency:.2f}s outside physiological range")
        latency = float("nan")

    responded = not reasons

    # Peak constriction velocity (most negative derivative), baseline-normalized.
    if len(tp) > 2:
        dt = np.gradient(tp)
        vel = np.gradient(dp) / np.where(dt == 0, 1e-9, dt)
        peak_vel = float(np.min(vel[: max(min_i + 1, 2)]) / baseline)
    else:
        peak_vel = float("nan")

    # Redilation: time after the minimum to recover 75% of the constriction.
    recover_to = d_min + 0.75 * (baseline - d_min)
    after = np.where((tp > t_min) & (dp >= recover_to))[0]
    t75 = float(tp[after[0]] - t_min) if len(after) else float("nan")

    return {
        "valid": True,
        "baseline": round(baseline, 5),
        "baseline_noise": round(baseline_noise, 6),
        "snr": round(snr, 2) if np.isfinite(snr) else None,
        "min_diameter": round(d_min, 5),
        "constriction_pct": round(100.0 * amplitude, 2),
        "latency_s": round(latency, 3) if latency == latency else None,
        "peak_velocity_per_s": round(peak_vel, 4) if peak_vel == peak_vel else None,
        "time_to_min_s": round(t_min - onset, 3),
        "recovery_75_s": round(t75, 3) if t75 == t75 else None,
        "responded": bool(responded),
        "reasons": reasons,
        "samples": int(len(d)),
    }


def analyze_plr(trace: PLRTrace, baseline_end: float | None = None) -> dict:
    """Analyze a completed :class:`PLRTrace`.

    Args:
        baseline_end: End of the pre-flash window. Defaults to the flash onset.
    """
    if trace.flash_onset is None:
        return {"valid": False, "reason": "flash never fired"}
    t = np.asarray(trace.times, dtype=float)
    if len(t) < 10:
        return {"valid": False, "reason": "too few samples"}
    onset = float(trace.flash_onset)
    baseline_end = onset if baseline_end is None else baseline_end

    left = _analyze_eye(t, np.asarray(trace.left, dtype=float), onset, baseline_end)
    right = _analyze_eye(t, np.asarray(trace.right, dtype=float), onset, baseline_end)

    out: dict = {"valid": left.get("valid") and right.get("valid"),
                 "left": left, "right": right, "flash_onset_s": round(onset, 3)}
    if not out["valid"]:
        return out

    la, ra = left["constriction_pct"], right["constriction_pct"]
    out["response_asymmetry_pct"] = round(abs(la - ra), 2)
    out["weaker_response"] = "left" if la < ra else "right"
    out["both_responded"] = bool(left["responded"] and right["responded"])

    warnings: list[str] = []
    if left["latency_s"] is not None and right["latency_s"] is not None:
        diff = abs(left["latency_s"] - right["latency_s"])
        out["latency_difference_s"] = round(diff, 3)
        # Direct and consensual responses are near-simultaneous. A large
        # difference means the measurement failed (issue #5).
        if diff > MAX_LATENCY_DIFFERENCE_S:
            warnings.append(
                f"inter-eye latency difference {diff:.2f}s exceeds "
                f"{MAX_LATENCY_DIFFERENCE_S}s; the direct and consensual responses "
                "are near-simultaneous, so this indicates measurement failure")
            out["both_responded"] = False

    for eye in ("left", "right"):
        for r in out[eye].get("reasons", []):
            warnings.append(f"{eye}: {r}")

    out["warnings"] = warnings
    out["measurement_trustworthy"] = bool(out["both_responded"] and not warnings)
    if not out["measurement_trustworthy"]:
        out["interpretation"] = (
            "No trustworthy reflex was measured. This almost certainly means the "
            "iris landmarks cannot resolve your pupil (see module docstring), not "
            "that your reflex is absent.")
    return out


class PLRSession:
    """Drive one flash trial and collect the trace.

    Call :meth:`update` once per frame with the current iris diameters. It
    returns the display state (whether to paint the screen white) and, once
    complete, the analysis.
    """

    def __init__(self, baseline_s: float = DEFAULT_BASELINE_S,
                 flash_s: float = DEFAULT_FLASH_S,
                 recovery_s: float = DEFAULT_RECOVERY_S) -> None:
        self.baseline_s = baseline_s
        self.flash_s = flash_s
        self.recovery_s = recovery_s
        self.trace = PLRTrace()
        self._t0: float | None = None
        self.finished = False
        self.result: dict | None = None

    @property
    def total_s(self) -> float:
        return self.baseline_s + self.flash_s + self.recovery_s

    def update(self, now_s: float, d_left: float, d_right: float) -> dict:
        if self._t0 is None:
            self._t0 = now_s
        elapsed = now_s - self._t0

        flashing = self.baseline_s <= elapsed < self.baseline_s + self.flash_s
        if flashing and self.trace.flash_onset is None:
            self.trace.flash_onset = elapsed

        if d_left > 0 and d_right > 0:
            self.trace.add(elapsed, d_left, d_right)

        if elapsed >= self.total_s and not self.finished:
            self.finished = True
            self.result = analyze_plr(self.trace)

        phase = ("baseline" if elapsed < self.baseline_s else
                 "flash" if flashing else
                 "recovery" if elapsed < self.total_s else "done")
        return {
            "phase": phase,
            "flashing": flashing,
            "elapsed": round(elapsed, 2),
            "progress": min(elapsed / self.total_s, 1.0),
            "finished": self.finished,
            "samples": len(self.trace.times),
        }
