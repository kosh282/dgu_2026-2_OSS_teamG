"""Tests for the clinical-style metrics: symmetry, movement, rPPG."""

import math

import numpy as np
import pytest

from facemesh.medical.geometry import (
    IDX_BOTTOM, IDX_EYE_L, IDX_EYE_R, IDX_ORIGIN, IDX_TOP, normalize, normalize_3d,
)
from facemesh.medical.movement import EXPRESSIVE_IDX, MovementAnalyzer, tremor_power
from facemesh.medical.rppg import (
    FOREHEAD, LEFT_CHEEK, RIGHT_CHEEK, HeartRateEstimator,
    estimate_bpm, estimate_respiration, hrv_metrics, inter_beat_intervals,
    pos_signal, skin_mask,
)
from facemesh.medical.symmetry import (
    REGIONS, DynamicSymmetry, SymmetryBaseline, symmetry_metrics,
)


class LM:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def _lms(pts):
    return [LM(*row) for row in pts]


def _scaffold(n=478, dim=3):
    pts = np.zeros((n, dim))
    pts[IDX_TOP][:2] = [0.5, 0.20]
    pts[IDX_BOTTOM][:2] = [0.5, 0.80]
    pts[IDX_ORIGIN][:2] = [0.5, 0.40]
    pts[IDX_EYE_R][:2] = [0.42, 0.40]
    pts[IDX_EYE_L][:2] = [0.58, 0.40]
    return pts


def _rot3(rx, ry, rz):
    ax, ay, az = map(math.radians, (rx, ry, rz))
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# --- geometry ------------------------------------------------------------
def test_normalize_2d_invariant_to_rigid_motion():
    base = _scaffold(dim=2)
    base[20] = [0.55, 0.6]
    A = normalize(_lms(base), 1, 1)
    a = math.radians(23)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    B = normalize(_lms((base * 1.7) @ R.T + [0.3, -0.2]), 1, 1)
    assert np.abs(A - B).max() < 1e-6


def test_normalize_3d_invariant_to_full_3d_rotation():
    base = _scaffold()
    base[20] = [0.55, 0.6, 0.04]
    A = normalize_3d(_lms(base), 1, 1)
    B = normalize_3d(_lms((base * 1.6) @ _rot3(20, 35, -15).T + [0.3, -0.2, 0.1]), 1, 1)
    assert np.abs(A - B).max() < 1e-6


def test_normalize_puts_midline_at_zero():
    N = normalize(_lms(_scaffold(dim=2)), 1, 1)
    assert abs(N[IDX_ORIGIN][0]) < 1e-9 and abs(N[IDX_TOP][0]) < 1e-9


# --- symmetry ------------------------------------------------------------
def _sym_face(rng):
    p = _scaffold()
    for pairs in REGIONS.values():
        for r, l in pairs:
            dx = 0.03 + rng.random() * 0.06
            y = 0.35 + rng.random() * 0.30
            p[r] = [0.5 - dx, y, 0]
            p[l] = [0.5 + dx, y, 0]
    return p


def test_symmetric_face_scores_near_one():
    m = symmetry_metrics(_lms(_sym_face(np.random.default_rng(5))), 640, 640)
    assert m["symmetry"] > 0.99 and m["asymmetry"] < 1e-6


def test_mouth_droop_localizes_to_mouth():
    p = _sym_face(np.random.default_rng(5))
    for _, l in REGIONS["mouth"]:
        p[l][1] += 0.05
    m = symmetry_metrics(_lms(p), 640, 640)
    assert m["worst_region"] == "mouth" and m["symmetry"] < 0.9


def test_3d_detects_depth_droop_that_2d_misses():
    """A purely out-of-plane droop is invisible in 2D but must be caught in 3D."""
    p = _sym_face(np.random.default_rng(5))
    base2 = symmetry_metrics(_lms(p), 640, 640, use_3d=False)["symmetry"]
    base3 = symmetry_metrics(_lms(p), 640, 640, use_3d=True)["symmetry"]
    for _, l in REGIONS["mouth"]:
        p[l][2] += 0.05                       # depth only
    d2 = symmetry_metrics(_lms(p), 640, 640, use_3d=False)
    d3 = symmetry_metrics(_lms(p), 640, 640, use_3d=True)
    assert abs(d2["symmetry"] - base2) < 1e-6        # 2D blind
    assert d3["symmetry"] < base3 - 0.05             # 3D detects
    assert d3["worst_region"] == "mouth"


def test_symmetry_unaffected_by_3d_head_rotation():
    p = _sym_face(np.random.default_rng(2))
    for _, l in REGIONS["mouth"]:
        p[l][1] += 0.03
    a = symmetry_metrics(_lms(p), 640, 640)["symmetry"]
    moved = (p - 0.5) @ _rot3(15, 25, -10).T * 1.4 + 0.5
    b = symmetry_metrics(_lms(moved), 640, 640)["symmetry"]
    assert abs(a - b) < 1e-3


def test_no_duplicate_or_self_pairs():
    for pairs in REGIONS.values():
        seen = set()
        for a, b in pairs:
            assert a != b
            key = frozenset((a, b))
            assert key not in seen
            seen.add(key)


def test_baseline_separates_natural_from_acquired_asymmetry():
    rng = np.random.default_rng(9)
    p = _sym_face(rng)
    for _, l in REGIONS["mouth"]:
        p[l][1] += 0.012                      # this person's natural asymmetry

    bl = SymmetryBaseline()
    for _ in range(15):
        bl.add_sample(symmetry_metrics(_lms(p + rng.normal(0, 0.0008, p.shape)), 640, 640))
    bl.fit()

    normal = bl.compare(symmetry_metrics(_lms(p), 640, 640))
    assert normal["relative_symmetry"] > 0.95   # their normal reads as normal

    for _, l in REGIONS["mouth"]:
        p[l][1] += 0.05                        # acquired droop on top
    acquired = bl.compare(symmetry_metrics(_lms(p), 640, 640))
    assert acquired["relative_symmetry"] < normal["relative_symmetry"] - 0.1
    assert acquired["worst_delta_region"] == "mouth"


def test_baseline_requires_enough_samples():
    bl = SymmetryBaseline()
    bl.add_sample(symmetry_metrics(_lms(_sym_face(np.random.default_rng(1))), 640, 640))
    with pytest.raises(ValueError):
        bl.fit()


def test_dynamic_symmetry_captures_transient_peak():
    ds = DynamicSymmetry(window_s=3.0)
    p_rest = _sym_face(np.random.default_rng(3))
    out = None
    for k in range(90):
        t = k / 30
        droop = 0.05 * math.sin(math.pi * min(t / 1.5, 1.0)) if t < 1.5 else 0.0
        pf = p_rest.copy()
        for _, l in REGIONS["mouth"]:
            pf[l][1] += droop
        out = ds.update(symmetry_metrics(_lms(pf), 640, 640), t)
    assert out["peak_asymmetry"] > out["mean_asymmetry"]
    assert out["peak_worst_region"] == "mouth"


# --- movement ------------------------------------------------------------
def _expr_face(scale, t):
    p = _scaffold(dim=2)
    for k, i in enumerate(EXPRESSIVE_IDX):
        p[i] = [0.45 + 0.002 * k,
                0.5 + 0.001 * k + scale * 0.02 * math.sin(2 * math.pi * 0.8 * t + k)]
    return _lms(p)


def _run_movement(scale, frames=90, fps=30.0, task="smile"):
    ma = MovementAnalyzer(window_s=3.0)
    ma.set_task(task)
    t, out = 0.0, None
    for _ in range(frames):
        t += 1 / fps
        out = ma.update(_expr_face(scale, t), 640, 640, t)
    return out


def test_expressive_face_scores_high():
    out = _run_movement(1.0)
    assert out["expressivity"] > 0.5 and out["reduced_expressivity"] is False


def test_reduced_expression_flagged():
    out = _run_movement(0.05)
    assert out["reduced_expressivity"] is True and out["expressivity"] < 0.3


def test_movement_ignores_rigid_head_motion():
    ma = MovementAnalyzer(window_s=3.0)
    base = _scaffold(dim=2)
    for k, i in enumerate(EXPRESSIVE_IDX):
        base[i] = [0.45 + 0.002 * k, 0.5 + 0.001 * k]
    t, out = 0.0, None
    for _ in range(90):
        t += 1 / 30
        a = math.radians(20 * math.sin(t))
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        moved = (base - 0.5) @ R.T + [0.5 + 0.1 * math.sin(t), 0.5]
        out = ma.update(_lms(moved), 640, 640, t)
    assert out["amplitude"] < 1e-6


def test_tremor_power_isolates_band():
    fps, n = 30.0, 150
    t = np.arange(n) / fps
    ratio_t, hz = tremor_power(np.sin(2 * np.pi * 5.0 * t), fps)
    ratio_s, _ = tremor_power(np.sin(2 * np.pi * 1.0 * t), fps)
    assert ratio_t > 0.8 and abs(hz - 5.0) < 0.5
    assert ratio_s < 0.1


def test_tremor_detected_end_to_end():
    """Tremor must come from the signed position series, not rectified jitter."""
    ma = MovementAnalyzer(window_s=3.0)
    ma.set_task("rest")
    out, fps = None, 30.0
    for k in range(120):
        t = k / fps
        p = _scaffold(dim=2)
        for j, i in enumerate(EXPRESSIVE_IDX):
            p[i] = [0.45 + 0.002 * j,
                    0.5 + 0.001 * j + 0.004 * math.sin(2 * math.pi * 5.0 * t)]
        out = ma.update(_lms(p), 640, 640, t)
    assert out["tremor_detected"] and 4.0 <= out["tremor_hz"] <= 6.0


def test_expressivity_uninterpretable_without_task():
    ma = MovementAnalyzer(window_s=2.0)
    out = ma.update(_expr_face(1.0, 0.1), 640, 640, 0.1)
    assert out["interpretable"] is False and out["task"] is None
    ma.set_task("smile")
    out = ma.update(_expr_face(1.0, 0.2), 640, 640, 0.2)
    assert out["interpretable"] is True and out["task"] == "smile"


def test_task_summary_separates_rest_from_smile():
    ma = MovementAnalyzer(window_s=2.0)
    t, fps = 0.0, 30.0
    ma.set_task("rest")
    for _ in range(60):
        t += 1 / fps
        ma.update(_expr_face(0.01, t), 640, 640, t)
    ma.set_task("smile")
    for _ in range(60):
        t += 1 / fps
        ma.update(_expr_face(1.0, t), 640, 640, t)
    s = ma.task_summary()
    assert s["smile"]["peak_expressivity"] > s["rest"]["peak_expressivity"]


# --- rPPG ----------------------------------------------------------------
def _rppg_landmarks():
    lms = [LM(0.5, 0.5) for _ in range(478)]
    for k, i in enumerate(FOREHEAD):
        lms[i] = LM(0.35 + 0.03 * k, 0.20 + 0.01 * (k % 2))
    for k, i in enumerate(LEFT_CHEEK):
        lms[i] = LM(0.58 + 0.02 * k, 0.50 + 0.01 * (k % 2))
    for k, i in enumerate(RIGHT_CHEEK):
        lms[i] = LM(0.25 + 0.02 * k, 0.50 + 0.01 * (k % 2))
    lms[33] = LM(0.40, 0.40)
    lms[263] = LM(0.60, 0.40)
    return lms


SKIN_BGR = (140, 160, 205)


def _synthetic_rgb(bpm, seconds=10.0, fps=30.0, seed=1):
    n = int(seconds * fps)
    t = np.arange(n) / fps
    pulse = np.sin(2 * np.pi * (bpm / 60.0) * t)
    rng = np.random.default_rng(seed)
    drift = 2 * np.sin(2 * np.pi * 0.05 * t)
    R = 180 + 0.3 * pulse + rng.normal(0, 0.4, n) + drift
    G = 120 + 1.2 * pulse + rng.normal(0, 0.4, n) + drift
    B = 100 + 0.5 * pulse + rng.normal(0, 0.4, n) + drift
    return np.stack([R, G, B], axis=1), fps


@pytest.mark.parametrize("true_bpm", [54.0, 72.0, 96.0, 120.0])
def test_estimate_bpm_recovers_known_rate(true_bpm):
    rgb, fps = _synthetic_rgb(true_bpm)
    bpm, conf = estimate_bpm(pos_signal(rgb), fps)
    assert abs(bpm - true_bpm) < 4.0 and conf > 0.5


def test_estimate_bpm_rejects_short_signal():
    bpm, conf = estimate_bpm(np.zeros(10), 30.0)
    assert bpm == 0.0 and conf == 0.0


def test_skin_mask_rejects_hair_and_non_skin():
    lms = _rppg_landmarks()
    skin = np.full((480, 640, 3), SKIN_BGR, np.uint8)
    n_skin = int(skin_mask(skin, lms).sum())

    hair = skin.copy()
    hair[:144] = (20, 20, 25)                 # dark hair over the forehead
    assert int(skin_mask(hair, lms).sum()) < n_skin

    green = skin.copy()
    green[:, :320] = (40, 200, 40)            # non-skin chrominance
    assert int(skin_mask(green, lms).sum()) < n_skin


def test_hrv_recovers_variability():
    fps, bpm = 30.0, 72.0
    t, phase, sig = 0.0, 0.0, []
    while t < 20.0:
        inst = bpm + 3.0 * math.sin(2 * math.pi * 0.2 * t)
        phase += 2 * math.pi * (inst / 60.0) / fps
        sig.append(math.sin(phase))
        t += 1 / fps
    ibis = inter_beat_intervals(np.array(sig), fps, bpm)
    hrv = hrv_metrics(ibis)
    assert abs(60.0 / ibis.mean() - bpm) < 4.0
    assert hrv["sdnn_ms"] > 0 and hrv["rmssd_ms"] > 0


@pytest.mark.parametrize("true_rpm", [12.0, 18.0])
def test_respiration_recovers_known_rate(true_rpm):
    fps, n = 30.0, int(30 * 30)
    t = np.arange(n) / fps
    sig = np.sin(2 * np.pi * (72 / 60) * t) + 0.8 * np.sin(2 * np.pi * (true_rpm / 60) * t)
    rpm, conf = estimate_respiration(sig, fps)
    assert abs(rpm - true_rpm) < 2.5 and conf > 0.5


def test_motion_rejection_drops_bad_frames():
    lms = _rppg_landmarks()
    frame = np.full((480, 640, 3), SKIN_BGR, np.uint8)

    still = HeartRateEstimator(window_s=10.0, min_seconds=3.0, max_motion=0.6)
    for k in range(60):
        still.update(frame, lms, k / 30)
    assert still.rejected == 0

    jitter = HeartRateEstimator(window_s=10.0, min_seconds=3.0, max_motion=0.6)
    for k in range(60):
        shift = 0.15 * (k % 2)
        jitter.update(frame, [LM(l.x + shift, l.y) for l in lms], k / 30)
    assert jitter.rejected > 30


def test_heart_rate_not_ready_before_warmup():
    est = HeartRateEstimator(min_seconds=5.0)
    out = est.update(np.full((100, 100, 3), 120, np.uint8), _rppg_landmarks(), 0.5)
    assert out["ready"] is False and out["bpm"] == 0.0
