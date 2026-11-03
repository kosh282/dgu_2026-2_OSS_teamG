"""Blink biomarkers: rate, duration, completeness, per-eye asymmetry."""
import numpy as np

from facemesh.medical.blink import BlinkAnalytics, EyeBlinkDetector
from facemesh.metrics import EYE_LANDMARKS_LEFT, EYE_LANDMARKS_RIGHT

FPS = 60.0


class LM:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def _set_eye(pts, idx, openness):
    p1, p2, p3, p4, p5, p6 = idx
    pts[p1] = [0.0, 0.0]
    pts[p4] = [0.10, 0.0]
    g = 0.03 * openness
    pts[p2] = [0.03, g]
    pts[p3] = [0.06, g]
    pts[p6] = [0.03, -g]
    pts[p5] = [0.06, -g]


def _face(open_l, open_r):
    pts = np.zeros((478, 2))
    _set_eye(pts, EYE_LANDMARKS_LEFT, open_l)
    _set_eye(pts, EYE_LANDMARKS_RIGHT, open_r)
    return [LM(x, y) for x, y in pts]


def _profile(t, period, dur, depth):
    return depth if (t % period) < dur else 1.0


def _warmup(obj, seconds=2.0, single_eye=False):
    t = 0.0
    for _ in range(int(FPS * seconds)):
        t += 1 / FPS
        if single_eye:
            obj.update(0.30, t)
        else:
            obj.update(_face(1.0, 1.0), 640, 480, t)
    return t


def test_complete_blink_detected():
    d = EyeBlinkDetector()
    t = _warmup(d, single_eye=True)
    for _ in range(int(FPS * 6)):
        t += 1 / FPS
        d.update(0.30 * _profile(t, 2.0, 0.2, 0.02), t)
    assert len(d.blinks) >= 2
    assert all(b.complete for b in d.blinks)


def test_incomplete_blink_flagged():
    d = EyeBlinkDetector()
    t = _warmup(d, single_eye=True)
    for _ in range(int(FPS * 6)):
        t += 1 / FPS
        d.update(0.30 * _profile(t, 2.0, 0.2, 0.55), t)
    assert len(d.blinks) >= 2
    assert not any(b.complete for b in d.blinks)


def test_blink_duration_measured():
    d = EyeBlinkDetector()
    t = _warmup(d, single_eye=True)
    for _ in range(int(FPS * 6)):
        t += 1 / FPS
        d.update(0.30 * _profile(t, 2.0, 0.25, 0.02), t)
    mean = np.mean([b.duration_s for b in d.blinks])
    assert abs(mean - 0.25) < 0.06


def test_artifacts_outside_physiological_bounds_rejected():
    """A 2-second 'closure' is not a blink."""
    d = EyeBlinkDetector()
    t = _warmup(d, single_eye=True)
    for _ in range(int(FPS * 2)):
        t += 1 / FPS
        d.update(0.02, t)          # eyes shut for 2s
    for _ in range(int(FPS * 1)):
        t += 1 / FPS
        d.update(0.30, t)
    assert len(d.blinks) == 0


def test_blink_rate_over_window():
    ba = BlinkAnalytics(window_s=60.0)
    t = _warmup(ba)
    out = None
    for _ in range(int(FPS * 30)):
        t += 1 / FPS
        o = _profile(t, 3.0, 0.2, 0.02)     # 20/min
        out = ba.update(_face(o, o), 640, 480, t)
    assert 15 <= out["rate_per_min"] <= 25
    assert not out["low_rate"]


def test_low_rate_flagged():
    ba = BlinkAnalytics(window_s=60.0)
    t = _warmup(ba)
    out = None
    for _ in range(int(FPS * 40)):
        t += 1 / FPS
        o = _profile(t, 12.0, 0.2, 0.02)    # ~5/min
        out = ba.update(_face(o, o), 640, 480, t)
    assert out["low_rate"]


def test_high_incomplete_fraction_flagged():
    ba = BlinkAnalytics(window_s=60.0)
    t = _warmup(ba)
    out = None
    for _ in range(int(FPS * 30)):
        t += 1 / FPS
        o = _profile(t, 2.0, 0.2, 0.55)
        out = ba.update(_face(o, o), 640, 480, t)
    assert out["incomplete_fraction"] > 0.4 and out["high_incomplete"]


def test_per_eye_closure_asymmetry():
    """Unilateral weak closure must identify the weaker eye."""
    ba = BlinkAnalytics(window_s=60.0)
    t = _warmup(ba)
    out = None
    for _ in range(int(FPS * 30)):
        t += 1 / FPS
        phase = t % 2.0
        ol = 0.60 if phase < 0.2 else 1.0   # left closes weakly
        orr = 0.02 if phase < 0.2 else 1.0  # right closes fully
        out = ba.update(_face(ol, orr), 640, 480, t)
    assert out["weaker_eye"] == "left"
    assert out["closure_asymmetry"] > 0.2


def test_interblink_interval():
    ba = BlinkAnalytics(window_s=60.0)
    t = _warmup(ba)
    out = None
    for _ in range(int(FPS * 20)):
        t += 1 / FPS
        o = _profile(t, 4.0, 0.2, 0.02)
        out = ba.update(_face(o, o), 640, 480, t)
    assert abs(out["mean_interblink_s"] - 4.0) < 0.3
