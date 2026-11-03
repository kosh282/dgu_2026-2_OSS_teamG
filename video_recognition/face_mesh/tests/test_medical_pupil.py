"""Gaze dynamics: saccades, fixation, nystagmus screen, pupil proxy."""
import math

import numpy as np

from facemesh.medical.pupil import GazeDynamics, detect_oscillation
from facemesh.metrics import (
    LEFT_EYE_CORNERS, LEFT_EYE_LIDS, LEFT_IRIS,
    RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS, RIGHT_IRIS,
)

FPS = 60.0


class LM:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def _face(h, v, iris_scale=1.0, left_scale=None):
    """Two eyes at distinct positions (33 / 263 are the outer corners)."""
    pts = np.zeros((478, 2))
    layouts = [
        (RIGHT_IRIS, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS, 0.34, 0.46, iris_scale),
        (LEFT_IRIS, LEFT_EYE_CORNERS, LEFT_EYE_LIDS, 0.54, 0.66,
         left_scale if left_scale is not None else iris_scale),
    ]
    for ring, corners, lids, x0, x1, scale in layouts:
        yt, yb = 0.40, 0.50
        pts[corners[0]] = [x0, 0.45]
        pts[corners[1]] = [x1, 0.45]
        pts[lids[0]] = [(x0 + x1) / 2, yt]
        pts[lids[1]] = [(x0 + x1) / 2, yb]
        cx, cy = x0 + h * (x1 - x0), yt + v * (yb - yt)
        r = 0.012 * scale
        for k, i in enumerate(ring):
            a = 2 * math.pi * k / len(ring)
            pts[i] = [cx + r * math.cos(a), cy + r * math.sin(a)]
    return [LM(x, y) for x, y in pts]


def test_oscillation_detector_isolates_band():
    t = np.arange(200) / FPS
    r_fast, hz = detect_oscillation(np.sin(2 * np.pi * 5 * t), FPS)
    r_slow, _ = detect_oscillation(np.sin(2 * np.pi * 0.4 * t), FPS)
    assert r_fast > 0.5 and 4.0 <= hz <= 6.0
    assert r_slow < 0.5


def test_saccade_detected_with_amplitude():
    gd = GazeDynamics(window_s=5.0)
    t, out = 0.0, None
    for _ in range(int(FPS)):
        t += 1 / FPS
        out = gd.update(_face(0.30, 0.5), 640, 480, t)
    for i in range(3):
        t += 1 / FPS
        out = gd.update(_face(0.30 + 0.15 * (i + 1), 0.5), 640, 480, t)
    for _ in range(int(FPS)):
        t += 1 / FPS
        out = gd.update(_face(0.75, 0.5), 640, 480, t)
    assert out["saccade_count"] >= 1
    assert out["mean_saccade_amplitude"] > 0.1
    assert out["mean_saccade_speed"] > 0


def test_steady_fixation_has_no_saccades():
    gd = GazeDynamics(window_s=5.0)
    rng = np.random.default_rng(1)
    t, out = 0.0, None
    for _ in range(int(FPS * 3)):
        t += 1 / FPS
        out = gd.update(_face(0.5 + rng.normal(0, 0.002),
                              0.5 + rng.normal(0, 0.002)), 640, 480, t)
    assert out["saccade_count"] == 0
    assert out["fixation_stability"] < 0.02


def test_unstable_fixation_has_larger_spread():
    gd = GazeDynamics(window_s=5.0)
    rng = np.random.default_rng(2)
    t, out = 0.0, None
    for _ in range(int(FPS * 3)):
        t += 1 / FPS
        out = gd.update(_face(0.5 + rng.normal(0, 0.03),
                              0.5 + rng.normal(0, 0.03)), 640, 480, t)
    assert out["fixation_stability"] > 0.02


def test_nystagmus_suspected_on_oscillation():
    gd = GazeDynamics(window_s=3.0)
    t, out = 0.0, None
    for _ in range(int(FPS * 3)):
        t += 1 / FPS
        h = 0.5 + 0.05 * math.sin(2 * math.pi * 5.0 * t)
        out = gd.update(_face(h, 0.5), 640, 480, t)
    assert out["nystagmus_suspected"] and 4.0 <= out["nystagmus_hz"] <= 6.0


def test_steady_gaze_does_not_trigger_nystagmus():
    gd = GazeDynamics(window_s=3.0)
    rng = np.random.default_rng(3)
    t, out = 0.0, None
    for _ in range(int(FPS * 3)):
        t += 1 / FPS
        out = gd.update(_face(0.5 + rng.normal(0, 0.001), 0.5), 640, 480, t)
    assert not out["nystagmus_suspected"]


def test_pupil_proxy_scales_with_iris():
    small = GazeDynamics().update(_face(0.5, 0.5, iris_scale=1.0), 640, 480, 0.1)
    big = GazeDynamics().update(_face(0.5, 0.5, iris_scale=1.5), 640, 480, 0.1)
    assert big["pupil_proxy_left"] > small["pupil_proxy_left"] * 1.3


def test_pupil_asymmetry_detected():
    out = GazeDynamics().update(_face(0.5, 0.5, iris_scale=1.0, left_scale=1.6), 640, 480, 0.1)
    assert out["pupil_asymmetry"] > 0.01
    assert out["pupil_proxy_left"] > out["pupil_proxy_right"]
