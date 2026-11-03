"""Lighting rejection in HeartRateEstimator (issue #3)."""
import numpy as np

from facemesh.medical.rppg import (
    FOREHEAD, LEFT_CHEEK, RIGHT_CHEEK, HeartRateEstimator, lighting_quality,
)

SKIN = (140, 160, 205)


class LM:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def _lms():
    l = [LM(0.5, 0.5) for _ in range(478)]
    for k, i in enumerate(FOREHEAD):
        l[i] = LM(0.35 + 0.03 * k, 0.20 + 0.01 * (k % 2))
    for k, i in enumerate(LEFT_CHEEK):
        l[i] = LM(0.58 + 0.02 * k, 0.50 + 0.01 * (k % 2))
    for k, i in enumerate(RIGHT_CHEEK):
        l[i] = LM(0.25 + 0.02 * k, 0.50 + 0.01 * (k % 2))
    l[33] = LM(0.40, 0.40)
    l[263] = LM(0.60, 0.40)
    return l


L = _lms()


def _frame(val=None, left=None, right=None):
    f = np.full((480, 640, 3), SKIN, np.uint8)
    if val is not None:
        f[:, :] = val
    if left is not None:
        f[:, :320] = left
    if right is not None:
        f[:, 320:] = right
    return f


def test_lighting_quality_detects_imbalance():
    good = lighting_quality(_frame(), L)
    uneven = lighting_quality(_frame(left=(60, 70, 90), right=(200, 220, 250)), L)
    assert uneven["imbalance"] > good["imbalance"]
    assert good["imbalance"] < 0.05


def test_lighting_quality_measured_on_skin_not_bounding_box():
    q = lighting_quality(_frame(), L)
    assert q["pixels"] > 100          # masked skin pixels, not the whole box


def test_even_lighting_is_not_rejected():
    e = HeartRateEstimator(window_s=10, min_seconds=3)
    for k in range(60):
        e.update(_frame(), L, k / 30)
    assert e.rejected_lighting == 0


def test_uneven_lighting_is_rejected():
    """406 of 931 frames in the real session were unevenly lit and used anyway."""
    e = HeartRateEstimator(window_s=10, min_seconds=3)
    out = None
    for k in range(60):
        out = e.update(_frame(left=(60, 70, 90), right=(200, 220, 250)), L, k / 30)
    assert e.rejected_lighting > 50
    assert out["lighting_rejected"]
    assert "uneven lighting" in out["lighting_reasons"]


def test_dark_and_overexposed_frames_rejected():
    for val in [(10, 12, 15), (252, 253, 254)]:
        e = HeartRateEstimator(window_s=10, min_seconds=3)
        for k in range(30):
            e.update(_frame(val=val), L, k / 30)
        assert e.rejected_lighting > 20


def test_sudden_illumination_change_rejected():
    e = HeartRateEstimator(window_s=10, min_seconds=3)
    for k in range(40):
        val = SKIN if k < 20 else (200, 215, 240)
        e.update(_frame(val=val), L, k / 30)
    assert e.rejected_lighting >= 1


def test_motion_and_lighting_counted_separately():
    e = HeartRateEstimator(window_s=10, min_seconds=3, max_motion=0.6)
    for k in range(40):
        shift = 0.15 * (k % 2)
        e.update(_frame(), [LM(l.x + shift, l.y) for l in L], k / 30)
    assert e.rejected_motion > 0
    assert e.rejected == e.rejected_motion + e.rejected_lighting


def test_rejected_frames_never_enter_the_window():
    e = HeartRateEstimator(window_s=10, min_seconds=3)
    for k in range(60):
        e.update(_frame(left=(60, 70, 90), right=(200, 220, 250)), L, k / 30)
    assert len(e._t) == 0            # nothing usable was collected
