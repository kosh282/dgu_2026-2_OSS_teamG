"""Harmonic-locking regression tests (issue #1).

A real session from a subject with a resting rate of 46 bpm was reported as
83-84 bpm. These tests reproduce that and pin the fix.
"""
import numpy as np
import pytest

from facemesh.medical.rppg import HARMONIC_RATIO, MIN_BPM, estimate_bpm, pos_signal


def _pulse_rgb(bpm, seconds=10, fps=30.0, harmonic=0.6, seed=1):
    """A physiological pulse: sharp systolic peak -> strong 2nd harmonic."""
    n = int(seconds * fps)
    t = np.arange(n) / fps
    f = bpm / 60.0
    wave = np.sin(2 * np.pi * f * t) + harmonic * np.sin(2 * np.pi * 2 * f * t + 0.4)
    rng = np.random.default_rng(seed)
    drift = 2 * np.sin(2 * np.pi * 0.05 * t)
    R = 180 + 0.3 * wave + rng.normal(0, 0.4, n) + drift
    G = 120 + 1.2 * wave + rng.normal(0, 0.4, n) + drift
    B = 100 + 0.5 * wave + rng.normal(0, 0.4, n) + drift
    return np.stack([R, G, B], axis=1), fps


def test_reproduces_the_bug_when_correction_disabled():
    """Without the fix, a 46 bpm pulse reads as ~92 -- confidently."""
    rgb, fps = _pulse_rgb(46, harmonic=1.0)
    bpm, conf = estimate_bpm(pos_signal(rgb), fps, correct_harmonic=False)
    assert bpm > 85                # locked onto the harmonic
    assert conf > 0.4              # and it was confident about it


def test_bradycardia_recovered_with_correction():
    """Anastasija's case: true 46 bpm."""
    rgb, fps = _pulse_rgb(46, harmonic=1.0)
    bpm, _ = estimate_bpm(pos_signal(rgb), fps)
    assert abs(bpm - 46) < 5


@pytest.mark.parametrize("true_bpm", [42, 46, 50, 55, 60, 72, 96, 120, 150])
@pytest.mark.parametrize("harmonic", [0.6, 1.0, 1.4])
def test_all_rates_recovered_regardless_of_harmonic_strength(true_bpm, harmonic):
    rgb, fps = _pulse_rgb(true_bpm, harmonic=harmonic)
    bpm, _ = estimate_bpm(pos_signal(rgb), fps)
    assert abs(bpm - true_bpm) <= 5


def _pure_pulse(bpm, seconds=12, fps=30.0, harmonic=0.0, seed=3):
    n = int(seconds * fps)
    t = np.arange(n) / fps
    f = bpm / 60.0
    wave = np.sin(2 * np.pi * f * t) + harmonic * np.sin(2 * np.pi * 2 * f * t)
    rng = np.random.default_rng(seed)
    R = 180 + 0.3 * wave + rng.normal(0, 0.4, n)
    G = 120 + 1.2 * wave + rng.normal(0, 0.4, n)
    B = 100 + 0.5 * wave + rng.normal(0, 0.4, n)
    return np.stack([R, G, B], axis=1), fps


@pytest.mark.parametrize("true_bpm", [84, 92, 100, 120, 150, 170])
def test_genuine_high_rates_are_not_halved(true_bpm):
    """The fix must not turn a real tachycardia into a fake bradycardia."""
    rgb, fps = _pure_pulse(true_bpm)
    bpm, _ = estimate_bpm(pos_signal(rgb), fps)
    assert abs(bpm - true_bpm) <= 5


def test_min_bpm_lowered_so_fundamental_is_not_at_the_edge():
    assert MIN_BPM <= 40.0


def test_harmonic_ratio_is_a_sane_threshold():
    assert 0.1 <= HARMONIC_RATIO <= 0.5


def test_half_below_min_bpm_is_not_selected():
    """A 72 bpm pulse must not be 'corrected' to 36 just because MIN_BPM allows it."""
    rgb, fps = _pure_pulse(72, harmonic=0.0)
    bpm, _ = estimate_bpm(pos_signal(rgb), fps)
    assert abs(bpm - 72) <= 5


def test_short_signal_still_rejected():
    assert estimate_bpm(np.zeros(10), 30.0) == (0.0, 0.0)
