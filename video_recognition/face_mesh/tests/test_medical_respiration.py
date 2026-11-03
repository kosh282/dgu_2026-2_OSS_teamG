"""Respiration subharmonic/drift regression tests (issue #4).

A live session reported 8 breaths/min for a healthy adult at rest (normal:
12-20). Two causes: postural/lighting wander around 0.13 Hz (~8/min) sitting
inside a naive 6-30/min band, and genuine breathing subharmonics.
"""
import numpy as np
import pytest

from facemesh.medical.rppg import (
    MAX_RPM, MIN_RPM, RESP_HIGHPASS_RPM, estimate_respiration, highpass,
)

FPS = 30.0


def _envelope(rpm, seconds=20, wander=0.0, subh=0.0, seed=2):
    n = int(seconds * FPS)
    t = np.arange(n) / FPS
    pulse = np.sin(2 * np.pi * (72 / 60) * t)
    breath = 0.8 * np.sin(2 * np.pi * (rpm / 60) * t)
    slow = wander * np.sin(2 * np.pi * 0.13 * t + 0.5)     # ~7.8/min wander
    sub = subh * np.sin(2 * np.pi * (rpm / 120) * t)       # subharmonic
    rng = np.random.default_rng(seed)
    return pulse + breath + slow + sub + rng.normal(0, 0.05, n)


def _naive_peak(signal, lo=6.0, hi=30.0):
    """The old approach: raw FFT peak in a 6-30/min band, no filtering."""
    n = len(signal)
    w = (signal - signal.mean()) * np.hanning(n)
    spec = np.abs(np.fft.rfft(w, n=2048)) ** 2
    freqs = np.fft.rfftfreq(2048, d=1 / FPS) * 60
    band = (freqs >= lo) & (freqs <= hi)
    return float(freqs[band][np.argmax(spec[band])])


def test_reproduces_the_bug_with_a_naive_peak():
    """Without filtering, a 16/min breather with wander reads as ~8-10/min."""
    naive = _naive_peak(_envelope(16, wander=1.5))
    assert naive < 11.0


def test_wander_no_longer_wins():
    rpm, _ = estimate_respiration(_envelope(16, wander=1.5), FPS)
    assert abs(rpm - 16) <= 3


def test_subharmonic_corrected():
    rpm, _ = estimate_respiration(_envelope(20, subh=1.2), FPS)
    assert abs(rpm - 20) <= 3


@pytest.mark.parametrize("true_rpm", [10, 12, 14, 16, 18, 20, 24])
@pytest.mark.parametrize("wander,subh", [(0, 0), (1.5, 0), (0, 1.2), (1.5, 1.2)])
def test_all_rates_recovered(true_rpm, wander, subh):
    rpm, _ = estimate_respiration(_envelope(true_rpm, wander=wander, subh=subh), FPS)
    assert abs(rpm - true_rpm) <= 3


def test_highpass_attenuates_wander_but_keeps_breathing():
    t = np.arange(600) / FPS

    def gain(freq_rpm):
        x = np.sin(2 * np.pi * (freq_rpm / 60) * t)
        y = highpass(x, FPS, RESP_HIGHPASS_RPM)
        return float(np.sqrt((y[100:-100] ** 2).mean()) /
                     np.sqrt((x[100:-100] ** 2).mean()))

    assert gain(7.8) < 0.15           # wander crushed
    assert gain(16.0) > 0.5           # breathing preserved
    assert gain(20.0) > gain(12.0)    # monotonic


def test_highpass_is_zero_phase():
    """Forward-backward filtering must not shift the signal in time."""
    t = np.arange(600) / FPS
    x = np.sin(2 * np.pi * (18 / 60) * t)
    y = highpass(x, FPS, RESP_HIGHPASS_RPM)
    mid = slice(150, 450)
    # peak alignment: correlation is maximal at zero lag
    corr = np.correlate(y[mid] / np.abs(y[mid]).max(), x[mid], mode="same")
    assert int(np.argmax(corr)) == len(corr) // 2


def test_band_starts_above_wander():
    assert MIN_RPM >= 9.0 and MAX_RPM <= 30.0


def test_short_signal_rejected():
    assert estimate_respiration(np.zeros(10), FPS) == (0.0, 0.0)
