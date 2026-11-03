"""PLR trust gating (issue #5).

A live session reported latencies of 1.033s (left) and 1.536s (right) with
`responded=True`. Real PLR latency is ~0.2-0.3s and the two eyes respond
near-simultaneously. The module was measuring landmark jitter, not a reflex.
"""
import numpy as np
import pytest

from facemesh.medical.plr import (
    LATENCY_MAX_S, LATENCY_MIN_S, MAX_LATENCY_DIFFERENCE_S, MIN_SNR, PLRSession,
)

FPS = 60.0


def _run(fl, fr, baseline=2.0, flash=0.5, recovery=4.0):
    s = PLRSession(baseline_s=baseline, flash_s=flash, recovery_s=recovery)
    t = 0.0
    while not s.finished:
        t += 1 / FPS
        s.update(t, fl(t), fr(t))
    return s.result


def _real(rng, base=0.12, amp=0.30, lat=0.25, onset=2.0, tc=0.25, tr=1.5, noise=0.001):
    def f(t):
        if t < onset + lat:
            v = base
        else:
            dt = t - onset - lat
            c = amp * (1 - np.exp(-dt / tc))
            r = c * np.exp(-max(dt - 0.4, 0) / tr)
            v = base * (1 - (r if dt > 0.4 else c))
        return v + rng.normal(0, noise)
    return f


@pytest.mark.parametrize("sigma", [0.002, 0.006, 0.012])
def test_pure_noise_is_never_a_reflex(sigma):
    """The bug: over N samples the minimum of noise sits ~3-4 sigma low, so a
    plain 3-sigma test always passed. Normalize by the expected extreme."""
    rng = np.random.default_rng(0)
    flat = lambda t: 0.12 + rng.normal(0, sigma)   # noqa: E731
    r = _run(flat, flat)
    assert not r["left"]["responded"]
    assert not r["measurement_trustworthy"]
    assert "constriction within baseline noise" in r["left"]["reasons"]


def test_noise_snr_is_around_one_regardless_of_amplitude():
    snrs = []
    for sigma in (0.002, 0.006, 0.012):
        rng = np.random.default_rng(0)
        flat = lambda t: 0.12 + rng.normal(0, sigma)   # noqa: E731
        snrs.append(_run(flat, flat)["left"]["snr"])
    assert all(s < MIN_SNR for s in snrs)
    assert max(snrs) - min(snrs) < 0.5     # scale-free, as it should be


def test_genuine_reflex_is_accepted():
    rng = np.random.default_rng(1)
    r = _run(_real(rng), _real(rng))
    assert r["left"]["responded"] and r["measurement_trustworthy"]
    assert LATENCY_MIN_S <= r["left"]["latency_s"] <= LATENCY_MAX_S


def test_implausible_latency_rejected():
    """1.03s was the real reported value; physiological is 0.2-0.3s."""
    rng = np.random.default_rng(2)
    r = _run(_real(rng, lat=1.03), _real(rng, lat=1.03))
    assert not r["left"]["responded"]
    assert any("outside physiological range" in x for x in r["left"]["reasons"])
    assert r["left"]["latency_s"] is None


def test_inter_eye_latency_difference_flags_measurement_failure():
    """Direct and consensual responses are near-simultaneous."""
    rng = np.random.default_rng(3)
    r = _run(_real(rng, lat=0.20), _real(rng, lat=0.55))
    assert r["latency_difference_s"] > MAX_LATENCY_DIFFERENCE_S
    assert not r["both_responded"]
    assert not r["measurement_trustworthy"]
    assert any("near-simultaneous" in w for w in r["warnings"])


def test_genuine_amplitude_asymmetry_still_reported():
    """A real one-sided weakness (same latency) must survive the gating."""
    rng = np.random.default_rng(4)
    r = _run(_real(rng, amp=0.30), _real(rng, amp=0.12))
    assert r["weaker_response"] == "right"
    assert r["measurement_trustworthy"]


def test_untrustworthy_result_explains_itself():
    rng = np.random.default_rng(5)
    flat = lambda t: 0.12 + rng.normal(0, 0.006)   # noqa: E731
    r = _run(flat, flat)
    assert "interpretation" in r
    assert "cannot resolve your pupil" in r["interpretation"]


def test_trustworthy_result_has_no_interpretation_note():
    rng = np.random.default_rng(6)
    r = _run(_real(rng), _real(rng))
    assert "interpretation" not in r
