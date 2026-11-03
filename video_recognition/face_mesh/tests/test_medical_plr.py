"""Pupillary light reflex: phases, feature extraction, deficits."""
import numpy as np
import pytest

from facemesh.medical.plr import MIN_RESPONSE, PLRSession, PLRTrace, analyze_plr

FPS = 60.0


def _pupil(t, onset, baseline=0.12, amp=0.30, latency=0.25, tau_c=0.25, tau_r=1.5):
    """Flat, then latency, fast constriction, slow redilation."""
    if t < onset + latency:
        return baseline
    dt = t - onset - latency
    constrict = amp * (1 - np.exp(-dt / tau_c))
    redilate = constrict * np.exp(-max(dt - 0.4, 0) / tau_r)
    return baseline * (1 - (redilate if dt > 0.4 else constrict))


def _run(amp_left=0.30, amp_right=0.30, latency=0.25, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    s = PLRSession(baseline_s=2.0, flash_s=0.5, recovery_s=4.0)
    t = 0.0
    while not s.finished:
        t += 1 / FPS
        dl = _pupil(t, 2.0, amp=amp_left, latency=latency) + rng.normal(0, noise)
        dr = _pupil(t, 2.0, amp=amp_right, latency=latency) + rng.normal(0, noise)
        s.update(t, dl, dr)
    return s.result


def test_recovers_known_amplitude():
    r = _run(amp_left=0.30, amp_right=0.30)
    assert r["valid"]
    assert abs(r["left"]["constriction_pct"] - 30) < 7
    assert r["left"]["responded"]


def test_recovers_known_latency():
    r = _run(latency=0.25)
    assert abs(r["left"]["latency_s"] - 0.25) < 0.08


def test_no_response_flagged():
    """A flat trace must not be reported as a reflex."""
    s = PLRSession(baseline_s=2.0, flash_s=0.5, recovery_s=3.0)
    rng = np.random.default_rng(1)
    t = 0.0
    while not s.finished:
        t += 1 / FPS
        d = 0.12 + rng.normal(0, 0.0005)
        s.update(t, d, d)
    assert not s.result["left"]["responded"]
    assert s.result["left"]["constriction_pct"] < MIN_RESPONSE * 100


def test_asymmetric_response_identifies_weaker_eye():
    r = _run(amp_left=0.30, amp_right=0.08)
    assert r["weaker_response"] == "right"
    assert r["response_asymmetry_pct"] > 10
    assert not r["both_responded"] or r["right"]["constriction_pct"] < r["left"]["constriction_pct"]


def test_peak_velocity_is_negative_during_constriction():
    r = _run()
    assert r["left"]["peak_velocity_per_s"] < 0


def test_recovery_time_measured():
    r = _run()
    assert r["left"]["recovery_75_s"] is not None
    assert r["left"]["recovery_75_s"] > 0


def test_phases_sequence_in_order():
    s = PLRSession(baseline_s=1.0, flash_s=0.5, recovery_s=1.0)
    t, phases = 0.0, []
    while not s.finished:
        t += 1 / FPS
        st = s.update(t, 0.12, 0.12)
        if st["phase"] not in phases:
            phases.append(st["phase"])
    assert phases[:3] == ["baseline", "flash", "recovery"]


def test_flash_onset_recorded_once():
    s = PLRSession(baseline_s=1.0, flash_s=0.5, recovery_s=1.0)
    t = 0.0
    while not s.finished:
        t += 1 / FPS
        s.update(t, 0.12, 0.12)
    assert s.trace.flash_onset == pytest.approx(1.0, abs=0.05)


def test_zero_diameter_samples_are_dropped():
    """Frames with no face must not enter the trace."""
    s = PLRSession(baseline_s=0.5, flash_s=0.2, recovery_s=0.5)
    t = 0.0
    while not s.finished:
        t += 1 / FPS
        d = 0.0 if int(t * FPS) % 2 == 0 else 0.12
        s.update(t, d, d)
    assert all(x > 0 for x in s.trace.left)


def test_analyze_without_flash_is_invalid():
    assert analyze_plr(PLRTrace())["reason"] == "flash never fired"


def test_analyze_with_too_few_samples_is_invalid():
    tr = PLRTrace()
    tr.flash_onset = 1.0
    for i in range(5):
        tr.add(i / 10, 0.1, 0.1)
    assert analyze_plr(tr)["reason"] == "too few samples"
