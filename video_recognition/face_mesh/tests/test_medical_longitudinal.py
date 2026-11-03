"""Longitudinal session comparison."""
import json

import numpy as np
import pytest

from facemesh.medical.longitudinal import (
    compare, compare_many, summarize_session, trend,
)


def _write(tmp_path, name, sym_mu, n=240, phi=0.8, bad=0, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 0.01, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    vals = x + sym_mu
    lines = []
    for i, v in enumerate(vals):
        isbad = i < bad
        med = {"quality": {"ok": not isbad, "score": 0.4 if isbad else 1.0,
                           "reasons": ["too dark"] if isbad else []}}
        if not isbad:
            med["symmetry"] = {"symmetry": float(v), "asymmetry": float(1 - v),
                               "worst_region": "mouth"}
            med["blink"] = {"rate_per_min": 18.0, "incomplete_fraction": 0.1}
        lines.append(json.dumps({"frame": i, "faces": [{"medical": med}]}))
    p = tmp_path / f"{name}.jsonl"
    p.write_text("\n".join(lines))
    return p


def test_quality_gated_frames_excluded(tmp_path):
    s = summarize_session(_write(tmp_path, "g", 0.90, n=200, bad=50), n_boot=200)
    assert s.frames == 200 and s.usable_frames == 150
    assert s.usable_fraction == pytest.approx(0.75)


def test_identical_sessions_are_stable(tmp_path):
    """Noise must not be reported as change."""
    a = summarize_session(_write(tmp_path, "a", 0.90, seed=1), n_boot=400)
    b = summarize_session(_write(tmp_path, "b", 0.90, seed=2), n_boot=400)
    assert compare(a, b)["symmetry"]["verdict"] == "stable"


def test_real_deterioration_detected(tmp_path):
    a = summarize_session(_write(tmp_path, "t1", 0.92, seed=3), n_boot=400)
    b = summarize_session(_write(tmp_path, "t2", 0.70, seed=4), n_boot=400)
    c = compare(a, b)["symmetry"]
    assert c["verdict"] == "worsened" and c["intervals_separated"]
    assert c["delta"] < 0


def test_real_improvement_detected(tmp_path):
    a = summarize_session(_write(tmp_path, "u1", 0.70, seed=5), n_boot=400)
    b = summarize_session(_write(tmp_path, "u2", 0.92, seed=6), n_boot=400)
    assert compare(a, b)["symmetry"]["verdict"] == "improved"


def test_lower_is_better_metric_inverts(tmp_path):
    """Asymmetry falling is an improvement, not a worsening."""
    a = summarize_session(_write(tmp_path, "v1", 0.70, seed=7), n_boot=400)
    b = summarize_session(_write(tmp_path, "v2", 0.92, seed=8), n_boot=400)
    assert compare(a, b)["asymmetry"]["verdict"] == "improved"


def test_trend_across_sessions(tmp_path):
    paths = [_write(tmp_path, f"s{i}", 0.95 - 0.06 * i, seed=10 + i) for i in range(4)]
    res = compare_many(paths, n_boot=200)
    t = res["trends"]["symmetry"]
    assert t["slope_per_session"] < -0.03
    assert t["r_squared"] > 0.9
    assert t["sessions"] == 4


def test_trend_needs_three_sessions(tmp_path):
    a = summarize_session(_write(tmp_path, "a", 0.9, seed=1), n_boot=100)
    b = summarize_session(_write(tmp_path, "b", 0.9, seed=2), n_boot=100)
    assert trend([a, b], "symmetry") is None


def test_compare_many_shape(tmp_path):
    paths = [_write(tmp_path, f"m{i}", 0.9, seed=i) for i in range(3)]
    res = compare_many(paths, n_boot=100)
    assert len(res["sessions"]) == 3
    assert len(res["comparisons"]) == 2       # consecutive pairs
    assert "symmetry" in res["trends"]


def test_intervals_bracket_estimates(tmp_path):
    s = summarize_session(_write(tmp_path, "x", 0.88, seed=4), n_boot=300)
    i = s.metrics["symmetry"]
    assert i.low <= i.estimate <= i.high
