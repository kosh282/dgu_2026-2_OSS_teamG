"""Guided task protocol: sequencing, gating, scoring."""
import pytest

from facemesh.medical.protocol import DEFAULT_PROTOCOL, Task, TaskProtocol


def sym(a, region="mouth"):
    return {"asymmetry": a, "symmetry": max(0.0, 1 - a / 0.15), "worst_region": region}


def mov(e):
    return {"expressivity": e}


def qual(ok=True, reasons=None):
    return {"ok": ok, "score": 1.0 if ok else 0.4, "reasons": reasons or []}


def _run(protocol, asym_for, expr_for, quality=None, fps=30.0, limit=60.0):
    t = 0.0
    while not protocol.finished and t < limit:
        t += 1 / fps
        name = protocol.current.name if protocol.current else None
        protocol.update(t, sym(asym_for(name)), mov(expr_for(name)),
                        quality() if quality else qual())
    return protocol


def test_visits_every_task_in_order():
    p = TaskProtocol()
    seen = []
    t = 0.0
    while not p.finished and t < 60:
        t += 1 / 30
        st = p.update(t, sym(0.02), mov(0.5), qual())
        if st["task"] and st["task"] not in seen:
            seen.append(st["task"])
    assert seen == [t.name for t in DEFAULT_PROTOCOL]
    assert p.finished


def test_worst_task_identified():
    p = _run(TaskProtocol(),
             lambda n: 0.09 if n == "smile" else (0.01 if n == "rest" else 0.03),
             lambda n: 0.05 if n == "rest" else 0.8)
    s = p.summary()["summary"]
    assert s["worst_task"] == "smile"
    assert s["worst_task_region"] == "mouth"


def test_movement_excess_asymmetry():
    """Asymmetry that only appears on movement is the interesting signal."""
    p = _run(TaskProtocol(),
             lambda n: 0.09 if n == "smile" else (0.01 if n == "rest" else 0.02),
             lambda n: 0.8)
    s = p.summary()["summary"]
    assert s["rest_asymmetry"] == pytest.approx(0.01, abs=1e-6)
    assert s["movement_excess_asymmetry"] > 0.07


def test_settle_period_not_recorded():
    """Frames during the settle window must not contribute measurements."""
    task = Task("only", "do a thing", seconds=2.0, settle_s=1.0)
    p = TaskProtocol([task])
    t = 0.0
    for _ in range(15):                 # 0.5s, all inside settle
        t += 1 / 30
        p.update(t, sym(0.5), mov(1.0), qual())
    assert p.results["only"].frames == 0


def test_bad_quality_frames_skipped_not_counted():
    task = Task("only", "do a thing", seconds=3.0, settle_s=0.0)
    p = TaskProtocol([task])
    t = 0.0
    for i in range(60):
        t += 1 / 30
        ok = i % 2 == 0
        p.update(t, sym(0.05), mov(0.5), qual(ok, [] if ok else ["too dark"]))
    r = p.results["only"]
    assert r.frames > 0 and r.skipped_frames > 0
    assert r.frames + r.skipped_frames == 60


def test_blocked_state_surfaces_reason():
    p = TaskProtocol([Task("only", "x", seconds=3.0, settle_s=0.0)])
    st = p.update(0.5, sym(0.02), mov(0.5), qual(False, ["too dark"]))
    assert st["blocked"] and "too dark" in st["reason"]
    assert st["recording"] is False


def test_skip_advances():
    p = TaskProtocol()
    first = p.current.name
    p.skip()
    assert p.current.name != first


def test_peak_not_mean_is_reported():
    """A transient asymmetry spike must be captured by the peak."""
    task = Task("smile", "smile", seconds=3.0, settle_s=0.0)
    p = TaskProtocol([task])
    t = 0.0
    for i in range(90):
        t += 1 / 30
        a = 0.10 if i == 45 else 0.01     # one spike
        p.update(t, sym(a), mov(0.5), qual())
    s = p.summary()["tasks"]["smile"]
    assert s["peak_asymmetry"] == pytest.approx(0.10, abs=1e-6)


def test_summary_finalizes_in_progress_task():
    """Quitting mid-task must not silently discard that task's measurements."""
    p = TaskProtocol([Task("smile", "smile", seconds=10.0, settle_s=0.0)])
    t = 0.0
    for _ in range(30):                 # only 1s of a 10s task
        t += 1 / 30
        p.update(t, sym(0.07), mov(0.6), qual())
    assert not p.finished
    s = p.summary()["tasks"]["smile"]
    assert s["frames"] == 30 and s["peak_asymmetry"] == pytest.approx(0.07, abs=1e-6)


def test_summary_shape():
    p = _run(TaskProtocol(), lambda n: 0.02, lambda n: 0.5)
    s = p.summary()
    assert set(s) == {"tasks", "summary"}
    assert s["summary"]["tasks_completed"] == len(DEFAULT_PROTOCOL)
    for r in s["tasks"].values():
        assert {"task", "frames", "peak_asymmetry", "peak_expressivity"} <= set(r)
