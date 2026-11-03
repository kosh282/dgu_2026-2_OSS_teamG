from facemesh.drowsiness import DrowsinessMonitor


def _run(monitor, ear, mar, seconds, t0, fps=30):
    t = t0
    last = None
    for _ in range(int(seconds * fps)):
        t += 1 / fps
        last = monitor.update(ear, mar, t)
    return last, t


def test_awake_when_eyes_open():
    dm = DrowsinessMonitor(window_s=10)
    out, _ = _run(dm, ear=0.30, mar=0.1, seconds=3, t0=0)
    assert out["level"] == "awake"
    assert out["perclos"] == 0.0


def test_microsleep_triggers_alert():
    dm = DrowsinessMonitor(window_s=10, microsleep_s=1.0)
    _run(dm, 0.30, 0.1, 2, 0)
    out, _ = _run(dm, 0.05, 0.1, 1.5, 2)
    assert out["microsleep"] is True
    assert out["level"] == "ALERT"


def test_yawn_counted():
    dm = DrowsinessMonitor(window_s=10, mar_threshold=0.6, yawn_min_s=0.4)
    _run(dm, 0.30, 0.1, 1, 0)
    _run(dm, 0.30, 0.8, 0.7, 1)          # sustained open mouth
    out, _ = _run(dm, 0.30, 0.1, 0.3, 2)  # close -> yawn registered
    assert out["yawns"] >= 1


def test_perclos_in_range():
    dm = DrowsinessMonitor(window_s=5)
    out, _ = _run(dm, 0.10, 0.1, 3, 0)
    assert 0.0 <= out["perclos"] <= 1.0
