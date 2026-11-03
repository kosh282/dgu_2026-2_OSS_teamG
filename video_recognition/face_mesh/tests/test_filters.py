import random
import statistics

from facemesh.filters import OneEuroFilter, PoseSmoother


def test_one_euro_reduces_noise():
    random.seed(0)
    f = OneEuroFilter(freq=30, min_cutoff=0.5, beta=0.01)
    raw, filt, t = [], [], 0.0
    for _ in range(60):
        t += 1 / 30
        x = 10.0 + random.gauss(0, 2.0)
        raw.append(x)
        filt.append(f(x, t))
    assert statistics.pstdev(filt[5:]) < statistics.pstdev(raw[5:])


def test_one_euro_tracks_step():
    f = OneEuroFilter(freq=30, min_cutoff=1.0, beta=0.5)
    t, y = 0.0, 0.0
    for _ in range(120):
        t += 1 / 30
        target = 0.0 if t < 2 else 40.0
        y = f(target, t)
    assert abs(y - 40.0) < 2.0


def test_pose_smoother_returns_triple():
    s = PoseSmoother()
    out = s.update(1.0, 2.0, 3.0, timestamp=0.1)
    assert len(out) == 3


def test_landmark_smoother_reduces_jitter():
    import numpy as np
    from facemesh.filters import LandmarkSmoother
    rng = np.random.default_rng(0)
    sm = LandmarkSmoother(min_cutoff=0.5, beta=0.01)
    base = rng.random((478, 3))
    t, raw_j, sm_j = 0.0, [], []
    prev_raw = prev_sm = None
    for _ in range(60):
        t += 1 / 30
        noisy = base + rng.normal(0, 0.01, base.shape)
        out = sm.apply(0, noisy, t)
        if prev_raw is not None:
            raw_j.append(np.abs(noisy - prev_raw).mean())
            sm_j.append(np.abs(out - prev_sm).mean())
        prev_raw, prev_sm = noisy, out
    assert np.mean(sm_j) < np.mean(raw_j) / 3


def test_landmark_smoother_tracks_motion():
    import numpy as np
    from facemesh.filters import LandmarkSmoother
    sm = LandmarkSmoother()
    base = np.zeros((10, 3))
    t = 0.0
    for _ in range(30):
        t += 1 / 30
        sm.apply(1, base, t)
    moved = base + 0.2
    for _ in range(40):
        t += 1 / 30
        out = sm.apply(1, moved, t)
    assert np.abs(out - moved).max() < 0.01
