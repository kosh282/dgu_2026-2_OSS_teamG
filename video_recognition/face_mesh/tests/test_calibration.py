import numpy as np
import pytest

from facemesh.calibration import GazeCalibrator, grid_targets


def test_affine_recovers_exactly():
    A = np.array([[1600., 200., -300.], [120., 900., -250.]])
    cal = GazeCalibrator(degree=1)
    for h in (0.2, 0.5, 0.8):
        for v in (0.2, 0.5, 0.8):
            x, y = A @ np.array([1, h, v])  # note: design is [1,h,v]
            cal.add_sample(h, v, x, y)
    cal.fit()
    got = cal.apply(0.4, 0.6)
    exp = A @ np.array([1, 0.4, 0.6])
    assert np.allclose(got, exp, atol=1e-6)


def test_quadratic_beats_affine_on_curved_data():
    def true(h, v):
        return (900 * h + 120 * h * v + 40, 700 * v - 150 * h * h + 30)
    def build(deg):
        c = GazeCalibrator(degree=deg)
        for h in (0.1, 0.3, 0.5, 0.7, 0.9):
            for v in (0.2, 0.5, 0.8):
                x, y = true(h, v)
                c.add_sample(h, v, x, y)
        return c.fit()
    assert build(2) < build(1)


def test_save_load_roundtrip(tmp_path):
    cal = GazeCalibrator(degree=1)
    for h, v in [(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)]:
        cal.add_sample(h, v, h * 100, v * 100)
    cal.fit()
    p = tmp_path / "cal.json"
    cal.save(p)
    loaded = GazeCalibrator.load(p)
    assert np.allclose(loaded.coef, cal.coef)
    assert loaded.degree == cal.degree


def test_too_few_points_raises():
    cal = GazeCalibrator(degree=1)
    cal.add_sample(0.5, 0.5, 1, 1)
    with pytest.raises(ValueError):
        cal.fit()


def test_grid_targets_count():
    assert len(grid_targets(640, 480)) == 9
