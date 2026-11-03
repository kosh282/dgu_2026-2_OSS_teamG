"""Baseline persistence: save, load, compare across sessions."""
import json

import numpy as np
import pytest

from facemesh.medical.geometry import IDX_BOTTOM, IDX_EYE_L, IDX_EYE_R, IDX_ORIGIN, IDX_TOP
from facemesh.medical.symmetry import REGIONS, SymmetryBaseline, symmetry_metrics


class LM:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def _face(rng, droop=0.0):
    p = np.zeros((478, 3))
    p[IDX_TOP] = [0.5, 0.20, 0]
    p[IDX_BOTTOM] = [0.5, 0.80, 0]
    p[IDX_ORIGIN] = [0.5, 0.40, 0]
    p[IDX_EYE_R] = [0.42, 0.40, 0]
    p[IDX_EYE_L] = [0.58, 0.40, 0]
    for pairs in REGIONS.values():
        for r, l in pairs:
            dx = 0.03 + rng.random() * 0.06
            y = 0.35 + rng.random() * 0.30
            p[r] = [0.5 - dx, y, 0]
            p[l] = [0.5 + dx, y, 0]
    for _, l in REGIONS["mouth"]:
        p[l][1] += droop
    return p


def _fit(rng, droop=0.0, subject="tester", n=12):
    p = _face(rng, droop)
    bl = SymmetryBaseline(subject=subject)
    for _ in range(n):
        jitter = p + rng.normal(0, 0.0008, p.shape)
        bl.add_sample(symmetry_metrics([LM(*r) for r in jitter], 640, 640))
    bl.fit()
    return bl, p


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(9)
    bl, _ = _fit(rng)
    path = bl.save(tmp_path / "b.json")
    loaded = SymmetryBaseline.load(path)
    assert loaded.subject == "tester"
    assert loaded.baseline.keys() == bl.baseline.keys()
    for k in bl.baseline:
        assert abs(loaded.baseline[k] - bl.baseline[k]) < 1e-9


def test_saved_file_has_metadata(tmp_path):
    rng = np.random.default_rng(3)
    bl, _ = _fit(rng, subject="anastasija")
    data = json.loads(bl.save(tmp_path / "b.json").read_text())
    assert data["subject"] == "anastasija"
    assert data["n_samples"] == 12
    assert data["created"] and data["version"] == 1


def test_loaded_baseline_can_compare(tmp_path):
    rng = np.random.default_rng(9)
    bl, p = _fit(rng, droop=0.012)
    loaded = SymmetryBaseline.load(bl.save(tmp_path / "b.json"))

    own = loaded.compare(symmetry_metrics([LM(*r) for r in p], 640, 640))
    assert own["relative_symmetry"] > 0.95        # their own normal reads normal

    p2 = p.copy()
    for _, l in REGIONS["mouth"]:
        p2[l][1] += 0.05                          # acquired droop
    worse = loaded.compare(symmetry_metrics([LM(*r) for r in p2], 640, 640))
    assert worse["relative_symmetry"] < own["relative_symmetry"] - 0.1
    assert worse["worst_delta_region"] == "mouth"


def test_save_unfitted_raises(tmp_path):
    with pytest.raises(RuntimeError):
        SymmetryBaseline().save(tmp_path / "x.json")


def test_load_unfitted_file_raises(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"baseline": None}))
    with pytest.raises(ValueError):
        SymmetryBaseline.load(p)
