"""Appearance-based re-identification from geometric mesh embeddings."""
import math

import numpy as np

from facemesh.medical.identity import (
    AppearanceTracker, FaceGallery, face_embedding, similarity,
)


class LM:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def _person(seed, n=478):
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 0.05, (n, 3))
    p[10] = [0.0, -0.35, 0.0]
    p[152] = [0.0, 0.40, 0.0]
    p[168] = [0.0, 0.0, 0.0]
    p[33] = [-0.10 - 0.01 * (seed % 3), 0.0, 0.0]
    p[263] = [0.10 + 0.01 * (seed % 3), 0.0, 0.0]
    p[1] = [0.0, 0.15 + 0.02 * (seed % 4), 0.04]
    p[234] = [-0.20 - 0.02 * (seed % 5), 0.05, 0]
    p[454] = [0.20 + 0.02 * (seed % 5), 0.05, 0]
    return p


def _lms(p):
    return [LM(*r) for r in p]


def _rot3(rx, ry, rz):
    ax, ay, az = map(math.radians, (rx, ry, rz))
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rotate_physically(p, W, H, rx, ry, rz, scale=1.0, shift=(0, 0, 0)):
    """Rotation in pixel space — how a real head moves."""
    px = p * [W, H, W]
    px = (px * scale) @ _rot3(rx, ry, rz).T + np.array(shift) * [W, H, W]
    return px / [W, H, W]


def test_embedding_pose_invariant_at_any_aspect():
    a = _person(1)
    for W, H in [(640, 640), (640, 480), (1280, 720)]:
        e1 = face_embedding(_lms(a), W, H)
        moved = _rotate_physically(a, W, H, 18, 30, -12, 1.6, (0.3, -0.2, 0.1))
        assert similarity(e1, face_embedding(_lms(moved), W, H)) > 0.9999


def test_same_person_beats_different_person():
    rng = np.random.default_rng(0)
    same, diff = [], []
    for seed in range(1, 8):
        base = _person(seed)
        eb = face_embedding(_lms(base), 640, 480)
        same.append(similarity(eb, face_embedding(_lms(base + rng.normal(0, 0.001, base.shape)),
                                                  640, 480)))
        diff.append(similarity(eb, face_embedding(_lms(_person(seed + 100)), 640, 480)))
    assert min(same) > max(diff)


def _gallery(rng):
    g = FaceGallery(threshold=0.9995)
    for seed, name in [(1, "ana"), (2, "bob"), (3, "cleo")]:
        p = _person(seed)
        for _ in range(3):
            g.enroll(name, face_embedding(_lms(p + rng.normal(0, 0.0005, p.shape)), 640, 480))
    return g


def test_gallery_identifies_enrolled():
    rng = np.random.default_rng(1)
    g = _gallery(rng)
    for seed, expect in [(1, "ana"), (2, "bob"), (3, "cleo")]:
        probe = _person(seed) + rng.normal(0, 0.0008, (478, 3))
        name, _ = g.identify(face_embedding(_lms(probe), 640, 480))
        assert name == expect


def test_unknown_face_rejected():
    g = _gallery(np.random.default_rng(1))
    name, score = g.identify(face_embedding(_lms(_person(999)), 640, 480))
    assert name is None and score < g.threshold


def test_empty_gallery_returns_none():
    name, score = FaceGallery().identify(face_embedding(_lms(_person(1)), 640, 480))
    assert name is None and score == 0.0


def test_gallery_save_load(tmp_path):
    g = _gallery(np.random.default_rng(1))
    g.save(tmp_path / "g.json")
    g2 = FaceGallery.load(tmp_path / "g.json")
    assert sorted(g2.names) == ["ana", "bob", "cleo"]
    name, _ = g2.identify(face_embedding(_lms(_person(2)), 640, 480))
    assert name == "bob"


def test_identity_survives_position_swap():
    """The case position-based tracking provably cannot handle."""
    tr = AppearanceTracker(threshold=0.9995)
    p1, p2 = _person(1), _person(2)
    a = tr.update([_lms(p1 - [0.2, 0, 0]), _lms(p2 + [0.2, 0, 0])], 640, 480)
    b = tr.update([_lms(p2 - [0.2, 0, 0]), _lms(p1 + [0.2, 0, 0])], 640, 480)
    assert b == [a[1], a[0]]


def test_identity_survives_leaving_and_returning():
    tr = AppearanceTracker(threshold=0.9995)
    p1, p2 = _person(1), _person(2)
    first = tr.update([_lms(p1)], 640, 480)
    tr.update([], 640, 480)
    tr.update([_lms(p2)], 640, 480)
    assert tr.update([_lms(p1)], 640, 480) == first


def test_no_auto_enroll_yields_unknown():
    tr = AppearanceTracker(gallery=FaceGallery(), auto_enroll=False)
    assert tr.update([_lms(_person(1))], 640, 480) == ["unknown"]
