"""Tests for the quality gate."""
import math

import numpy as np

from facemesh.medical.geometry import IDX_BOTTOM, IDX_EYE_L, IDX_EYE_R, IDX_ORIGIN, IDX_TOP
from facemesh.medical.quality import (
    REASON_BLOWN, REASON_DARK, REASON_MOTION, REASON_NO_FACE,
    REASON_POSE, REASON_TOO_SMALL, REASON_UNEVEN, QualityGate,
)

W = H = 480


class LM:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def face(scale=0.5, cx=0.5, cy=0.5):
    p = np.zeros((478, 2))
    half = scale / 2
    p[IDX_TOP] = [cx, cy - half]
    p[IDX_BOTTOM] = [cx, cy + half]
    p[IDX_ORIGIN] = [cx, cy]
    p[IDX_EYE_R] = [cx - 0.08, cy]
    p[IDX_EYE_L] = [cx + 0.08, cy]
    for i in range(478):
        if not p[i].any():
            p[i] = [cx, cy]
    return [LM(x, y) for x, y in p]


def yaw_matrix(deg):
    a = math.radians(deg)
    M = np.eye(4)
    M[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    return M


def frame(val=130, left=None, right=None):
    f = np.full((H, W, 3), val, np.uint8)
    if left is not None:
        f[:, : W // 2] = left
    if right is not None:
        f[:, W // 2:] = right
    return f


def test_good_conditions_pass():
    g = QualityGate()
    g.check(face(), W, H, 0.0, frame_bgr=frame(), matrix=yaw_matrix(3))
    r = g.check(face(), W, H, 1 / 30, frame_bgr=frame(), matrix=yaw_matrix(3))
    assert r["ok"] and r["score"] == 1.0 and not r["reasons"]


def test_no_face():
    r = QualityGate().check([], W, H, 0.0)
    assert not r["ok"] and REASON_NO_FACE in r["reasons"]


def test_face_too_small():
    r = QualityGate().check(face(scale=0.08), W, H, 0.0, frame_bgr=frame())
    assert REASON_TOO_SMALL in r["reasons"] and not r["ok"]


def test_pose_too_far():
    r = QualityGate().check(face(), W, H, 0.0, frame_bgr=frame(), matrix=yaw_matrix(35))
    assert REASON_POSE in r["reasons"]


def test_too_dark_and_overexposed():
    assert REASON_DARK in QualityGate().check(face(), W, H, 0.0, frame_bgr=frame(20))["reasons"]
    assert REASON_BLOWN in QualityGate().check(face(), W, H, 0.0, frame_bgr=frame(250))["reasons"]


def test_uneven_lighting():
    r = QualityGate().check(face(), W, H, 0.0, frame_bgr=frame(left=40, right=200))
    assert REASON_UNEVEN in r["reasons"]


def test_motion_detected_in_image_space():
    """Motion must be measured in image space; the face-local frame is motion-invariant."""
    g = QualityGate()
    g.check(face(cx=0.30), W, H, 0.0, frame_bgr=frame())
    r = g.check(face(cx=0.75), W, H, 1 / 30, frame_bgr=frame())
    assert REASON_MOTION in r["reasons"] and r["motion"] > 0
