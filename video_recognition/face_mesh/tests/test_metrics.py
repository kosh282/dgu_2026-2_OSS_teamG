import math

import numpy as np

from facemesh.metrics import (
    EYE_LANDMARKS_LEFT,
    LEFT_EYE_CORNERS,
    LEFT_EYE_LIDS,
    LEFT_IRIS,
    RIGHT_EYE_CORNERS,
    RIGHT_EYE_LIDS,
    RIGHT_IRIS,
    eye_aspect_ratio,
    gaze_direction,
    head_pose_from_matrix,
    mouth_aspect_ratio,
    MOUTH_HORIZONTAL,
    MOUTH_VERTICAL,
)


def _blank(lm_cls, n=478):
    return [lm_cls() for _ in range(n)]


def _set_eye(lm, idx, openness):
    p1, p2, p3, p4, p5, p6 = idx
    lm[p1].x, lm[p1].y = 0.0, 0.0
    lm[p4].x, lm[p4].y = 0.10, 0.0
    g = 0.03 * openness
    lm[p2].x, lm[p2].y = 0.03, g
    lm[p3].x, lm[p3].y = 0.06, g
    lm[p6].x, lm[p6].y = 0.03, -g
    lm[p5].x, lm[p5].y = 0.06, -g


def test_ear_open_greater_than_closed(lm_cls):
    op, cl = _blank(lm_cls), _blank(lm_cls)
    _set_eye(op, EYE_LANDMARKS_LEFT, 1.0)
    _set_eye(cl, EYE_LANDMARKS_LEFT, 0.1)
    assert eye_aspect_ratio(op, EYE_LANDMARKS_LEFT, 640, 480) > \
           eye_aspect_ratio(cl, EYE_LANDMARKS_LEFT, 640, 480)


def test_mar_open_greater_than_closed(lm_cls):
    op, cl = _blank(lm_cls), _blank(lm_cls)
    op[MOUTH_VERTICAL[0]].y = 0.40; op[MOUTH_VERTICAL[1]].y = 0.60
    op[MOUTH_HORIZONTAL[0]].x = 0.40; op[MOUTH_HORIZONTAL[1]].x = 0.60
    cl[MOUTH_VERTICAL[0]].y = 0.49; cl[MOUTH_VERTICAL[1]].y = 0.51
    cl[MOUTH_HORIZONTAL[0]].x = 0.40; cl[MOUTH_HORIZONTAL[1]].x = 0.60
    assert mouth_aspect_ratio(op, 640, 480) > mouth_aspect_ratio(cl, 640, 480)


def _set_gaze(lm, h, v):
    for ring, corners, lids in (
        (LEFT_IRIS, LEFT_EYE_CORNERS, LEFT_EYE_LIDS),
        (RIGHT_IRIS, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS),
    ):
        x0, x1, yt, yb = 0.40, 0.60, 0.40, 0.50
        lm[corners[0]].x = x0; lm[corners[1]].x = x1
        lm[lids[0]].y = yt; lm[lids[1]].y = yb
        cx, cy = x0 + h * (x1 - x0), yt + v * (yb - yt)
        for i in ring:
            lm[i].x, lm[i].y = cx, cy


def test_gaze_center_left_right(lm_cls):
    c, l, r = _blank(lm_cls), _blank(lm_cls), _blank(lm_cls)
    _set_gaze(c, 0.5, 0.5); _set_gaze(l, 0.15, 0.5); _set_gaze(r, 0.85, 0.5)
    assert gaze_direction(c)["dir"] == "center"
    assert gaze_direction(l)["dir"] == "left"
    assert gaze_direction(r)["dir"] == "right"


def test_head_pose_recovers_yaw():
    a = math.radians(25)
    M = np.eye(4)
    M[:3, :3] = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0],
                          [-math.sin(a), 0, math.cos(a)]])
    pitch, yaw, roll = head_pose_from_matrix(M)
    assert abs(yaw - 25) < 0.5 and abs(pitch) < 0.5 and abs(roll) < 0.5
