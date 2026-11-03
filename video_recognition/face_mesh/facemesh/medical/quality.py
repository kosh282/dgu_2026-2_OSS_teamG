"""Quality gating — refuse to report metrics when conditions are bad.

Confidently wrong output is worse than no output. Before any clinical-style
metric is trusted, the capture conditions have to be adequate: the face must be
large enough, roughly frontal, evenly and adequately lit, and not moving too
fast. This module scores those conditions and produces a single verdict.

Every metric in this package should be suppressed (or flagged) when
:meth:`QualityGate.check` reports ``ok=False``.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import IDX_BOTTOM, IDX_EYE_L, IDX_EYE_R, IDX_TOP, to_pixels

# Reasons, ordered by how strongly they invalidate a measurement.
REASON_NO_FACE = "no face"
REASON_TOO_SMALL = "face too small"
REASON_POSE = "head turned too far"
REASON_DARK = "too dark"
REASON_BLOWN = "overexposed"
REASON_UNEVEN = "uneven lighting"
REASON_MOTION = "moving too fast"


def face_size(landmarks, width: int, height: int) -> float:
    """Face height (forehead to chin) as a fraction of frame height."""
    P = to_pixels(landmarks, width, height)
    return float(np.linalg.norm(P[IDX_BOTTOM] - P[IDX_TOP]) / max(height, 1))


def pose_angles(matrix) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees from a 4x4 transformation matrix."""
    R = np.asarray(matrix, dtype=float)[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch, yaw, roll = math.atan2(-R[1, 2], R[1, 1]), math.atan2(-R[2, 0], sy), 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def pose_from_landmarks(landmarks, width: int, height: int) -> float:
    """Crude yaw proxy when no matrix is available: eye-line asymmetry.

    Returns an absolute value in ``[0, 1]``; 0 means the nose sits midway
    between the eye corners (frontal), larger means turned.
    """
    P = to_pixels(landmarks, width, height)
    left, right = P[IDX_EYE_L], P[IDX_EYE_R]
    mid = (left + right) / 2.0
    nose = P[IDX_TOP] + (P[IDX_BOTTOM] - P[IDX_TOP]) * 0.35
    span = np.linalg.norm(left - right) + 1e-9
    # Perpendicular offset of the nose from the eye-line midpoint.
    return float(abs(np.dot(nose - mid, (left - right) / span)) / span)


def lighting(frame_bgr, landmarks, width: int, height: int) -> dict:
    """Mean brightness and left/right imbalance over the face box."""
    P = to_pixels(landmarks, width, height)
    x0, y0 = np.clip(P.min(axis=0), [0, 0], [width - 1, height - 1]).astype(int)
    x1, y1 = np.clip(P.max(axis=0), [0, 0], [width - 1, height - 1]).astype(int)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return {"brightness": 0.0, "imbalance": 1.0}
    box = frame_bgr[y0:y1, x0:x1]
    gray = box.mean(axis=2)
    mid = gray.shape[1] // 2
    lhs, rhs = gray[:, :mid].mean(), gray[:, mid:].mean()
    denom = max(lhs + rhs, 1e-6)
    return {
        "brightness": float(gray.mean() / 255.0),
        "imbalance": float(abs(lhs - rhs) / denom),
    }


class QualityGate:
    """Assess whether the current frame is good enough to measure.

    Args:
        min_face_height: Face height as a fraction of frame height.
        max_pose_deg: Maximum absolute pitch/yaw/roll (needs a pose matrix).
        min_brightness / max_brightness: Acceptable mean face luminance (0-1).
        max_imbalance: Max left/right lighting imbalance (0-1).
        max_motion: Max mean landmark displacement per second (normalized units).
    """

    def __init__(self, min_face_height: float = 0.18, max_pose_deg: float = 15.0,
                 min_brightness: float = 0.15, max_brightness: float = 0.92,
                 max_imbalance: float = 0.25, max_motion: float = 0.60) -> None:
        self.min_face_height = min_face_height
        self.max_pose_deg = max_pose_deg
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.max_imbalance = max_imbalance
        self.max_motion = max_motion
        self._prev: np.ndarray | None = None
        self._prev_t: float | None = None

    def motion(self, landmarks, width: int, height: int, now_s: float) -> float:
        """Mean landmark speed in inter-ocular units per second.

        Measured in *image* space (scaled by inter-ocular distance), NOT in the
        face-local frame — that frame removes rigid motion by construction, so
        measuring there would always return zero.
        """
        P = to_pixels(landmarks, width, height)
        iod = np.linalg.norm(P[IDX_EYE_R] - P[IDX_EYE_L]) + 1e-9
        N = P / iod
        speed = 0.0
        if self._prev is not None and self._prev_t is not None:
            dt = now_s - self._prev_t
            if dt > 1e-6:
                speed = float(np.linalg.norm(N - self._prev, axis=1).mean() / dt)
        self._prev, self._prev_t = N, now_s
        return speed

    def check(self, landmarks, width: int, height: int, now_s: float,
              frame_bgr=None, matrix=None) -> dict:
        """Return ``{ok, score, reasons, ...}`` for the current frame."""
        if landmarks is None or len(landmarks) == 0:
            return {"ok": False, "score": 0.0, "reasons": [REASON_NO_FACE],
                    "face_height": 0.0, "motion": 0.0}

        reasons: list[str] = []
        fh = face_size(landmarks, width, height)
        if fh < self.min_face_height:
            reasons.append(REASON_TOO_SMALL)

        if matrix is not None:
            pitch, yaw, roll = pose_angles(matrix)
            if max(abs(pitch), abs(yaw), abs(roll)) > self.max_pose_deg:
                reasons.append(REASON_POSE)

        light = {"brightness": None, "imbalance": None}
        if frame_bgr is not None:
            light = lighting(frame_bgr, landmarks, width, height)
            if light["brightness"] < self.min_brightness:
                reasons.append(REASON_DARK)
            elif light["brightness"] > self.max_brightness:
                reasons.append(REASON_BLOWN)
            if light["imbalance"] > self.max_imbalance:
                reasons.append(REASON_UNEVEN)

        speed = self.motion(landmarks, width, height, now_s)
        if speed > self.max_motion:
            reasons.append(REASON_MOTION)

        # Score: 1.0 when everything passes comfortably, degrading per failure.
        score = max(0.0, 1.0 - 0.25 * len(reasons))
        return {
            "ok": not reasons,
            "score": round(score, 2),
            "reasons": reasons,
            "face_height": round(fh, 3),
            "motion": round(speed, 3),
            "brightness": None if light["brightness"] is None else round(light["brightness"], 3),
            "imbalance": None if light["imbalance"] is None else round(light["imbalance"], 3),
        }
