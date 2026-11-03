"""Derived metrics — eye-aspect-ratio / blink detection and head pose.

These build on the raw 478 landmarks (and the optional facial transformation
matrix) returned by the detector. Nothing here touches MediaPipe directly, so
the functions are easy to unit-test with plain numbers.
"""

from __future__ import annotations

import math

import numpy as np

# --- Eye landmarks (MediaPipe 478-point topology) -------------------------
# Each list is the classic 6-point EAR set, ordered:
#   [corner, top-a, top-b, corner, bottom-b, bottom-a]
# "left"/"right" are the subject's eyes (left eye appears on the right of a
# non-mirrored image).
EYE_LANDMARKS_LEFT = (362, 385, 387, 263, 373, 380)
EYE_LANDMARKS_RIGHT = (33, 160, 158, 133, 153, 144)

# Nose tip — a stable anchor for the head-pose gizmo.
NOSE_TIP = 1

# Mouth landmarks for the Mouth Aspect Ratio (yawn detection).
MOUTH_VERTICAL = (13, 14)     # upper-inner lip, lower-inner lip
MOUTH_HORIZONTAL = (61, 291)  # left corner, right corner

# Iris landmarks (require the 478-point refined model). Centers + rings.
LEFT_IRIS = (474, 475, 476, 477)
RIGHT_IRIS = (469, 470, 471, 472)
# Eye corners (inner, outer) and lid extremes (top, bottom) per eye.
LEFT_EYE_CORNERS = (362, 263)   # inner, outer
LEFT_EYE_LIDS = (386, 374)      # top, bottom
RIGHT_EYE_CORNERS = (133, 33)   # inner, outer
RIGHT_EYE_LIDS = (159, 145)     # top, bottom


def _px(landmark, width: int, height: int) -> np.ndarray:
    return np.array([landmark.x * width, landmark.y * height], dtype=float)


def eye_aspect_ratio(landmarks, indices, width: int, height: int) -> float:
    """Eye Aspect Ratio for one eye, computed in pixel space.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|). Open eyes sit around
    0.25-0.35; a closed eye drops toward ~0.1.
    """
    p = [_px(landmarks[i], width, height) for i in indices]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return float(vertical / (2.0 * horizontal)) if horizontal > 0 else 0.0


def average_ear(landmarks, width: int, height: int) -> float:
    """Mean EAR across both eyes."""
    left = eye_aspect_ratio(landmarks, EYE_LANDMARKS_LEFT, width, height)
    right = eye_aspect_ratio(landmarks, EYE_LANDMARKS_RIGHT, width, height)
    return (left + right) / 2.0


def mouth_aspect_ratio(landmarks, width: int, height: int) -> float:
    """Mouth Aspect Ratio = vertical opening / horizontal width.

    Rises sharply during a yawn; a relaxed closed mouth sits near ~0.05-0.2.
    """
    top = _px(landmarks[MOUTH_VERTICAL[0]], width, height)
    bottom = _px(landmarks[MOUTH_VERTICAL[1]], width, height)
    left = _px(landmarks[MOUTH_HORIZONTAL[0]], width, height)
    right = _px(landmarks[MOUTH_HORIZONTAL[1]], width, height)
    horizontal = np.linalg.norm(left - right)
    return float(np.linalg.norm(top - bottom) / horizontal) if horizontal > 0 else 0.0


def _iris_center(landmarks, ring) -> tuple[float, float]:
    xs = [landmarks[i].x for i in ring]
    ys = [landmarks[i].y for i in ring]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _eye_gaze(landmarks, ring, corners, lids) -> tuple[float, float]:
    """Normalized iris position within one eye: (h, v) in ~[0, 1], 0.5 center."""
    cx, cy = _iris_center(landmarks, ring)
    x0, x1 = landmarks[corners[0]].x, landmarks[corners[1]].x
    yt, yb = landmarks[lids[0]].y, landmarks[lids[1]].y
    lo, hi = min(x0, x1), max(x0, x1)
    h = (cx - lo) / (hi - lo) if hi > lo else 0.5
    v = (cy - yt) / (yb - yt) if yb > yt else 0.5
    return h, v


def gaze_direction(landmarks) -> dict:
    """Estimate gaze from iris position relative to the eye opening.

    Returns ``{"h": float, "v": float, "dir": str}`` where ``h``/``v`` are in
    roughly ``[0, 1]`` (0.5 = centered) and ``dir`` is a coarse label such as
    ``"center"``, ``"left"`` or ``"up-right"``. Labels are screen-relative.
    """
    hl, vl = _eye_gaze(landmarks, LEFT_IRIS, LEFT_EYE_CORNERS, LEFT_EYE_LIDS)
    hr, vr = _eye_gaze(landmarks, RIGHT_IRIS, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS)
    h, v = (hl + hr) / 2.0, (vl + vr) / 2.0

    horiz = "left" if h < 0.42 else "right" if h > 0.58 else ""
    vert = "up" if v < 0.43 else "down" if v > 0.60 else ""
    label = "-".join(p for p in (vert, horiz) if p) or "center"
    return {"h": round(h, 3), "v": round(v, 3), "dir": label}


class BlinkCounter:
    """Count blinks via EAR thresholding with a short debounce.

    A blink is registered when the eye reopens after having been below
    ``ear_threshold`` for at least ``consec_frames`` consecutive frames.
    """

    def __init__(self, ear_threshold: float = 0.21, consec_frames: int = 2) -> None:
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.count = 0
        self._below = 0
        self._closed = False

    def update(self, ear: float) -> int:
        if ear < self.ear_threshold:
            self._below += 1
            if self._below >= self.consec_frames:
                self._closed = True
        else:
            if self._closed:
                self.count += 1  # rising edge: eye just reopened
            self._below = 0
            self._closed = False
        return self.count

    @property
    def is_closed(self) -> bool:
        return self._closed


class BlinkTracker:
    """Track per-eye blink counts and a rolling blinks-per-minute rate.

    Maintains independent counters for the left eye, right eye, and the
    two-eye average (the average drives the "total" count and the rate).
    """

    def __init__(self, ear_threshold: float = 0.21, consec_frames: int = 2,
                 rate_window_s: float = 60.0) -> None:
        self.left = BlinkCounter(ear_threshold, consec_frames)
        self.right = BlinkCounter(ear_threshold, consec_frames)
        self.both = BlinkCounter(ear_threshold, consec_frames)
        self.rate_window_s = rate_window_s
        self._events: list[float] = []  # timestamps (s) of counted blinks

    def update(self, landmarks, width: int, height: int, now_s: float) -> dict:
        ear_l = eye_aspect_ratio(landmarks, EYE_LANDMARKS_LEFT, width, height)
        ear_r = eye_aspect_ratio(landmarks, EYE_LANDMARKS_RIGHT, width, height)
        ear_avg = (ear_l + ear_r) / 2.0

        self.left.update(ear_l)
        self.right.update(ear_r)
        before = self.both.count
        total = self.both.update(ear_avg)
        if total > before:  # a (combined) blink just completed
            self._events.append(now_s)

        # Prune events outside the rolling window; rate = events / window (per min).
        cutoff = now_s - self.rate_window_s
        self._events = [t for t in self._events if t >= cutoff]
        rate_per_min = len(self._events) * (60.0 / self.rate_window_s)

        return {
            "ear": round(ear_avg, 4),
            "ear_left": round(ear_l, 4),
            "ear_right": round(ear_r, 4),
            "left": self.left.count,
            "right": self.right.count,
            "total": total,
            "rate_per_min": round(rate_per_min, 1),
            "closed": self.both.is_closed,
        }


# --- Head pose ------------------------------------------------------------
def head_pose_from_matrix(matrix) -> tuple[float, float, float]:
    """Decompose a facial transformation matrix into Euler angles (degrees).

    Args:
        matrix: 4x4 (or 3x3) facial transformation matrix from the detector
            (``output_transformation_matrixes=True``).

    Returns:
        ``(pitch, yaw, roll)`` in degrees — nod, shake, tilt respectively.
    """
    R = np.asarray(matrix, dtype=float)[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal-lock fallback
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def head_pose_axes_2d(matrix, length: float = 70.0) -> list[tuple[int, int]]:
    """Project the rotated unit axes to 2D screen offsets.

    Returns three ``(dx, dy)`` pixel offsets for the X, Y and Z axes,
    relative to a chosen origin (image y grows downward, so it is negated).
    """
    R = np.asarray(matrix, dtype=float)[:3, :3]
    axes = np.eye(3) * length
    offsets = []
    for axis in axes:
        v = R @ axis
        offsets.append((int(round(v[0])), int(round(-v[1]))))
    return offsets
