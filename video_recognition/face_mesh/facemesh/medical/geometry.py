"""Shared face-local normalization for the medical metrics.

Maps the raw landmarks into a canonical frame — origin between the eyes, the
nose-to-chin midline pointing down, the inter-ocular axis pointing right, and
scale set by inter-ocular distance — so rigid head motion (translation,
rotation, distance to camera) is removed. What's left is expression and
asymmetry, which is what these metrics want.

Both a 2D (image-plane) and a full 3D frame are available. The 3D frame uses
MediaPipe's ``z`` coordinate and builds a proper orthonormal basis, so
out-of-plane deviation (e.g. a drooping lip pulling backwards) is measured
rather than silently projected away.
"""

from __future__ import annotations

import numpy as np

IDX_TOP = 10      # forehead centre (top of midline)
IDX_BOTTOM = 152  # chin (bottom of midline)
IDX_ORIGIN = 168  # sellion, between the eyes
IDX_EYE_R = 33    # right eye outer corner
IDX_EYE_L = 263   # left eye outer corner


def to_pixels(landmarks, width: int, height: int) -> np.ndarray:
    return np.array([[lm.x * width, lm.y * height] for lm in landmarks], dtype=float)


def to_pixels_3d(landmarks, width: int, height: int) -> np.ndarray:
    """``(N, 3)`` array. ``z`` is scaled by width, matching MediaPipe's convention."""
    return np.array(
        [[lm.x * width, lm.y * height, getattr(lm, "z", 0.0) * width] for lm in landmarks],
        dtype=float,
    )


def normalize(landmarks, width: int, height: int) -> np.ndarray:
    """Return landmarks in the 2D face-local frame, shape ``(N, 2)``.

    ``y`` runs along the nose-to-chin axis, ``x`` is perpendicular (the
    mid-sagittal line sits at ``x = 0``); units are inter-ocular distances.
    """
    P = to_pixels(landmarks, width, height)
    origin = P[IDX_ORIGIN]
    axis = P[IDX_BOTTOM] - P[IDX_TOP]
    axis = axis / (np.linalg.norm(axis) + 1e-9)          # down the midline
    normal = np.array([-axis[1], axis[0]])               # perpendicular
    scale = np.linalg.norm(P[IDX_EYE_R] - P[IDX_EYE_L]) + 1e-9

    rel = P - origin
    x = rel @ normal / scale
    y = rel @ axis / scale
    return np.stack([x, y], axis=1)


def normalize_3d(landmarks, width: int, height: int) -> np.ndarray:
    """Return landmarks in the 3D face-local frame, shape ``(N, 3)``.

    Axes: ``x`` along the eye line (mid-sagittal plane at ``x = 0``), ``y``
    down the midline, ``z`` out of the face (right-handed). Units are
    inter-ocular distances. Built by Gram-Schmidt so the basis is orthonormal
    even when the raw landmark axes are not exactly perpendicular.
    """
    P = to_pixels_3d(landmarks, width, height)
    origin = P[IDX_ORIGIN]

    # Primary axis: down the facial midline.
    y_axis = P[IDX_BOTTOM] - P[IDX_TOP]
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)

    # Secondary: eye line, orthogonalized against y (Gram-Schmidt).
    eye = P[IDX_EYE_L] - P[IDX_EYE_R]
    scale = np.linalg.norm(eye) + 1e-9
    x_axis = eye - np.dot(eye, y_axis) * y_axis
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)

    basis = np.stack([x_axis, y_axis, z_axis])   # (3, 3) rows are axes
    return (P - origin) @ basis.T / scale
