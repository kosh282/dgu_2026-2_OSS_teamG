"""Render self-test — verify drawing works without a camera or model.

This synthesizes a structured 478-point landmark cloud and renders every
mesh feature onto a blank canvas. It exercises the full drawing pipeline
(normalized->pixel mapping, all connection sets, blendshape overlay) so you
can confirm your install renders correctly before plugging in a webcam.

    python scripts/render_demo.py            # writes render_demo.png

Note: the output is an abstract mesh, not a real face — real landmark
positions come from the model at runtime via main.py.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facemesh.drawing import (
    FEATURE_GROUPS,
    draw_blendshapes_overlay,
    draw_drowsiness_banner,
    draw_face_landmarks,
    draw_face_panels,
    draw_gaze_cursor,
    draw_head_pose_axes,
    draw_rec_indicator,
    draw_status_bar,
)
from facemesh.metrics import head_pose_axes_2d, head_pose_from_matrix


@dataclass
class _LM:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0
    name: str = ""


@dataclass
class _Cat:
    score: float
    category_name: str
    index: int = 0
    display_name: str = ""


def synth_landmarks(n: int = 478, seed: int = 7) -> list[_LM]:
    """478 deterministic points inside a face-shaped oval."""
    rng = np.random.default_rng(seed)
    pts: list[_LM] = []
    for _ in range(n):
        # Sample within a centered ellipse so the mesh looks face-ish.
        r = math.sqrt(rng.random())
        theta = rng.random() * 2 * math.pi
        x = 0.5 + 0.30 * r * math.cos(theta)
        y = 0.5 + 0.38 * r * math.sin(theta)
        pts.append(_LM(x=x, y=y, z=rng.random() * 0.1))
    return pts


def main() -> int:
    canvas = np.full((600, 600, 3), 18, dtype=np.uint8)
    landmarks = synth_landmarks()

    draw_face_landmarks(canvas, landmarks, FEATURE_GROUPS["all"], draw_points=True)

    fake_blendshapes = [
        _Cat(0.82, "mouthSmileLeft"), _Cat(0.79, "mouthSmileRight"),
        _Cat(0.41, "eyeBlinkLeft"), _Cat(0.12, "browInnerUp"),
        _Cat(0.05, "jawOpen"),
    ]
    draw_blendshapes_overlay(canvas, fake_blendshapes)

    # v0.2: head-pose gizmo (synthetic yaw/pitch) + metrics panel.
    import math
    yaw, pitch = math.radians(-25), math.radians(10)
    Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0],
                   [-math.sin(yaw), 0, math.cos(yaw)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]])
    M = np.eye(4)
    M[:3, :3] = Ry @ Rx
    p, y, r = head_pose_from_matrix(M)
    draw_head_pose_axes(canvas, (300, 300), head_pose_axes_2d(M, length=80))

    # v0.4: multi-face panels, drowsiness banner, calibrated gaze cursor.
    draw_face_panels(canvas, [
        ("Face 0", ["EAR 0.29 (open)", "Blinks 7 (L7/R6)", "Rate 18/min",
                    "Gaze up-right", f"P{p:+.0f} Y{y:+.0f} R{r:+.0f}",
                    "PERCLOS 22% [drowsy]"]),
        ("Face 1", ["EAR 0.31 (open)", "Blinks 4 (L4/R4)", "PERCLOS 5% [awake]"]),
    ])
    draw_drowsiness_banner(canvas, "drowsy")
    draw_gaze_cursor(canvas, (360, 250))
    draw_rec_indicator(canvas, 73.0, blink_on=True)
    draw_status_bar(canvas, "28.4 FPS | faces:2 | all | live smooth", "? help")

    out = Path(__file__).resolve().parent.parent / "render_demo.png"
    cv2.imwrite(str(out), canvas)
    print(f"Rendered {len(landmarks)} landmarks -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
