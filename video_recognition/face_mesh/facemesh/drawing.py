"""Rendering helpers — turn landmarks into a mesh overlay with OpenCV.

Colors are BGR tuples (OpenCV convention). The public entry point is
:func:`draw_face_landmarks`, which renders one or more "feature sets"
(tesselation, contours, irises, ...) onto a frame.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import cv2

from mediapipe.tasks.python.vision import FaceLandmarksConnections as _C

# Each feature maps to (connection_set, line_color_BGR, thickness).
# Connection sets are iterables of objects with ``.start`` / ``.end`` indices.
FEATURE_SETS = {
    "tesselation": (_C.FACE_LANDMARKS_TESSELATION, (90, 110, 90), 1),
    "face_oval": (_C.FACE_LANDMARKS_FACE_OVAL, (224, 224, 224), 1),
    "lips": (_C.FACE_LANDMARKS_LIPS, (120, 120, 255), 1),
    "left_eye": (_C.FACE_LANDMARKS_LEFT_EYE, (80, 230, 80), 1),
    "right_eye": (_C.FACE_LANDMARKS_RIGHT_EYE, (80, 230, 80), 1),
    "left_eyebrow": (_C.FACE_LANDMARKS_LEFT_EYEBROW, (80, 230, 80), 1),
    "right_eyebrow": (_C.FACE_LANDMARKS_RIGHT_EYEBROW, (80, 230, 80), 1),
    "nose": (_C.FACE_LANDMARKS_NOSE, (224, 224, 224), 1),
    "left_iris": (_C.FACE_LANDMARKS_LEFT_IRIS, (0, 200, 255), 2),
    "right_iris": (_C.FACE_LANDMARKS_RIGHT_IRIS, (0, 200, 255), 2),
}

# Convenient named groups the CLI exposes via --features.
FEATURE_GROUPS = {
    "tesselation": ["tesselation"],
    "contours": [
        "face_oval", "lips", "left_eye", "right_eye",
        "left_eyebrow", "right_eyebrow", "nose",
    ],
    "irises": ["left_iris", "right_iris"],
    "all": [
        "tesselation", "face_oval", "lips", "left_eye", "right_eye",
        "left_eyebrow", "right_eyebrow", "nose", "left_iris", "right_iris",
    ],
}


def _to_pixels(landmarks, width: int, height: int) -> list[tuple[int, int]]:
    """Map normalized landmark coords to integer pixel coordinates."""
    pts = []
    for lm in landmarks:
        x = min(max(lm.x, 0.0), 1.0)
        y = min(max(lm.y, 0.0), 1.0)
        pts.append((int(x * width), int(y * height)))
    return pts


def _draw_feature(image, pts: Sequence[tuple[int, int]], connections, color, thickness) -> None:
    n = len(pts)
    for conn in connections:
        s, e = conn.start, conn.end
        if s < n and e < n:
            cv2.line(image, pts[s], pts[e], color, thickness, cv2.LINE_AA)


def draw_face_landmarks(
    image,
    face_landmarks,
    features: Iterable[str] = ("tesselation", "contours", "irises"),
    *,
    draw_points: bool = False,
    point_color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw the mesh for one face, in place.

    Args:
        image: BGR frame to draw on (modified in place).
        face_landmarks: A single face's list of normalized landmarks.
        features: Feature names or groups (see ``FEATURE_GROUPS`` /
            ``FEATURE_SETS``). Unknown names are ignored.
        draw_points: Also plot a dot at every landmark.
        point_color: BGR color for landmark dots.
    """
    height, width = image.shape[:2]
    pts = _to_pixels(face_landmarks, width, height)

    # Expand groups (e.g. "contours") into concrete feature names, de-duped.
    resolved: list[str] = []
    for f in features:
        for name in FEATURE_GROUPS.get(f, [f]):
            if name not in resolved:
                resolved.append(name)

    for name in resolved:
        spec = FEATURE_SETS.get(name)
        if spec is None:
            continue
        connections, color, thickness = spec
        _draw_feature(image, pts, connections, color, thickness)

    if draw_points:
        for p in pts:
            cv2.circle(image, p, 1, point_color, -1, cv2.LINE_AA)


def draw_blendshapes_overlay(image, blendshapes, top_n: int = 5) -> None:
    """Render the strongest expression blendshapes as a small HUD panel.

    Args:
        image: BGR frame (modified in place).
        blendshapes: One face's list of blendshape ``Category`` objects.
        top_n: How many of the highest-scoring expressions to show.
    """
    if not blendshapes:
        return
    ranked = sorted(blendshapes, key=lambda c: c.score, reverse=True)[:top_n]

    pad, line_h, bar_w = 8, 22, 120
    panel_w = 230
    panel_h = pad * 2 + line_h * len(ranked)
    x0, y0 = 10, 10

    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

    for i, cat in enumerate(ranked):
        y = y0 + pad + i * line_h + 14
        name = (cat.category_name or "")[:14]
        cv2.putText(image, name, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (235, 235, 235), 1, cv2.LINE_AA)
        bx = x0 + panel_w - bar_w - pad
        cv2.rectangle(image, (bx, y - 9), (bx + bar_w, y - 1), (60, 60, 60), -1)
        fill = int(bar_w * min(max(cat.score, 0.0), 1.0))
        cv2.rectangle(image, (bx, y - 9), (bx + fill, y - 1), (0, 200, 255), -1)


def draw_head_pose_axes(image, origin_xy, axis_offsets, *, thickness: int = 2) -> None:
    """Draw a 3D orientation gizmo (X red, Y green, Z blue) at ``origin_xy``.

    Args:
        image: BGR frame (modified in place).
        origin_xy: ``(x, y)`` pixel anchor — typically the nose tip.
        axis_offsets: three ``(dx, dy)`` screen offsets from
            :func:`metrics.head_pose_axes_2d`, in X, Y, Z order.
    """
    ox, oy = int(origin_xy[0]), int(origin_xy[1])
    colors = ((0, 0, 255), (0, 255, 0), (255, 60, 60))  # X, Y, Z in BGR
    for (dx, dy), color in zip(axis_offsets, colors):
        cv2.line(image, (ox, oy), (ox + dx, oy + dy), color, thickness, cv2.LINE_AA)
    cv2.circle(image, (ox, oy), 3, (255, 255, 255), -1, cv2.LINE_AA)


def draw_gaze(image, landmarks, gaze, *, length: int = 40) -> None:
    """Mark iris centers and draw a short gaze-direction arrow per eye.

    Args:
        image: BGR frame (modified in place).
        landmarks: one face's normalized landmarks (needs the iris points).
        gaze: dict from :func:`metrics.gaze_direction` with ``h``/``v``.
    """
    from .metrics import LEFT_IRIS, RIGHT_IRIS  # local import avoids a cycle

    h, w = image.shape[:2]
    dx = int((gaze["h"] - 0.5) * 2 * length)
    dy = int((gaze["v"] - 0.5) * 2 * length)
    for ring in (LEFT_IRIS, RIGHT_IRIS):
        cx = int(sum(landmarks[i].x for i in ring) / len(ring) * w)
        cy = int(sum(landmarks[i].y for i in ring) / len(ring) * h)
        cv2.circle(image, (cx, cy), 3, (255, 230, 0), -1, cv2.LINE_AA)
        cv2.arrowedLine(image, (cx, cy), (cx + dx, cy + dy),
                        (255, 230, 0), 2, cv2.LINE_AA, tipLength=0.35)


def draw_metrics_panel(image, lines, *, anchor: str = "br") -> None:
    """Render a small translucent text panel of metric lines.

    Args:
        image: BGR frame (modified in place).
        lines: list of short strings to display, one per row.
        anchor: ``"br"`` (bottom-right) or ``"tr"`` (top-right).
    """
    if not lines:
        return
    h, w = image.shape[:2]
    pad, line_h = 8, 20
    panel_w = 200
    panel_h = pad * 2 + line_h * len(lines)
    x1 = w - panel_w - 10
    y1 = 10 if anchor == "tr" else h - panel_h - 10

    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + panel_w, y1 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

    for i, text in enumerate(lines):
        y = y1 + pad + i * line_h + 13
        cv2.putText(image, text, (x1 + pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (60, 255, 120), 1, cv2.LINE_AA)


def draw_face_panels(image, panels, colors=None) -> None:
    """Stack a compact metrics panel per face down the right edge.

    Args:
        image: BGR frame (modified in place).
        panels: list of ``(title, lines)`` tuples, one per face.
        colors: optional list of title BGR colors aligned with ``panels``.
    """
    h, w = image.shape[:2]
    pad, line_h, panel_w = 6, 18, 200
    y = 10
    for pi, (title, lines) in enumerate(panels):
        rows = [title, *lines]
        panel_h = pad * 2 + line_h * len(rows)
        if y + panel_h > h - 10:
            break  # ran out of vertical room
        x1 = w - panel_w - 10
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y), (x1 + panel_w, y + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
        title_color = colors[pi] if colors else (0, 200, 255)
        for i, text in enumerate(rows):
            color = title_color if i == 0 else (60, 255, 120)
            ty = y + pad + i * line_h + 13
            cv2.putText(image, text, (x1 + pad, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, color, 1, cv2.LINE_AA)
        y += panel_h + 8


# Distinct, readable colors (BGR) cycled per tracked face ID.
_ID_PALETTE = [
    (80, 220, 100), (0, 180, 255), (240, 150, 60), (200, 90, 220),
    (60, 200, 255), (230, 120, 120), (120, 220, 220), (170, 170, 60),
]


def color_for_id(face_id: int):
    """A stable BGR color for a tracked face ID."""
    return _ID_PALETTE[face_id % len(_ID_PALETTE)]


def draw_face_label(image, face, label: str, color) -> None:
    """Outline a face and tag it with ``label`` (its track ID) in ``color``.

    Makes multiple recognized people visible at once, each with a consistent
    color and ID that follows them across frames.
    """
    h, w = image.shape[:2]
    xs = [lm.x for lm in face]
    ys = [lm.y for lm in face]
    x0, y0 = int(min(xs) * w), int(min(ys) * h)
    x1, y1 = int(max(xs) * w), int(max(ys) * h)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = max(y0, th + 8)
    cv2.rectangle(image, (x0, ty - th - 8), (x0 + tw + 10, ty), color, -1)
    cv2.putText(image, label, (x0 + 5, ty - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (20, 20, 20), 1, cv2.LINE_AA)


def draw_drowsiness_banner(image, level: str) -> None:
    """Top banner reflecting the drowsiness ``level`` (awake/drowsy/ALERT)."""
    if level == "awake":
        return
    w = image.shape[1]
    color = (0, 0, 220) if level == "ALERT" else (0, 170, 230)  # red / amber
    text = "DROWSINESS ALERT" if level == "ALERT" else "drowsy"
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), color, -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    cv2.putText(image, text, (w // 2 - 110, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)


def draw_calibration_target(image, point, progress: float = 0.0) -> None:
    """Draw a fixation target with a dwell-progress ring at ``point``."""
    x, y = int(point[0]), int(point[1])
    cv2.circle(image, (x, y), 16, (120, 120, 120), 2, cv2.LINE_AA)
    cv2.circle(image, (x, y), 4, (60, 255, 120), -1, cv2.LINE_AA)
    if progress > 0:
        end = int(360 * min(max(progress, 0.0), 1.0))
        cv2.ellipse(image, (x, y), (16, 16), -90, 0, end, (60, 255, 120), 3, cv2.LINE_AA)


def draw_gaze_cursor(image, point) -> None:
    """Draw a crosshair cursor at a calibrated on-screen gaze ``point``."""
    h, w = image.shape[:2]
    x = int(min(max(point[0], 0), w - 1))
    y = int(min(max(point[1], 0), h - 1))
    cv2.circle(image, (x, y), 12, (255, 230, 0), 2, cv2.LINE_AA)
    cv2.line(image, (x - 18, y), (x + 18, y), (255, 230, 0), 1, cv2.LINE_AA)
    cv2.line(image, (x, y - 18), (x, y + 18), (255, 230, 0), 1, cv2.LINE_AA)


def draw_medical_panel(image, lines, disclaimer: str = "") -> None:
    """Left-side panel for clinical-style metrics, with a disclaimer footer."""
    if not lines:
        return
    h = image.shape[0]
    pad, line_h, panel_w = 8, 20, 250
    rows = len(lines)
    panel_h = pad * 2 + line_h * rows
    x1, y1 = 10, h - panel_h - 60

    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + panel_w, y1 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    for i, text in enumerate(lines):
        y = y1 + pad + i * line_h + 13
        cv2.putText(image, text, (x1 + pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 220, 255), 1, cv2.LINE_AA)
    if disclaimer:
        cv2.putText(image, disclaimer, (x1, y1 + panel_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 160), 1, cv2.LINE_AA)


def draw_task_prompt(image, state: dict) -> None:
    """Full-width prompt banner with a progress bar for the guided protocol."""
    h, w = image.shape[:2]
    bar_h = 84
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)

    if state.get("finished"):
        cv2.putText(image, "Assessment complete", (20, 46), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (60, 255, 120), 2, cv2.LINE_AA)
        return

    step = f"{state['index'] + 1}/{state['total']}"
    cv2.putText(image, f"{step}  {state['prompt']}", (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)

    if state.get("blocked"):
        status, colour = f"paused: {state['reason']}", (0, 165, 255)
    elif state.get("settling"):
        status, colour = "get ready...", (180, 180, 180)
    else:
        status, colour = "recording", (60, 255, 120)
    cv2.putText(image, status, (20, 58), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, colour, 1, cv2.LINE_AA)

    x0, x1, y = 20, w - 20, 72
    cv2.rectangle(image, (x0, y), (x1, y + 8), (60, 60, 60), -1)
    fill = int((x1 - x0) * state["progress"])
    cv2.rectangle(image, (x0, y), (x0 + fill, y + 8), colour, -1)


def draw_flash(image) -> None:
    """Fill the frame with white — the light stimulus for the pupillary reflex."""
    image[:] = 255


def draw_plr_status(image, state: dict) -> None:
    """Phase banner + progress bar for a pupillary-light-reflex trial."""
    h, w = image.shape[:2]
    bar_h = 64
    dark = state.get("flashing")
    fg = (30, 30, 30) if dark else (235, 235, 235)
    if not dark:
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (18, 18, 22), -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)

    phase = state.get("phase", "")
    text = {"baseline": "Look at the camera. Measuring baseline...",
            "flash": "FLASH", "recovery": "Keep looking. Recovering...",
            "done": "Complete"}.get(phase, phase)
    cv2.putText(image, text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, fg, 2, cv2.LINE_AA)
    cv2.putText(image, f"{state.get('samples', 0)} samples", (20, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, fg, 1, cv2.LINE_AA)
    x0, x1, y = 20, w - 20, 58
    cv2.rectangle(image, (x0, y), (x1, y + 5), (90, 90, 90), -1)
    fill = int((x1 - x0) * state.get("progress", 0.0))
    cv2.rectangle(image, (x0, y), (x0 + fill, y + 5), (60, 255, 120), -1)


def draw_status_bar(image, left_text: str, right_text: str = "") -> None:
    """Slim translucent status bar along the bottom edge."""
    h, w = image.shape[:2]
    bar_h = 26
    overlay = image.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    cv2.putText(image, left_text, (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (60, 255, 120), 1, cv2.LINE_AA)
    if right_text:
        (tw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(image, right_text, (w - tw - 10, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (180, 180, 180), 1, cv2.LINE_AA)


def draw_rec_indicator(image, elapsed_s: float, blink_on: bool = True) -> None:
    """Blinking red dot + ``REC mm:ss`` in the top-left corner."""
    mm, ss = int(elapsed_s) // 60, int(elapsed_s) % 60
    x, y = 16, 58
    if blink_on:
        cv2.circle(image, (x, y), 7, (0, 0, 235), -1, cv2.LINE_AA)
    cv2.putText(image, f"REC {mm:02d}:{ss:02d}", (x + 16, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def draw_searching(image, text: str = "Searching for a face...") -> None:
    """Centered hint shown when no face is detected."""
    h, w = image.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x, y = (w - tw) // 2, (h + th) // 2
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)


def draw_help_overlay(image, entries) -> None:
    """Full translucent overlay listing keybindings.

    Args:
        entries: list of ``(key, description)`` tuples.
    """
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)

    cv2.putText(image, "Keyboard shortcuts", (40, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 255, 120), 2, cv2.LINE_AA)
    row_y, line_h = 96, 28
    for i, (key, desc) in enumerate(entries):
        y = row_y + i * line_h
        cv2.putText(image, f"{key:>5}", (48, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(image, desc, (120, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(image, "press ? to close", (48, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
