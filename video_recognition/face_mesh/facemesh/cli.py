"""Face Mesh — command line entry point.

Examples
--------
    # Everything on, low-latency, smoothed head pose
    python main.py --live --head-pose --blink --gaze --smooth

    # Driver-monitoring style drowsiness detection
    python main.py --blink --drowsiness

    # Calibrate gaze to the screen, then show a gaze cursor
    python main.py --gaze --calibrate --save-calibration cal.json

    # Track several faces with a panel each
    python main.py --num-faces 3 --blink --multiface

    # Export a session with landmarks, then replay it offline:
    python main.py --source clip.mp4 --blink --gaze --head-pose \
        --export session.jsonl --export-landmarks
    python scripts/replay.py session.jsonl

Interactive keys: q/ESC quit · ? help · m mesh · f features · p points
· k smooth mesh · b blendshapes · h head pose · e blink · g gaze · d drowsiness
· r record · s snapshot.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import (
    BlinkTracker,
    DrowsinessMonitor,
    FaceMeshDetector,
    FaceTracker,
    FPSMeter,
    GazeCalibrator,
    LandmarkSmoother,
    PoseSmoother,
    ThreadedCamera,
    VideoRecorder,
    create_exporter,
    draw_blendshapes_overlay,
    draw_calibration_target,
    draw_drowsiness_banner,
    color_for_id,
    draw_face_label,
    draw_face_landmarks,
    draw_face_panels,
    draw_gaze,
    draw_gaze_cursor,
    draw_head_pose_axes,
    draw_help_overlay,
    draw_medical_panel,
    draw_metrics_panel,
    draw_rec_indicator,
    draw_task_prompt,
    draw_searching,
    draw_status_bar,
    ensure_model,
    gaze_direction,
    grid_targets,
    head_pose_axes_2d,
    head_pose_from_matrix,
    landmarks_to_list,
    resolve_source,
)
from .drawing import FEATURE_GROUPS
from .medical import (
    DISCLAIMER,
    AppearanceTracker,
    get_backend,
    BlinkAnalytics,
    FaceGallery,
    GazeDynamics,
    DynamicSymmetry,
    SymmetryBaseline,
    TaskProtocol,
    HeartRateEstimator,
    MovementAnalyzer,
    QualityGate,
    symmetry_metrics,
)
from .metrics import NOSE_TIP, average_ear, mouth_aspect_ratio


class _LM:
    """Lightweight landmark with x/y/z (for wrapping smoothed coordinates)."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def _to_array(face):
    return np.array([[lm.x, lm.y, lm.z] for lm in face], dtype=float)


def _wrap(coords):
    return [_LM(x, y, z) for x, y, z in coords]


# Keybindings shown in the help overlay.
_HELP = [
    ("q", "quit"), ("?", "toggle this help"), ("m", "mesh on/off"),
    ("f", "cycle features"), ("p", "landmark points"), ("k", "smooth mesh"),
    ("b", "blendshapes"), ("h", "head pose"), ("e", "blink"), ("g", "gaze"),
    ("d", "drowsiness"), ("r", "record"), ("s", "snapshot"),
]

_FEATURE_CYCLE = ["all", "tesselation", "contours", "irises"]
_LEVEL_RANK = {"awake": 0, "drowsy": 1, "ALERT": 2}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time face mesh detection (MediaPipe + OpenCV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", default="0",
                   help="Webcam index (e.g. 0), or path to an image/video.")
    p.add_argument("--model", default=None,
                   help="Path to face_landmarker.task (auto-downloaded if omitted).")
    p.add_argument("--features", default="all", choices=list(FEATURE_GROUPS))
    p.add_argument("--num-faces", type=int, default=1)
    p.add_argument("--live", action="store_true",
                   help="Low-latency LIVE_STREAM mode (webcam only).")
    p.add_argument("--gpu", action="store_true",
                   help="Try the GPU delegate (falls back to CPU if unavailable).")
    p.add_argument("--width", type=int, default=None, help="Requested capture width (webcam).")
    p.add_argument("--height", type=int, default=None, help="Requested capture height (webcam).")
    p.add_argument("--cam-fps", type=float, default=None, help="Requested capture FPS (webcam).")
    p.add_argument("--no-mirror", action="store_true", help="Do not mirror the webcam.")
    p.add_argument("--no-threaded-capture", action="store_true",
                   help="Disable the background capture thread (webcam).")
    p.add_argument("--smooth-mesh", action="store_true",
                   help="1-Euro smoothing of all landmarks (reduces mesh shimmer).")
    p.add_argument("--points", action="store_true")
    p.add_argument("--blendshapes", action="store_true")
    p.add_argument("--head-pose", action="store_true")
    p.add_argument("--smooth", action="store_true",
                   help="1-Euro filter on head-pose angles.")
    p.add_argument("--blink", action="store_true",
                   help="Per-eye EAR, blink counts and rate.")
    p.add_argument("--gaze", action="store_true")
    p.add_argument("--drowsiness", action="store_true",
                   help="PERCLOS / microsleep / yawn monitoring with alerts.")
    p.add_argument("--multiface", action="store_true",
                   help="Show a metrics panel per face (auto when num-faces>1).")
    p.add_argument("--calibrate", action="store_true",
                   help="Run a 9-point gaze calibration first (webcam only).")
    p.add_argument("--save-calibration", default=None,
                   help="Save the fitted gaze calibration to this JSON file.")
    p.add_argument("--load-calibration", default=None,
                   help="Load a gaze calibration JSON (skips --calibrate).")
    p.add_argument("--symmetry", action="store_true",
                   help="Facial symmetry / asymmetry index (palsy-style screening).")
    p.add_argument("--movement", action="store_true",
                   help="Facial expressivity: amplitude + jitter (hypomimia-style).")
    p.add_argument("--heart-rate", action="store_true",
                   help="Contactless heart rate from skin colour (rPPG).")
    p.add_argument("--blink-analytics", action="store_true",
                   help="Blink rate, duration, completeness and per-eye asymmetry.")
    p.add_argument("--gaze-dynamics", action="store_true",
                   help="Saccades, fixation stability, nystagmus screen, pupil proxy.")
    p.add_argument("--reid", action="store_true",
                   help="Track people by facial appearance, not just position.")
    p.add_argument("--gallery", default=None,
                   help="Load/save an identity gallery JSON (implies --reid).")
    p.add_argument("--embedding", default="geometric",
                   choices=["geometric", "onnx", "insightface", "auto"],
                   help="Face-embedding backend for --reid. Deep backends are far "
                        "stronger but are biometric identifiers; see the docs.")
    p.add_argument("--embedding-model", default=None,
                   help="Path to an ArcFace .onnx model (for --embedding onnx).")
    p.add_argument("--medical", action="store_true",
                   help="Enable all clinical-style metrics (screening only).")
    p.add_argument("--baseline", default=None,
                   help="Load a saved symmetry baseline and report change from it.")
    p.add_argument("--record-baseline", default=None,
                   help="Capture a neutral baseline to this JSON file, then exit.")
    p.add_argument("--subject", default=None, help="Subject label stored in the baseline.")
    p.add_argument("--protocol", action="store_true",
                   help="Run the guided task protocol (rest, brows, eyes, smile, cheeks).")
    p.add_argument("--protocol-out", default=None,
                   help="Write the protocol results to this JSON file.")
    p.add_argument("--task", default=None,
                   help="Label the current prompt (e.g. rest, smile, raise-brows). "
                        "Expressivity is only interpretable with a task.")
    p.add_argument("--ear-threshold", type=float, default=0.21)
    p.add_argument("--det-conf", type=float, default=0.5)
    p.add_argument("--track-conf", type=float, default=0.5)
    p.add_argument("--output", default=None)
    p.add_argument("--export", default=None,
                   help="Per-frame metrics to .jsonl or .csv (by extension).")
    p.add_argument("--export-landmarks", action="store_true")
    p.add_argument("--no-display", action="store_true")
    return p.parse_args(argv)


class FaceState:
    """Per-face stateful helpers."""

    def __init__(self, ear_threshold: float) -> None:
        self.blink = BlinkTracker(ear_threshold)
        self.smoother = PoseSmoother()
        self.drowsiness = DrowsinessMonitor(ear_threshold)
        self.movement = MovementAnalyzer()
        self.heart = HeartRateEstimator()
        self.blink_analytics = BlinkAnalytics()
        self.gaze_dynamics = GazeDynamics()
        self.quality = QualityGate()
        self.dynamic_symmetry = DynamicSymmetry()


def _worse(a: str, b: str) -> str:
    return a if _LEVEL_RANK[a] >= _LEVEL_RANK[b] else b


def _process_face(frame, face, matrix, face_id, opts, states, now_s, clean_frame=None):
    """Draw + compute metrics for one face.

    Args:
        face: the (possibly smoothed) landmark list for this face.
        matrix: facial transformation matrix, or None.
        clean_frame: un-annotated frame, required for rPPG skin sampling.

    Returns ``(title, lines, record, level, med_lines)``.
    """
    h, w = frame.shape[:2]
    state = states.setdefault(face_id, FaceState(opts["ear_threshold"]))
    lines: list[str] = []
    med_lines: list[str] = []
    record: dict = {}
    level = "awake"

    if opts["blink"]:
        bm = state.blink.update(face, w, h, now_s)
        record["ear"] = bm["ear"]
        record["blink"] = {k: bm[k] for k in ("left", "right", "total", "rate_per_min")}
        tag = "closed" if bm["closed"] else "open"
        lines.append(f"EAR {bm['ear']:.2f} ({tag})")
        lines.append(f"Blinks {bm['total']} (L{bm['left']}/R{bm['right']})")
        lines.append(f"Rate {bm['rate_per_min']:.0f}/min")

    if opts["gaze"]:
        gaze = gaze_direction(face)
        draw_gaze(frame, face, gaze)
        record["gaze"] = dict(gaze)
        lines.append(f"Gaze {gaze['dir']}")
        cal = opts["calibrator"]
        if cal is not None and cal.is_fitted:
            gx, gy = cal.apply(gaze["h"], gaze["v"])
            draw_gaze_cursor(frame, (gx, gy))
            record["gaze"]["screen"] = [round(gx, 1), round(gy, 1)]

    if opts["head_pose"] and matrix is not None:
        pitch, yaw, roll = head_pose_from_matrix(matrix)
        if opts["smooth"]:
            pitch, yaw, roll = state.smoother.update(pitch, yaw, roll, now_s)
        nose = face[NOSE_TIP]
        draw_head_pose_axes(frame, (nose.x * w, nose.y * h), head_pose_axes_2d(matrix))
        record["head_pose"] = {"pitch": round(pitch, 1), "yaw": round(yaw, 1),
                               "roll": round(roll, 1)}
        lines.append(f"P{pitch:+.0f} Y{yaw:+.0f} R{roll:+.0f}")

    if opts["drowsiness"]:
        ear = record.get("ear") or average_ear(face, w, h)
        mar = mouth_aspect_ratio(face, w, h)
        dm = state.drowsiness.update(ear, mar, now_s)
        level = dm["level"]
        record["drowsiness"] = {k: dm[k] for k in ("perclos", "level", "microsleeps", "yawns")}
        lines.append(f"PERCLOS {dm['perclos']*100:.0f}% [{dm['level']}]")

    # --- Clinical-style metrics (screening only, quality-gated) ---
    med: dict = {}
    any_medical = opts["symmetry"] or opts["movement"] or opts["heart_rate"]
    if any_medical:
        q = state.quality.check(face, w, h, now_s, frame_bgr=clean_frame, matrix=matrix)
        med["quality"] = q
        if not q["ok"]:
            # Confidently wrong output is worse than none: suppress and say why.
            if face_id == 0:
                med_lines.append(f"Metrics paused: {', '.join(q['reasons'])}")
            record["medical"] = med
            if opts["export_landmarks"]:
                record["landmarks"] = landmarks_to_list(face)
            return f"Face {face_id}", lines, record, level, med_lines, med

    if opts["symmetry"]:
        sym = symmetry_metrics(face, w, h)
        dyn = state.dynamic_symmetry.update(sym, now_s)
        med["symmetry"] = {**sym, **dyn}
        bl = opts.get("baseline")
        if bl is not None and bl.is_fitted:
            rel = bl.compare(sym)
            med["symmetry"].update(rel)
        if face_id == 0:
            if bl is not None and bl.is_fitted:
                med_lines.append(f"Symmetry {rel['relative_symmetry']*100:.0f}% vs baseline "
                                 f"({rel['worst_delta_region']})")
            else:
                med_lines.append(f"Symmetry {sym['symmetry']*100:.0f}%  (worst: {sym['worst_region']})")
            med_lines.append(f"Peak asym {dyn['peak_asymmetry']:.3f} over {dyn['samples']}f")

    if opts["blink_analytics"]:
        ba = state.blink_analytics.update(face, w, h, now_s)
        med["blink"] = ba
        if face_id == 0:
            med_lines.append(f"Blink {ba['rate_per_min']:.0f}/min "
                             f"dur {ba['mean_duration_s']*1000:.0f}ms")
            if ba["high_incomplete"]:
                med_lines.append(f"Incomplete blinks {ba['incomplete_fraction']*100:.0f}%")
            if ba["low_rate"]:
                med_lines.append("Low blink rate")
            if ba["closure_asymmetry"] > 0.2 and ba["weaker_eye"]:
                med_lines.append(f"Weaker closure: {ba['weaker_eye']} eye")

    if opts["gaze_dynamics"]:
        gd = state.gaze_dynamics.update(face, w, h, now_s)
        med["gaze_dynamics"] = gd
        if face_id == 0:
            med_lines.append(f"Saccades {gd['saccades_in_window']} "
                             f"fix {gd['fixation_stability']:.3f}")
            if gd["nystagmus_suspected"]:
                med_lines.append(f"Oscillation {gd['nystagmus_hz']:.1f} Hz")
            if gd["pupil_asymmetry"] > 0.02:
                med_lines.append(f"Pupil asym {gd['pupil_asymmetry']:.3f}")

    if opts["movement"]:
        state.movement.set_task(opts.get("current_task") or opts.get("task"))
        mv = state.movement.update(face, w, h, now_s)
        med["movement"] = mv
        if face_id == 0:
            if not mv["interpretable"]:
                med_lines.append(f"Expressivity {mv['expressivity']*100:.0f}% (no task set)")
            else:
                flag = " LOW" if mv["reduced_expressivity"] else ""
                med_lines.append(f"Expressivity {mv['expressivity']*100:.0f}%{flag} [{mv['task']}]")
            if mv["tremor_detected"]:
                med_lines.append(f"Tremor {mv['tremor_hz']:.1f} Hz ({mv['tremor_ratio']:.2f})")

    if opts["heart_rate"] and clean_frame is not None:
        hr = state.heart.update(clean_frame, face, now_s)
        med["heart_rate"] = hr
        if face_id == 0:
            if hr["reliable"]:
                med_lines.append(f"Heart rate {hr['bpm']:.0f} bpm (conf {hr['confidence']:.2f})")
                if hr.get("rmssd_ms"):
                    med_lines.append(f"HRV rmssd {hr['rmssd_ms']:.0f} ms")
                if hr.get("respiration_reliable"):
                    med_lines.append(f"Respiration {hr['respiration_rpm']:.0f} /min")
                elif hr.get("respiration_rpm"):
                    med_lines.append("Respiration: low confidence")
            elif hr["motion_rejected"]:
                med_lines.append("Heart rate: holding (motion)")
            elif hr.get("lighting_rejected"):
                med_lines.append(f"Heart rate: holding ({', '.join(hr['lighting_reasons'])})")
            elif hr["ready"]:
                med_lines.append("Heart rate: low confidence")
            else:
                med_lines.append("Heart rate: measuring...")
    if med:
        record["medical"] = med

    if opts["export_landmarks"]:
        record["landmarks"] = landmarks_to_list(face)

    return f"Face {face_id}", lines, record, level, med_lines, med


def _matrix_for(result, idx):
    mats = result.facial_transformation_matrixes
    return mats[idx] if mats else None


def _hud(frame, text: str) -> None:
    cv2.putText(frame, text, (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 120), 1, cv2.LINE_AA)


def _load_baseline(args):
    if not args.baseline:
        return None
    bl = SymmetryBaseline.load(args.baseline)
    who = f" [{bl.subject}]" if bl.subject else ""
    print(f"Loaded symmetry baseline{who} <- {args.baseline}")
    return bl


def _opts_from_args(args, calibrator) -> dict:
    return {
        "blink": args.blink, "gaze": args.gaze, "head_pose": args.head_pose,
        "smooth": args.smooth, "drowsiness": args.drowsiness,
        "symmetry": args.symmetry or args.medical or args.protocol,
        "movement": args.movement or args.medical or args.protocol,
        "heart_rate": args.heart_rate or args.medical,
        "blink_analytics": args.blink_analytics or args.medical,
        "gaze_dynamics": args.gaze_dynamics or args.medical,
        "task": args.task,
        "baseline": _load_baseline(args),
        "multiface": args.multiface or args.num_faces > 1,
        "ear_threshold": args.ear_threshold, "calibrator": calibrator,
        "export_landmarks": bool(args.export and args.export_landmarks),
    }


def _render_panels(frame, panels, colors, opts, level) -> None:
    if opts["multiface"] and len(panels) > 1:
        draw_face_panels(frame, panels, colors)
    elif panels:
        draw_metrics_panel(frame, panels[0][1])
    if opts["drowsiness"]:
        draw_drowsiness_banner(frame, level)


def run_calibration(model_path, args, source):
    """Interactive 9-point gaze calibration on the webcam. Returns a fitted
    :class:`GazeCalibrator`, or None if aborted."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return None
    h, w = frame.shape[:2]
    cal = GazeCalibrator(degree=2)
    with FaceMeshDetector(model_path, running_mode="video",
                          min_face_detection_confidence=args.det_conf) as det:
        for (tx, ty) in grid_targets(w, h):
            samples: list[tuple[float, float]] = []
            start = time.monotonic()
            while (elapsed := time.monotonic() - start) < 1.6:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                res = det.detect(frame, timestamp_ms=int(time.monotonic() * 1000))
                if res.face_landmarks and elapsed > 0.5:  # skip the saccade in
                    g = gaze_direction(res.face_landmarks[0])
                    samples.append((g["h"], g["v"]))
                draw_calibration_target(frame, (tx, ty), elapsed / 1.6)
                draw_status_bar(frame, "Gaze calibration — look at the dot", "ESC to cancel")
                cv2.imshow("Face Mesh", frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    cap.release()
                    return None
            if samples:
                mh = sum(s[0] for s in samples) / len(samples)
                mv = sum(s[1] for s in samples) / len(samples)
                cal.add_sample(mh, mv, tx, ty)
    cap.release()
    min_pts = 6  # quadratic
    if cal.n_samples >= min_pts:
        rms = cal.fit()
        print(f"Gaze calibration fitted — validation error {rms:.0f}px")
        if args.save_calibration:
            cal.save(args.save_calibration)
            print(f"Saved gaze calibration -> {args.save_calibration}")
        return cal
    print("calibration failed: not enough samples.", file=sys.stderr)
    return None


def run_record_baseline(args, model_path: Path) -> int:
    """Capture a neutral resting baseline of facial symmetry, then save it."""
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("error: cannot open camera", file=sys.stderr)
        return 2

    bl = SymmetryBaseline(subject=args.subject)
    gate = QualityGate()
    target, start = 60, time.monotonic()
    print("Recording a NEUTRAL baseline. Rest your face, look at the camera.")

    with FaceMeshDetector(model_path, running_mode="video",
                          min_face_detection_confidence=args.det_conf) as det:
        while bl.n_samples < target:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            now = time.monotonic() - start
            res = det.detect(frame, timestamp_ms=int(now * 1000))
            msg = "no face"
            if res.face_landmarks:
                face = res.face_landmarks[0]
                h, w = frame.shape[:2]
                q = gate.check(face, w, h, now, frame_bgr=frame)
                if q["ok"]:
                    bl.add_sample(symmetry_metrics(face, w, h))
                    msg = f"captured {bl.n_samples}/{target}"
                else:
                    msg = ", ".join(q["reasons"])
            draw_status_bar(frame, f"Baseline: {msg}", "ESC to cancel")
            cv2.imshow("Face Mesh - baseline", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                cap.release()
                cv2.destroyAllWindows()
                print("cancelled.")
                return 1

    cap.release()
    cv2.destroyAllWindows()
    if bl.n_samples < 10:
        print("error: not enough good frames captured.", file=sys.stderr)
        return 1
    bl.fit()
    path = bl.save(args.record_baseline)
    print(f"Saved baseline ({bl.n_samples} samples) -> {path}")
    return 0


def run_image(args, model_path: Path) -> int:
    img = cv2.imread(args.source)
    if img is None:
        print(f"error: could not read image {args.source!r}", file=sys.stderr)
        return 2

    calibrator = GazeCalibrator.load(args.load_calibration) if args.load_calibration else None
    with FaceMeshDetector(
        model_path, running_mode="image", num_faces=args.num_faces,
        min_face_detection_confidence=args.det_conf,
        output_blendshapes=args.blendshapes,
        output_transformation_matrixes=args.head_pose,
        delegate="gpu" if args.gpu else "cpu",
    ) as det:
        result = det.detect(img)

    features = FEATURE_GROUPS[args.features]
    opts, states = _opts_from_args(args, calibrator), {}
    clean = img.copy()  # un-annotated copy for rPPG skin sampling
    panels, records, colors, level = [], [], [], "awake"
    med_lines: list[str] = []
    for idx, face in enumerate(result.face_landmarks):
        draw_face_landmarks(img, face, features, draw_points=args.points)
        if opts["multiface"]:
            draw_face_label(img, face, f"#{idx}", color_for_id(idx))
        title, lines, rec, lvl, meds, _ = _process_face(
            img, face, _matrix_for(result, idx), idx, opts, states, 0.0, clean)
        panels.append((title, lines))
        colors.append(color_for_id(idx))
        records.append(rec)
        level = _worse(level, lvl)
        if idx == 0:
            med_lines = meds

    if args.blendshapes and result.face_blendshapes:
        draw_blendshapes_overlay(img, result.face_blendshapes[0])
    _render_panels(img, panels, colors, opts, level)
    if med_lines:
        draw_medical_panel(img, med_lines, DISCLAIMER)

    print(f"Detected {len(result.face_landmarks)} face(s).")
    if args.export:
        with create_exporter(args.export) as exp:
            exp.write_frame(0, records)
        print(f"Exported metrics -> {args.export}")

    out = args.output or str(Path(args.source).with_name(Path(args.source).stem + "_mesh.jpg"))
    cv2.imwrite(out, img)
    print(f"Saved -> {out}")

    if not args.no_display:
        try:
            cv2.imshow("Face Mesh (press any key to close)", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    return 0


def _open_camera(args, kind, value):
    """Return (cam, threaded, mirror) — ThreadedCamera for webcam else VideoCapture."""
    mirror = kind == "webcam" and not args.no_mirror
    threaded = kind == "webcam" and not args.no_threaded_capture
    if threaded:
        cam = ThreadedCamera(value, width=args.width, height=args.height,
                             fps=args.cam_fps, mirror=mirror).start()
    else:
        cam = cv2.VideoCapture(value)
        if kind == "webcam":
            if args.width:
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            if args.height:
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return cam, threaded, mirror


def run_stream(args, model_path: Path, kind: str, value) -> int:
    # Gaze calibration first (opens/uses its own capture, then releases it).
    calibrator = None
    if args.load_calibration:
        calibrator = GazeCalibrator.load(args.load_calibration)
        note = f" (validation {calibrator.rms_error:.0f}px)" if calibrator.rms_error else ""
        print(f"Loaded gaze calibration <- {args.load_calibration}{note}")
    elif args.calibrate and kind == "webcam" and not args.no_display:
        calibrator = run_calibration(model_path, args, value)

    cam, threaded, mirror = _open_camera(args, kind, value)
    if not cam.isOpened():
        print(f"error: could not open source {value!r}", file=sys.stderr)
        return 2

    src_fps = 0.0 if threaded else (cam.get(cv2.CAP_PROP_FPS) or 0.0)

    if args.live and kind != "webcam":
        print("note: --live is ignored for files; using VIDEO mode.", file=sys.stderr)
    running_mode = "live_stream" if (args.live and kind == "webcam") else "video"

    exporter = create_exporter(args.export) if args.export else None
    recorder = VideoRecorder(fps=src_fps or 25.0)
    writer = None  # for --output; created lazily once frame size is known

    display = not args.no_display
    features = list(FEATURE_GROUPS[args.features])
    feature_idx = _FEATURE_CYCLE.index(args.features) if args.features in _FEATURE_CYCLE else 0
    mesh_on, points_on, blend_on = True, args.points, args.blendshapes
    smooth_mesh_on, help_on = args.smooth_mesh, False
    opts = _opts_from_args(args, calibrator)
    fps = FPSMeter()
    states: dict[int, FaceState] = {}
    use_reid = args.reid or bool(args.gallery)
    reid = None
    if use_reid:
        gal = FaceGallery.load(args.gallery) if (args.gallery and Path(args.gallery).exists()) else None
        if gal is not None:
            print(f"Loaded identity gallery ({len(gal.names)} people) <- {args.gallery}")
        backend = get_backend(args.embedding, args.embedding_model)
        if backend.name != "geometric":
            print(f"Using {backend.name} embeddings. Note: deep face embeddings are "
                  f"biometric identifiers - obtain consent and store them securely.")
        reid = AppearanceTracker(gallery=gal, backend=backend)
    tracker = FaceTracker()
    smoother = LandmarkSmoother()
    protocol = TaskProtocol() if args.protocol else None
    proto_state = None
    frame_idx, snapshots = 0, 0
    start = time.monotonic()

    with FaceMeshDetector(
        model_path, running_mode=running_mode, num_faces=args.num_faces,
        min_face_detection_confidence=args.det_conf,
        min_tracking_confidence=args.track_conf,
        output_blendshapes=args.blendshapes,
        output_transformation_matrixes=args.head_pose,
        delegate="gpu" if args.gpu else "cpu",
    ) as det:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                if threaded:  # camera warming up or reconnecting; stay responsive
                    if display and (cv2.waitKey(30) & 0xFF) in (ord("q"), 27):
                        break
                    continue
                break  # end of a video file

            now_s = time.monotonic() - start
            if kind == "video" and src_fps > 0:
                ts_ms = int(frame_idx / src_fps * 1000)
                now_s = frame_idx / src_fps
            else:
                ts_ms = int(now_s * 1000)

            result = det.detect(frame, timestamp_ms=ts_ms)
            face_ids = tracker.update(result.face_landmarks)
            names = (reid.update(result.face_landmarks, frame.shape[1], frame.shape[0], frame)
                     if reid is not None else None)

            need_clean = opts["heart_rate"] or opts["symmetry"] or opts["movement"]
            clean = frame.copy() if need_clean else None
            panels, records, colors, level = [], [], [], "awake"
            med_lines: list[str] = []
            for idx, face in enumerate(result.face_landmarks):
                fid = face_ids[idx]
                if smooth_mesh_on:
                    face = _wrap(smoother.apply(fid, _to_array(face), now_s))
                if mesh_on:
                    draw_face_landmarks(frame, face, features, draw_points=points_on)
                label = names[idx] if names else f"#{fid}"
                if opts["multiface"] or names:
                    draw_face_label(frame, face, label, color_for_id(fid))
                title, lines, rec, lvl, meds, med_raw = _process_face(
                    frame, face, _matrix_for(result, idx), fid, opts, states, now_s, clean)
                if protocol is not None and idx == 0:
                    proto_state = protocol.update(
                        now_s, med_raw.get("symmetry"), med_raw.get("movement"),
                        med_raw.get("quality"))
                    opts["current_task"] = proto_state.get("task")
                panels.append((title, lines))
                colors.append(color_for_id(fid))
                records.append(rec)
                level = _worse(level, lvl)
                if idx == 0:
                    med_lines = meds

            if blend_on and result.face_blendshapes:
                draw_blendshapes_overlay(frame, result.face_blendshapes[0])
            _render_panels(frame, panels, colors, opts, level)
            if med_lines:
                draw_medical_panel(frame, med_lines, DISCLAIMER)
            if proto_state is not None:
                draw_task_prompt(frame, proto_state)
            if not result.face_landmarks:
                draw_searching(frame)
            if exporter is not None:
                exporter.write_frame(frame_idx, records)

            mode_tag = "live" if running_mode == "live_stream" else running_mode
            extras = "".join(t for t, on in [(" smooth", smooth_mesh_on)] if on)
            draw_status_bar(
                frame,
                f"{fps.tick():4.1f} FPS | faces:{len(result.face_landmarks)} "
                f"| {_FEATURE_CYCLE[feature_idx]} | {mode_tag}{extras}",
                "? help",
            )
            if recorder.is_recording:
                draw_rec_indicator(frame, recorder.elapsed(), blink_on=int(now_s * 2) % 2 == 0)

            # --output writer (whole run); created once we know the frame size.
            if args.output and writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                         src_fps or 25.0, (w, h))
            if writer is not None:
                writer.write(frame)
            recorder.write(frame)  # on-demand recording (no-op unless active)

            if help_on:
                draw_help_overlay(frame, _HELP)

            if display:
                try:
                    cv2.imshow("Face Mesh", frame)
                except cv2.error:
                    display = False
                else:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    elif key == ord("m"):
                        mesh_on = not mesh_on
                    elif key == ord("p"):
                        points_on = not points_on
                    elif key == ord("k"):
                        smooth_mesh_on = not smooth_mesh_on
                    elif key == ord("b"):
                        blend_on = not blend_on
                    elif key == ord("h"):
                        opts["head_pose"] = not opts["head_pose"]
                    elif key == ord("e"):
                        opts["blink"] = not opts["blink"]
                    elif key == ord("g"):
                        opts["gaze"] = not opts["gaze"]
                    elif key == ord("d"):
                        opts["drowsiness"] = not opts["drowsiness"]
                    elif key == ord("?"):
                        help_on = not help_on
                    elif key == ord("n") and protocol is not None:
                        protocol.skip()
                    elif key == ord("r"):
                        on, path = recorder.toggle((frame.shape[1], frame.shape[0]), now_s)
                        print(f"Recording started -> {path}" if on else f"Recording saved -> {path}")
                    elif key == ord("f"):
                        feature_idx = (feature_idx + 1) % len(_FEATURE_CYCLE)
                        features = list(FEATURE_GROUPS[_FEATURE_CYCLE[feature_idx]])
                    elif key == ord("s"):
                        snap = f"snapshot_{snapshots:03d}.jpg"
                        cv2.imwrite(snap, frame)
                        print(f"Saved {snap}")
                        snapshots += 1

            if protocol is not None and protocol.finished:
                break

            frame_idx += 1

    if threaded:
        cam.stop()
    else:
        cam.release()
    if recorder.is_recording:
        print(f"Recording saved -> {recorder.stop()}")
    if writer is not None:
        writer.release()
        print(f"Saved -> {args.output} ({frame_idx} frames)")
    if exporter is not None:
        exporter.close()
        print(f"Exported metrics -> {args.export} ({frame_idx} frames)")
    if reid is not None and args.gallery:
        reid.gallery.save(args.gallery)
        print(f"Saved identity gallery ({len(reid.gallery.names)} people) -> {args.gallery}")
    if protocol is not None:
        summary = protocol.summary()
        print("\nProtocol results (screening only, not a diagnosis):")
        for name, r in summary["tasks"].items():
            print(f"  {name:12s} peak_asym={r['peak_asymmetry']:.3f} "
                  f"peak_expr={r['peak_expressivity']:.2f} frames={r['frames']}")
        roll = summary["summary"]
        if "worst_task" in roll:
            print(f"  worst task: {roll['worst_task']} ({roll['worst_task_region']})")
        if args.protocol_out:
            Path(args.protocol_out).write_text(json.dumps(summary, indent=2))
            print(f"Saved -> {args.protocol_out}")
    cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    # Multi-face mode implies tracking several people; raise the cap if left default.
    if args.multiface and args.num_faces == 1:
        args.num_faces = 5
    model_path = ensure_model(args.model) if args.model else ensure_model()
    if args.record_baseline:
        return run_record_baseline(args, model_path)
    kind, value = resolve_source(args.source)
    if kind == "image":
        return run_image(args, model_path)
    return run_stream(args, model_path, kind, value)


if __name__ == "__main__":
    raise SystemExit(main())
