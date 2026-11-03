<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Computer%20Vision-MediaPipe-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/Framework-OpenCV-darkgreen?style=flat-square&logo=opencv" />
  <img src="https://img.shields.io/badge/Tracking-478%20Facial%20Landmarks-informational?style=flat-square" />
  <img src="https://img.shields.io/badge/Head%20Pose-Pitch%20%7C%20Yaw%20%7C%20Roll-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Gaze-Screen%20Calibration-blueviolet?style=flat-square" />
  <img src="https://img.shields.io/badge/Drowsiness-PERCLOS%20Detection-red?style=flat-square" />
  <img src="https://img.shields.io/badge/Multi--Face-Supported-009688?style=flat-square" />
  <img src="https://img.shields.io/badge/Acceleration-GPU%20Delegate-success?style=flat-square" />
  <img src="https://img.shields.io/badge/Export-CSV%20%7C%20JSONL-lightgrey?style=flat-square" />
  <img src="https://img.shields.io/badge/Replay-Offline%20Viewer-795548?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-v1.4.0-brightgreen?style=flat-square" />
</p>

# Face Mesh

Real-time facial landmark detection and analysis in Python, built on
**MediaPipe** (Tasks API) and **OpenCV**. Tracks **478 3D landmarks** per face
and layers on head pose, blink/gaze, drowsiness, and multi-face identity —
from a webcam, image, or video.

**v1.1.0** builds on the first stable release with a higher-level live pipeline:
threaded capture, whole-mesh smoothing, in-app recording, and an on-screen HUD —
on top of the installable package, CI test suite, gaze calibration, timeline
plots, and per-person tracking from v1.0.

![mesh demo](render_demo.png)

> The image above comes from `scripts/render_demo.py`, a no-camera self-test.
> Real input produces a proper face-shaped mesh.

## Features

- 478-point mesh (tesselation, contours, irises) with selectable feature sets
- Image, video, and low-latency live-stream modes; optional GPU delegate
- Higher-level live cam: threaded capture, mesh smoothing, in-app recording, on-screen HUD
- Head pose (pitch/yaw/roll) with a 3D gizmo and 1-Euro smoothing
- Per-eye blink counts + blinks-per-minute rate
- Gaze estimation with optional screen calibration (affine or quadratic)
- Drowsiness monitoring — PERCLOS, microsleep and yawn alerts
- Multi-face tracking with per-person color-coded ID labels and a panel each
- Export to JSONL / CSV, with an offline replay viewer and timeline plots

## Install

```bash
pip install .                 # or: pip install -e ".[dev]" for tests + plots
```

This installs the `facemesh` command. To run straight from the source tree
without installing, use `python main.py` instead.

## Quick start

```bash
facemesh                                   # live webcam mesh
facemesh --live --head-pose --blink --gaze --smooth   # everything on
facemesh --blink --drowsiness              # driver-monitoring style
facemesh --gaze --calibrate                # 9-point gaze calibration
facemesh --smooth-mesh --width 1280 --height 720   # smoothed mesh at 720p
facemesh --multiface                       # tag several people at once
facemesh --source photo.jpg                # annotate an image
```

Run `facemesh --help` for the full list of options. No webcam? Verify your
install with `python scripts/render_demo.py`.

Interactive keys: `q` quit · `?` help · `m` mesh · `f` features · `k` smooth mesh ·
`h` head pose · `e` blink · `g` gaze · `d` drowsiness · `r` record · `s` snapshot.

## Architecture

```mermaid
flowchart TD
    SRC["Webcam / Image / Video"] --> DET["FaceMeshDetector<br/>(MediaPipe Tasks)"]
    DET -->|"478 landmarks<br/>+ pose matrix + blendshapes"| TRK["Face Tracker<br/>(stable IDs)"]
    TRK --> PROC{"Per-face processing"}
    PROC --> MET["Metrics<br/>EAR / gaze / MAR / head pose"]
    PROC --> FIL["Filters<br/>1-Euro smoothing"]
    PROC --> CAL["Gaze Calibration<br/>screen mapping"]
    PROC --> DRO["Drowsiness<br/>PERCLOS / microsleep / yawns"]
    MET --> RND["Rendering<br/>mesh / panels / banners / cursor"]
    FIL --> RND
    CAL --> RND
    DRO --> RND
    RND --> DISP["Live display"]
    MET --> EXP["Export<br/>JSONL / CSV"]
    EXP --> RPL["Replay viewer"]
    EXP --> PLT["Timeline plots"]

    classDef input fill:#1565c0,stroke:#0d47a1,color:#fff;
    classDef core fill:#6a1b9a,stroke:#4a148c,color:#fff;
    classDef proc fill:#2e7d32,stroke:#1b5e20,color:#fff;
    classDef out fill:#e65100,stroke:#bf360c,color:#fff;
    class SRC input;
    class DET,TRK core;
    class PROC,MET,FIL,CAL,DRO proc;
    class RND,DISP,EXP,RPL,PLT out;
```

## Analyze a session

Export with landmarks, then replay or plot it offline — no camera or model:

```bash
facemesh --source clip.mp4 --blink --drowsiness \
    --export session.jsonl --export-landmarks
python scripts/replay.py session.jsonl          # re-render the mesh
python scripts/plot_session.py session.jsonl    # EAR / blink / PERCLOS timeline
```

## As a library

```python
import cv2
from facemesh import FaceMeshDetector, draw_face_landmarks, ensure_model

model = ensure_model()                 # downloads/caches the .task bundle
img = cv2.imread("photo.jpg")
with FaceMeshDetector(model, running_mode="image") as det:
    result = det.detect(img)
for face in result.face_landmarks:     # each face = 478 normalized landmarks
    draw_face_landmarks(img, face, ["all"])
cv2.imwrite("out.jpg", img)
```

## Notes

MediaPipe's `FaceLandmarker` returns 478 normalized landmarks per face; the
renderer maps them to pixels over the published connection topologies. The
legacy `mp.solutions.face_mesh` API was removed in MediaPipe 0.10.30+, so this
project targets the current **Tasks API**, which needs the `face_landmarker.task`
bundle — downloaded automatically on first run.

## Roadmap

Shipped through v1.1.0: head pose, blink/gaze, drowsiness, live-stream, GPU
delegate, export, replay, calibration, multi-face tracking, tests, and a
threaded live pipeline with mesh smoothing, recording and a HUD.
Next: appearance-based re-identification, a validation UI for calibration, and
richer session analytics.

## License

MIT.
