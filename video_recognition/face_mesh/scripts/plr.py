"""Run a pupillary-light-reflex trial.

    python scripts/plr.py                       # one trial
    python scripts/plr.py --trials 3 -o plr.json

Sit in a DIM room, ~50 cm from the screen, looking at the camera. The window
shows your face for a few seconds (baseline), flashes white (stimulus), then
returns while your pupils recover.

READ THIS BEFORE BELIEVING ANY NUMBER
-------------------------------------
This is a toy reproduction, not pupillometry.

  * MediaPipe gives an IRIS ring, not the pupil. On a dark iris in visible
    light the pupil boundary is not resolved at all, so the measured diameter
    may barely move even when your pupil constricts normally.
  * A 30 fps webcam samples a ~250 ms latency with ~7 frames.
  * Screen brightness is uncalibrated.

A "no response" result here most likely means the method cannot see the pupil,
NOT that your reflex is absent. Never compare across sessions, devices or
people. If you have a real concern about your pupils, see a clinician.

SCREENING / EDUCATIONAL USE ONLY - not a diagnostic device.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from facemesh import FaceMeshDetector, ensure_model  # noqa: E402
from facemesh.drawing import draw_flash, draw_plr_status  # noqa: E402
from facemesh.medical import DISCLAIMER, PLRSession  # noqa: E402
from facemesh.medical.pupil import LEFT_IRIS, RIGHT_IRIS, iris_diameter  # noqa: E402


def _diameters(face, w, h):
    pr = np.array([face[33].x * w, face[33].y * h])
    pl = np.array([face[263].x * w, face[263].y * h])
    iod = float(np.linalg.norm(pr - pl))
    if iod <= 0:
        return 0.0, 0.0
    return (iris_diameter(face, LEFT_IRIS, w, h, iod),
            iris_diameter(face, RIGHT_IRIS, w, h, iod))


def run_trial(det, cap, args) -> dict | None:
    sess = PLRSession(args.baseline, args.flash, args.recovery)
    start = time.monotonic()
    while not sess.finished:
        ok, frame = cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        now = time.monotonic() - start
        h, w = frame.shape[:2]
        res = det.detect(frame, timestamp_ms=int(now * 1000))
        dl = dr = 0.0
        if res.face_landmarks:
            dl, dr = _diameters(res.face_landmarks[0], w, h)
        state = sess.update(now, dl, dr)

        if state["flashing"]:
            draw_flash(frame)
        draw_plr_status(frame, state)
        cv2.imshow("Pupillary light reflex - ESC to cancel", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            return None
    return sess.result


def _print(res: dict) -> None:
    if not res.get("valid"):
        print(f"  invalid: {res.get('reason', 'unknown')}")
        return
    for eye in ("left", "right"):
        e = res[eye]
        print(f"  {eye:<6} constriction {e['constriction_pct']:5.1f}%  "
              f"latency {e['latency_s'] if e['latency_s'] is not None else float('nan'):.3f}s  "
              f"t_min {e['time_to_min_s']:.2f}s  responded={e['responded']}")
    print(f"  asymmetry {res['response_asymmetry_pct']:.1f}%  "
          f"weaker={res['weaker_response']}  both_responded={res['both_responded']}")
    if res.get("warnings"):
        print("  WARNINGS:")
        for w in res["warnings"]:
            print(f"    - {w}")
    if not res.get("measurement_trustworthy"):
        print(f"  {res.get('interpretation', '')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pupillary light reflex (toy).")
    ap.add_argument("--source", default="0")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--baseline", type=float, default=3.0)
    ap.add_argument("--flash", type=float, default=0.5)
    ap.add_argument("--recovery", type=float, default=5.0)
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args(argv)

    print(__doc__.split("SCREENING")[0].split("READ THIS")[1])
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("error: cannot open camera", file=sys.stderr)
        return 2

    results = []
    with FaceMeshDetector(ensure_model(), running_mode="video") as det:
        for i in range(args.trials):
            print(f"\nTrial {i + 1}/{args.trials} - sit still, look at the camera")
            res = run_trial(det, cap, args)
            if res is None:
                print("cancelled.")
                break
            results.append(res)
            _print(res)
            if i + 1 < args.trials:
                time.sleep(2.0)   # let pupils redilate fully

    cap.release()
    cv2.destroyAllWindows()

    if args.output and results:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nSaved -> {args.output}")
    print(f"\n{DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
