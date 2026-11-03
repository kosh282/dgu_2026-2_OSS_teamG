"""Replay an exported session — no camera or model required.

Reads a JSONL file produced by ``main.py --export session.jsonl
--export-landmarks`` and re-renders each frame's mesh and metrics onto a blank
canvas, either to a live window or out to a video file. Handy for reviewing a
capture, generating figures, or debugging metrics offline.

    python scripts/replay.py session.jsonl
    python scripts/replay.py session.jsonl --output replay.mp4 --no-display

Records without landmarks still replay their metrics panel (no mesh).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facemesh.drawing import FEATURE_GROUPS, draw_face_landmarks, draw_face_panels


@dataclass
class _LM:
    x: float
    y: float
    z: float = 0.0


def _panel_lines(idx: int, face: dict) -> tuple[str, list[str]]:
    """Build (title, lines) for one face's stored metrics."""
    lines: list[str] = []
    if "ear" in face:
        lines.append(f"EAR {face['ear']:.2f}")
    if "blink" in face:
        b = face["blink"]
        lines.append(f"Blinks {b['total']} (L{b['left']}/R{b['right']})")
        lines.append(f"Rate {b['rate_per_min']:.0f}/min")
    if "gaze" in face:
        lines.append(f"Gaze {face['gaze']['dir']}")
    if "head_pose" in face:
        p = face["head_pose"]
        lines.append(f"P{p['pitch']:+.0f} Y{p['yaw']:+.0f} R{p['roll']:+.0f}")
    return f"Face {idx}", lines


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay an exported .jsonl session.")
    p.add_argument("input", help="Path to the exported .jsonl file.")
    p.add_argument("--output", default=None, help="Write the replay to this video file.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--features", default="all", choices=list(FEATURE_GROUPS))
    p.add_argument("--no-display", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    path = Path(args.input)
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 2
    if path.suffix.lower() != ".jsonl":
        print("error: replay expects a .jsonl file (export with --export x.jsonl).",
              file=sys.stderr)
        return 2

    W, H = args.width, args.height
    features = FEATURE_GROUPS[args.features]
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (W, H))

    display = not args.no_display
    frames = 0
    had_landmarks = False
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            canvas = np.full((H, W, 3), 18, dtype=np.uint8)
            panels = []
            for idx, face in enumerate(record.get("faces", [])):
                if "landmarks" in face:
                    had_landmarks = True
                    lms = [_LM(x, y, z) for x, y, z in face["landmarks"]]
                    draw_face_landmarks(canvas, lms, features)
                panels.append(_panel_lines(idx, face))
            if panels:
                draw_face_panels(canvas, panels)

            if writer is not None:
                writer.write(canvas)
            if display:
                try:
                    cv2.imshow("Replay", canvas)
                except cv2.error:
                    display = False
                else:
                    if (cv2.waitKey(int(1000 / args.fps)) & 0xFF) in (ord("q"), 27):
                        break
            frames += 1

    if writer is not None:
        writer.release()
        print(f"Wrote {frames} frames -> {args.output}")
    cv2.destroyAllWindows()
    if frames and not had_landmarks:
        print("note: no landmarks in this file; re-export with --export-landmarks "
              "to see the mesh.", file=sys.stderr)
    print(f"Replayed {frames} frame(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
