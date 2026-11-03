"""Plot session metrics over time from an exported file.

Reads a ``.jsonl`` or ``.csv`` session (from ``main.py --export ...``) and draws
timelines of EAR, blink count, PERCLOS and gaze — whichever are present — for
the primary face, saving a PNG.

    python scripts/plot_session.py session.jsonl
    python scripts/plot_session.py metrics.csv --output timeline.png

Requires matplotlib:  pip install "face-mesh[plot]"  (or  pip install matplotlib)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")


def _load(path: Path):
    """Yield (frame_index, primary_face_dict) from JSONL or CSV."""
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            faces = rec.get("faces", [])
            yield rec.get("frame", 0), (faces[0] if faces else {})
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("face") in ("", None):
                    yield int(row["frame"]), {}
                    continue
                if int(row["face"]) != 0:
                    continue
                face = {}
                if row.get("ear"):
                    face["ear"] = float(row["ear"])
                if row.get("blink_total"):
                    face["blink"] = {"total": float(row["blink_total"])}
                if row.get("drowsiness_perclos"):
                    face["drowsiness"] = {"perclos": float(row["drowsiness_perclos"])}
                yield int(row["frame"]), face
    else:
        sys.exit("input must be .jsonl or .csv")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Plot exported session metrics over time.")
    ap.add_argument("input")
    ap.add_argument("--output", default=None, help="PNG path (default: <input>_timeline.png)")
    args = ap.parse_args(argv)

    path = Path(args.input)
    if not path.exists():
        sys.exit(f"{path} not found")

    frames, ear, blinks, perclos = [], [], [], []
    for f, face in _load(path):
        frames.append(f)
        ear.append(face.get("ear"))
        blinks.append(face.get("blink", {}).get("total"))
        perclos.append(face.get("drowsiness", {}).get("perclos"))

    series = [("Eye Aspect Ratio", ear, "#2e7d32"),
              ("Blink count", blinks, "#1565c0"),
              ("PERCLOS", perclos, "#c62828")]
    series = [(n, v, c) for n, v, c in series if any(x is not None for x in v)]
    if not series:
        sys.exit("no plottable metrics found — export with --blink / --drowsiness.")

    fig, axes = plt.subplots(len(series), 1, figsize=(10, 2.4 * len(series)), sharex=True)
    if len(series) == 1:
        axes = [axes]
    for ax, (name, vals, color) in zip(axes, series):
        ax.plot(frames, vals, color=color, linewidth=1.5)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("frame")
    fig.suptitle(f"Session timeline — {path.name}")
    fig.tight_layout()

    out = args.output or str(path.with_name(path.stem + "_timeline.png"))
    fig.savefig(out, dpi=120)
    print(f"Saved timeline -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
