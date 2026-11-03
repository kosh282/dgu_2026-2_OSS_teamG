#!/usr/bin/env python3
"""Plot clinical/attention metrics over the course of a recorded session.

Reads an exported session log and draws each metric as a time series with a
rolling-mean overlay, one small multiple per metric. Works on any exported log
(auto-detects numeric fields) and is multi-face aware.

Examples
--------
    python scripts/session_trends.py --input session.jsonl --out trends.png
    python scripts/session_trends.py --input log.csv --fields hr,resp,symmetry
    python scripts/session_trends.py --input session.jsonl --face 1 --window 45
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from session_io import (  # noqa: E402
    face_ids,
    filter_face,
    load_records,
    numeric_fields,
    series,
    time_axis,
)


def rolling_mean(ys: list[float | None], window: int) -> list[float | None]:
    out: list[float | None] = []
    buf: list[float] = []
    for y in ys:
        if y is not None:
            buf.append(y)
            if len(buf) > window:
                buf.pop(0)
        out.append(sum(buf) / len(buf) if buf else None)
    return out


def _clean(xs, ys):
    return zip(*[(x, y) for x, y in zip(xs, ys) if y is not None]) if any(
        y is not None for y in ys
    ) else ([], [])


def plot_session(
    records: list[dict], fields: list[str], window: int, title: str, out: Path
) -> Path:
    xs, xlabel = time_axis(records)
    n = len(fields)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.4 * n), sharex=True, squeeze=False)
    for ax, field in zip(axes[:, 0], fields):
        ys = series(records, field)
        rx, ry = _clean(xs, ys)
        ax.plot(rx, ry, lw=0.8, alpha=0.45, color="#455a64", label="raw")
        mx, my = _clean(xs, rolling_mean(ys, window))
        ax.plot(mx, my, lw=1.8, color="#e91e63", label=f"rolling ({window})")
        ax.set_ylabel(field)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    axes[-1, 0].set_xlabel(xlabel)
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="session log (.jsonl/.csv)")
    ap.add_argument("--out", default="session_trends.png")
    ap.add_argument("--fields", default="", help="comma list; default = auto-detect")
    ap.add_argument("--face", type=int, default=None, help="restrict to one face id")
    ap.add_argument("--window", type=int, default=30, help="rolling-mean window (frames)")
    args = ap.parse_args(argv)

    records = load_records(args.input)
    ids = face_ids(records)
    face = args.face if args.face is not None else ids[0]
    if len(ids) > 1 and args.face is None:
        print(f"[trends] {len(ids)} faces present {ids}; plotting face {face} "
              f"(use --face to change)")
    records = filter_face(records, face)

    fields = [f.strip() for f in args.fields.split(",") if f.strip()] or numeric_fields(
        records
    )
    if not fields:
        print("[trends] no numeric metric fields found in log")
        return 2

    out = plot_session(
        records, fields, args.window, f"Session trends — face {face}", Path(args.out)
    )
    print(f"[trends] wrote {out}  ({len(fields)} metrics, {len(records)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
