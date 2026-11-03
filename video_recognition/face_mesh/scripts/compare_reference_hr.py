#!/usr/bin/env python3
"""Validate rPPG heart rate against a reference device (chest strap / oximeter).

This is the single most valuable thing the medical branch is missing: one honest
error number you can defend. It aligns the pipeline's per-frame HR with a
timestamped reference trace, then reports MAE, RMSE, bias and Bland-Altman
limits of agreement — the numbers a reviewer (or an admissions committee) will
actually trust.

Reference CSV format (two columns, header row):
    t,hr
    0.0,58
    1.0,59
    ...
`t` is seconds from session start (or a Unix epoch — pass --ref-epoch and
--session-epoch to align absolute clocks).

Usage
-----
    python scripts/compare_reference_hr.py --session session.jsonl --reference ref.csv
    python scripts/compare_reference_hr.py --session s.jsonl --reference r.csv \
        --tolerance 2.0 --plot bland_altman.png
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from session_io import face_ids, filter_face, load_records, time_axis

_HR_KEYS = ("hr", "bpm", "heart_rate", "rppg_hr", "hr_bpm")


def _pick(records: list[dict], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if any(k in r for r in records):
            return k
    return None


def load_reference(path: str) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    with Path(path).open(newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    start = 1 if rows and not _is_number(rows[0][0]) else 0
    for row in rows[start:]:
        if len(row) < 2:
            continue
        try:
            pts.append((float(row[0]), float(row[1])))
        except ValueError:
            continue
    if not pts:
        raise ValueError(f"no (t,hr) pairs parsed from reference {path}")
    return pts


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def nearest(ref: list[tuple[float, float]], t: float, tol: float) -> float | None:
    best, best_d = None, tol
    for rt, rhr in ref:
        d = abs(rt - t)
        if d <= best_d:
            best, best_d = rhr, d
    return best


def pair_up(
    session: list[dict],
    ref: list[tuple[float, float]],
    tolerance: float,
    session_offset: float,
) -> list[tuple[float, float]]:
    hr_k = _pick(session, _HR_KEYS)
    if hr_k is None:
        raise ValueError("no heart-rate field found in session log")
    xs, _ = time_axis(session)
    pairs: list[tuple[float, float]] = []
    for x, r in zip(xs, session):
        est = r.get(hr_k)
        if est is None:
            continue
        ref_hr = nearest(ref, x + session_offset, tolerance)
        if ref_hr is not None:
            pairs.append((float(est), float(ref_hr)))
    return pairs


def agreement_stats(pairs: list[tuple[float, float]]) -> dict:
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    diffs = [e - r for e, r in pairs]
    mae = sum(abs(d) for d in diffs) / n
    rmse = math.sqrt(sum(d * d for d in diffs) / n)
    bias = sum(diffs) / n
    sd = math.sqrt(sum((d - bias) ** 2 for d in diffs) / n) if n > 1 else 0.0
    # Pearson r
    es = [e for e, _ in pairs]
    rs = [r for _, r in pairs]
    me, mr = sum(es) / n, sum(rs) / n
    cov = sum((e - me) * (r - mr) for e, r in pairs)
    de = math.sqrt(sum((e - me) ** 2 for e in es))
    dr = math.sqrt(sum((r - mr) ** 2 for r in rs))
    pearson = cov / (de * dr) if de > 0 and dr > 0 else float("nan")
    return {
        "n": n,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "bias": round(bias, 2),
        "loa_low": round(bias - 1.96 * sd, 2),
        "loa_high": round(bias + 1.96 * sd, 2),
        "pearson_r": round(pearson, 3),
    }


def bland_altman_plot(pairs: list[tuple[float, float]], stats: dict, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = [(e + r) / 2 for e, r in pairs]
    diffs = [e - r for e, r in pairs]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(means, diffs, s=14, alpha=0.6, color="#e91e63", edgecolor="none")
    ax.axhline(stats["bias"], color="#263238", lw=1.4, label=f"bias {stats['bias']}")
    for y, name in ((stats["loa_high"], "+1.96 SD"), (stats["loa_low"], "-1.96 SD")):
        ax.axhline(y, color="#607d8b", ls="--", lw=1.0, label=f"{name} {y}")
    ax.set_xlabel("mean of rPPG & reference (bpm)")
    ax.set_ylabel("rPPG - reference (bpm)")
    ax.set_title(f"Bland-Altman  (n={stats['n']}, MAE {stats['mae']} bpm)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--face", type=int, default=None,
                    help="restrict session to one face id (default: first)")
    ap.add_argument("--tolerance", type=float, default=2.0,
                    help="max |t| gap (s) to match a reference sample")
    ap.add_argument("--session-epoch", type=float, default=0.0)
    ap.add_argument("--ref-epoch", type=float, default=0.0)
    ap.add_argument("--plot", default=None, help="write a Bland-Altman PNG here")
    args = ap.parse_args(argv)

    session = load_records(args.session)
    ids = face_ids(session)
    face = args.face if args.face is not None else ids[0]
    if len(ids) > 1:
        print(f"[compare] {len(ids)} faces present {ids}; comparing face {face} "
              f"(use --face to change)")
    session = filter_face(session, face)
    ref = load_reference(args.reference)
    offset = args.session_epoch - args.ref_epoch
    pairs = pair_up(session, ref, args.tolerance, offset)
    stats = agreement_stats(pairs)

    if stats["n"] == 0:
        print("[compare] no matched samples — check --tolerance and clock alignment")
        return 2

    print(f"[compare] matched {stats['n']} samples within +/-{args.tolerance}s")
    print(f"[compare] MAE {stats['mae']} bpm | RMSE {stats['rmse']} bpm | "
          f"bias {stats['bias']} bpm")
    print(f"[compare] 95% limits of agreement: "
          f"[{stats['loa_low']}, {stats['loa_high']}] bpm")
    print(f"[compare] Pearson r = {stats['pearson_r']}")
    if args.plot:
        bland_altman_plot(pairs, stats, args.plot)
        print(f"[compare] wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
