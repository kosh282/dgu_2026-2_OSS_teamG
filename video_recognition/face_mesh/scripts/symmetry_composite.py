#!/usr/bin/env python3
"""Combine per-region symmetry sub-metrics into ONE experimental composite score.

A Sunnybrook-style composite reduces several facial-symmetry sub-metrics to a
single 0-100 number. This computes such a score from whatever symmetry
components are present in the log.

    >>> READ THIS <<<
    The composite is UNCALIBRATED and NOT clinically validated. The weights are
    placeholders, not derived from clinician ratings against patients. Do not
    present this number as a Sunnybrook / House-Brackmann grade. It becomes
    defensible only after the dataset-validation step (scripts/validate_dataset.py
    on a licensed dataset). Until then it is a relative, within-session indicator.

Sub-metrics are auto-detected by name (any of):
    resting_symmetry, voluntary_symmetry, synkinesis,
    brow_symmetry, eye_symmetry, mouth_symmetry, nasolabial_symmetry, ...
Each is expected in 0..1 (1 = perfectly symmetric). Values >1 are min-maxed.

Usage
-----
    python scripts/symmetry_composite.py --input session.jsonl
    python scripts/symmetry_composite.py --input log.csv --weights mouth=2,eye=1.5
"""
from __future__ import annotations

import argparse

from session_io import load_records, numeric_fields, series

# Regions that count toward symmetry; synkinesis is a penalty (higher = worse).
_SYMMETRY_HINTS = ("symmetry", "sym")
_PENALTY_HINTS = ("synkinesis", "synk", "asymmetry")


def _detect_components(records: list[dict]) -> tuple[list[str], list[str]]:
    fields = numeric_fields(records)
    pos = [f for f in fields if any(h in f.lower() for h in _SYMMETRY_HINTS)
           and not any(h in f.lower() for h in _PENALTY_HINTS)]
    pen = [f for f in fields if any(h in f.lower() for h in _PENALTY_HINTS)]
    return pos, pen


def _mean(vals: list[float | None]) -> float | None:
    xs = [v for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else None


def _minmax(v: float, lo: float, hi: float) -> float:
    return 0.5 if hi <= lo else max(0.0, min(1.0, (v - lo) / (hi - lo)))


def composite_score(
    records: list[dict], weights: dict[str, float] | None = None
) -> dict:
    weights = weights or {}
    pos, pen = _detect_components(records)
    if not pos and not pen:
        return {"score": None, "components": {}, "note": "no symmetry fields found"}

    comp: dict[str, float] = {}
    for f in pos + pen:
        m = _mean(series(records, f))
        if m is None:
            continue
        norm = m if m <= 1.0 else _minmax(m, 0.0, m)  # squash odd scales
        comp[f] = round(norm, 3)

    num, den = 0.0, 0.0
    for f, val in comp.items():
        w = weights.get(f, next((weights[k] for k in weights if k in f.lower()), 1.0))
        contribution = (1.0 - val) if any(h in f.lower() for h in _PENALTY_HINTS) else val
        num += w * contribution
        den += w
    score = round(100 * num / den, 1) if den else None
    return {"score": score, "components": comp,
            "note": "EXPERIMENTAL / uncalibrated — not a clinical grade"}


def _parse_weights(s: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        out[k.strip().lower()] = float(v)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--weights", default="", help="e.g. mouth=2,eye=1.5")
    args = ap.parse_args(argv)

    records = load_records(args.input)
    result = composite_score(records, _parse_weights(args.weights) if args.weights else None)

    print("=" * 60)
    print("  EXPERIMENTAL composite symmetry — NOT a clinical grade")
    print("=" * 60)
    if result["score"] is None:
        print(f"[symmetry] {result['note']}")
        return 2
    print(f"[symmetry] composite: {result['score']}/100")
    for f, v in result["components"].items():
        print(f"    - {f}: {v}")
    print(f"[symmetry] {result['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
