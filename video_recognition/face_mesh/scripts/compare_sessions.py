"""Compare exported sessions over time.

    python scripts/compare_sessions.py week1.jsonl week2.jsonl [week3.jsonl ...]
    python scripts/compare_sessions.py *.jsonl -o longitudinal.json

Reports each session's metrics with confidence intervals, the change between
consecutive sessions, and a trend across three or more.

A change is called "separated" when the confidence intervals do not overlap.
That is weak evidence the change exceeds sampling noise - it is NOT a
hypothesis test and NOT a clinical finding. Lighting, distance, time of day and
mood all move these numbers; compare like with like.

SCREENING / EDUCATIONAL USE ONLY - not a diagnostic device.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from facemesh.medical import DISCLAIMER  # noqa: E402
from facemesh.medical.longitudinal import compare_many  # noqa: E402

_MARK = {"improved": "+", "worsened": "-", "changed": "~", "stable": "="}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare facial metrics across sessions.")
    ap.add_argument("sessions", nargs="+", help="exported .jsonl sessions, oldest first")
    ap.add_argument("-o", "--output", default=None, help="write full JSON here")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)

    paths = [Path(p) for p in args.sessions]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"error: not found: {', '.join(str(m) for m in missing)}", file=sys.stderr)
        return 2
    if len(paths) < 2:
        print("error: give at least two sessions to compare", file=sys.stderr)
        return 2

    result = compare_many(paths, n_boot=args.n_boot)

    print("Sessions")
    for s in result["sessions"]:
        print(f"  {s['session']:<24} {s['usable_frames']}/{s['frames']} usable "
              f"({s['usable_fraction']*100:.0f}%)")

    for comp in result["comparisons"]:
        print(f"\n{comp['from']} -> {comp['to']}")
        if not comp["metrics"]:
            print("  (no metrics in common)")
        for label, m in comp["metrics"].items():
            b, a = m["before"], m["after"]
            pct = f"{m['pct_change']:+.1f}%" if m["pct_change"] is not None else "  n/a"
            print(f"  {_MARK[m['verdict']]} {label:<20} "
                  f"{b['estimate']:.3f} [{b['ci_low']:.3f},{b['ci_high']:.3f}]  ->  "
                  f"{a['estimate']:.3f} [{a['ci_low']:.3f},{a['ci_high']:.3f}]  "
                  f"{pct:>8}  {m['verdict']}")

    if result["trends"]:
        print("\nTrends (>= 3 sessions)")
        for label, t in result["trends"].items():
            print(f"  {label:<20} slope {t['slope_per_session']:+.4f}/session "
                  f"(R2 {t['r_squared']:.2f})  {t['first']:.3f} -> {t['last']:.3f}")

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"\nSaved -> {args.output}")

    print("\n'stable' means the intervals overlap, not that nothing changed.")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
