"""Validate rPPG heart rate against a ground-truth reference.

Synthetic tests only prove the signal processing is correct; they say nothing
about real-world accuracy. This harness measures your *actual* error so the
project can report an honest number instead of an assumed one.

You need a reference heart rate: a pulse oximeter, a smartwatch/fitness band,
or a manual pulse count (count beats for 30 s, double it).

    # 1. Record a session while sitting still, reading your reference
    python scripts/validate_rppg.py record --seconds 60 --out session.jsonl

    # 2. Enter the reference values you observed, then see the error
    python scripts/validate_rppg.py report session.jsonl --reference 68 71 70

Report prints MAE, RMSE and Bland-Altman style bias/limits-of-agreement, plus
the fraction of samples the estimator itself flagged unreliable.

SCREENING / EDUCATIONAL USE ONLY — not a medical device.
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
from facemesh.medical import HeartRateEstimator, QualityGate  # noqa: E402


def cmd_record(args) -> int:
    model = ensure_model()
    cap = cv2.VideoCapture(args.source if not args.source.isdigit() else int(args.source))
    if not cap.isOpened():
        print("error: cannot open camera", file=sys.stderr)
        return 2

    est = HeartRateEstimator(window_s=args.window)
    gate = QualityGate()
    out = Path(args.out).open("w", encoding="utf-8")
    start = time.monotonic()
    n_total = n_used = n_skipped = 0
    hr: dict = {"bpm": 0.0, "confidence": 0.0, "ready": False, "reliable": False}

    print(f"Recording {args.seconds}s. Sit still, face even lighting.")
    print("Frames failing the quality gate are EXCLUDED from the estimate.")
    print("Read your reference device every ~15s and note the values.\n")

    with FaceMeshDetector(model, running_mode="video") as det:
        while (now := time.monotonic() - start) < args.seconds:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            res = det.detect(frame, timestamp_ms=int(now * 1000))
            if not res.face_landmarks:
                continue
            face = res.face_landmarks[0]
            h, w = frame.shape[:2]
            n_total += 1

            q = gate.check(face, w, h, now, frame_bgr=frame)
            # Fix (#2): a frame the gate rejected must not feed the estimator.
            # Previously every frame was passed to update(), so the measured
            # accuracy described a polluted pipeline rather than the algorithm.
            if q["ok"]:
                hr = est.update(frame, face, now)
                n_used += 1
            else:
                n_skipped += 1
            out.write(json.dumps({
                "t": round(now, 2), "hr": hr, "quality": q, "used": bool(q["ok"]),
            }) + "\n")

            if not q["ok"]:
                txt, colour = ", ".join(q["reasons"]), (0, 165, 255)
            elif hr["ready"]:
                txt = f"{hr['bpm']:.0f} bpm  conf {hr['confidence']:.2f}"
                colour = (60, 255, 120) if hr.get("reliable") else (0, 165, 255)
            else:
                txt, colour = "warming up", (200, 200, 200)
            cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
            cv2.putText(frame, f"{args.seconds - now:.0f}s left   used {n_used}  "
                               f"skipped {n_skipped}",
                        (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow("rPPG validation - press q to stop", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    out.close()

    usable = n_used / n_total if n_total else 0.0
    print(f"\nWrote {n_total} samples -> {args.out}")
    print(f"  used {n_used} ({usable*100:.0f}%)   skipped {n_skipped}")
    if args.strict and usable < args.min_usable:
        print(f"\nERROR: only {usable*100:.0f}% of frames were usable "
              f"(--min-usable {args.min_usable*100:.0f}%).\n"
              "Any accuracy figure from this session would be meaningless.\n"
              "Improve the lighting (face a lamp or window head-on) and re-record.",
              file=sys.stderr)
        return 1
    if usable < args.min_usable:
        print(f"\nWARNING: only {usable*100:.0f}% of frames were usable. The accuracy "
              f"figure will describe your capture conditions, not the algorithm.\n"
              "Consider re-recording with better lighting, or pass --strict to enforce this.")
    print(f"Now run:  python {Path(__file__).name} report {args.out} --reference <bpm> <bpm> ...")
    return 0


def cmd_report(args) -> int:
    path = Path(args.input)
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 2

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # Sessions recorded before the #2 fix have no "used" key; treat the quality
    # verdict as authoritative so old files are re-scored correctly.
    used = [r for r in rows if r.get("used", r["quality"]["ok"])]
    excluded = len(rows) - len(used)
    ready = [r for r in used if r["hr"]["ready"]]
    reliable = [r for r in ready if r["hr"].get("reliable")]
    if not ready:
        print("No ready samples — the session was too short or no face was found.")
        return 1

    est = np.array([r["hr"]["bpm"] for r in reliable]) if reliable else np.array([])
    print(f"samples: {len(rows)}   excluded by quality gate: {excluded}   "
          f"used: {len(used)}")
    print(f"ready: {len(ready)}   reliable: {len(reliable)} "
          f"({100*len(reliable)/len(ready):.0f}% of ready)")
    if len(est):
        print(f"estimated bpm: mean {est.mean():.1f}  sd {est.std():.1f}  "
              f"range {est.min():.0f}-{est.max():.0f}")

    q = np.array([r["quality"]["score"] for r in rows])
    print(f"mean quality score: {q.mean():.2f}")
    bad = [reason for r in rows for reason in r["quality"]["reasons"]]
    if bad:
        from collections import Counter
        print("quality issues:", dict(Counter(bad)))

    if not args.reference:
        print("\nNo --reference given, so no accuracy computed.")
        print("Re-run with e.g.  --reference 68 71 70   (your device's readings)")
        return 0

    ref = np.array(args.reference, dtype=float)
    if not len(est):
        print("\nNo reliable estimates to compare against the reference.")
        return 1

    ref_mean = ref.mean()
    errors = est - ref_mean
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))
    sd = float(np.std(errors))

    print(f"\nreference: mean {ref_mean:.1f} bpm (from {len(ref)} readings)")
    print(f"MAE       {mae:5.1f} bpm")
    print(f"RMSE      {rmse:5.1f} bpm")
    print(f"bias      {bias:+5.1f} bpm  (positive = rPPG reads high)")
    print(f"95% limits of agreement: {bias - 1.96*sd:+.1f} .. {bias + 1.96*sd:+.1f} bpm")
    print("\nPut this number in the README. An honest measured error beats an "
          "unstated one.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="capture a session with rPPG estimates")
    rec.add_argument("--source", default="0")
    rec.add_argument("--seconds", type=float, default=60.0)
    rec.add_argument("--window", type=float, default=10.0)
    rec.add_argument("--out", default="rppg_session.jsonl")
    rec.add_argument("--strict", action="store_true",
                     help="abort if too few frames pass the quality gate")
    rec.add_argument("--min-usable", type=float, default=0.7,
                     help="usable-frame fraction below which the session is unreliable")
    rec.set_defaults(func=cmd_record)

    rep = sub.add_parser("report", help="compare a session against reference bpm")
    rep.add_argument("input")
    rep.add_argument("--reference", type=float, nargs="*", default=[],
                     help="ground-truth bpm readings from your device")
    rep.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
