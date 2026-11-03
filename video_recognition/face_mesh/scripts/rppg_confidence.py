#!/usr/bin/env python3
"""Attach an honest confidence band to each rPPG heart-rate estimate.

After the five rPPG bug fixes, the pipeline knows *when* a heart-rate estimate
is trustworthy (quality gate, lighting check, harmonic guard). This turns that
into a number the report can show next to the HR: a band and a label, instead of
a bare BPM that looks authoritative even when the signal was garbage.

The band widens as signal quality drops and as the spectral peak's prominence
over its neighbours shrinks (a flat spectrum = an untrustworthy peak). It is a
*reporting* aid, not a calibrated clinical interval — that requires the
reference-device validation in compare_reference_hr.py.

Input log fields (auto-detected, first match wins):
    hr / bpm / heart_rate          : float  estimate to annotate
    quality / snr / signal_quality : float  0..1 quality, or dB SNR
    peak_prominence / prominence   : float  optional; spectral peak prominence

Usage
-----
    python scripts/rppg_confidence.py --input session.jsonl --out annotated.jsonl
    python scripts/rppg_confidence.py --input session.jsonl --summary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from session_io import load_records

_HR_KEYS = ("hr", "bpm", "heart_rate", "rppg_hr", "hr_bpm")
_Q_KEYS = ("quality", "signal_quality", "snr", "snr_db", "quality_index")
_PROM_KEYS = ("peak_prominence", "prominence", "peak_snr", "harmonic_ratio")


def _pick(records: list[dict], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if any(k in r for r in records):
            return k
    return None


def _norm_quality(q: float | None) -> float:
    """Map quality to 0..1. Values >1 are treated as dB SNR (~0 dB..20 dB)."""
    if q is None:
        return 0.0
    if q <= 1.0:
        return max(0.0, min(1.0, q))
    return max(0.0, min(1.0, q / 20.0))


def confidence_band(
    hr: float | None, quality: float | None, prominence: float | None
) -> dict:
    """Return {hr, lo, hi, half_width, confidence, label} for one estimate."""
    if hr is None:
        return {"hr": None, "lo": None, "hi": None, "half_width": None,
                "confidence": 0.0, "label": "no-estimate"}

    q = _norm_quality(quality)
    # Prominence in 0..1; if given as a ratio >1, squash it.
    if prominence is None:
        p = q  # fall back to quality when prominence isn't logged
    else:
        p = prominence if prominence <= 1.0 else 1.0 - 1.0 / (1.0 + prominence)
        p = max(0.0, min(1.0, p))

    # Confidence is dominated by the weaker of the two signals.
    conf = round(min(q, p) * 0.6 + (q * p) * 0.4, 3)

    # Band: +/-2 bpm at perfect confidence, up to +/-18 bpm at zero.
    half = round(2.0 + (1.0 - conf) * 16.0, 1)
    if conf >= 0.75:
        label = "high"
    elif conf >= 0.45:
        label = "moderate"
    elif conf >= 0.2:
        label = "low"
    else:
        label = "reject"  # do not display this HR as a measurement
    return {
        "hr": round(hr, 1),
        "lo": round(hr - half, 1),
        "hi": round(hr + half, 1),
        "half_width": half,
        "confidence": conf,
        "label": label,
    }


def annotate(records: list[dict]) -> tuple[list[dict], str | None]:
    hr_k = _pick(records, _HR_KEYS)
    if hr_k is None:
        return records, None
    q_k = _pick(records, _Q_KEYS)
    p_k = _pick(records, _PROM_KEYS)
    out = []
    for r in records:
        band = confidence_band(
            r.get(hr_k), r.get(q_k) if q_k else None, r.get(p_k) if p_k else None
        )
        out.append({**r, "hr_confidence": band})
    return out, hr_k


def summarize(annotated: list[dict]) -> dict:
    bands = [r["hr_confidence"] for r in annotated if "hr_confidence" in r]
    usable = [b for b in bands if b["label"] not in ("reject", "no-estimate")]
    counts: dict[str, int] = {}
    for b in bands:
        counts[b["label"]] = counts.get(b["label"], 0) + 1
    hrs = [b["hr"] for b in usable if b["hr"] is not None]
    return {
        "frames": len(bands),
        "usable": len(usable),
        "usable_pct": round(100 * len(usable) / len(bands), 1) if bands else 0.0,
        "labels": counts,
        "hr_mean_usable": round(sum(hrs) / len(hrs), 1) if hrs else None,
        "mean_confidence": round(
            sum(b["confidence"] for b in bands) / len(bands), 3
        ) if bands else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None, help="write annotated JSONL here")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args(argv)

    records = load_records(args.input)
    annotated, hr_k = annotate(records)
    if hr_k is None:
        print("[rppg] no heart-rate field found; nothing to annotate")
        return 2

    if args.out:
        with Path(args.out).open("w") as fh:
            for r in annotated:
                fh.write(json.dumps(r) + "\n")
        print(f"[rppg] wrote annotated log -> {args.out}")

    if args.summary or not args.out:
        s = summarize(annotated)
        print(f"[rppg] {s['usable']}/{s['frames']} frames usable "
              f"({s['usable_pct']}%), mean confidence {s['mean_confidence']}")
        print(f"[rppg] labels: {s['labels']}")
        if s["hr_mean_usable"] is not None:
            print(f"[rppg] mean HR over usable frames: {s['hr_mean_usable']} bpm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
