"""Validate the medical metrics against a labelled dataset.

Synthetic tests prove the *math*. Only a labelled dataset tells you whether the
metrics separate the populations they claim to. This harness runs the pipeline
over a folder of images or videos with ground-truth labels and reports real,
citable numbers: AUC, sensitivity/specificity at the best threshold, and a
confusion matrix.

It is dataset-agnostic. Supply a manifest CSV:

    path,label,subject
    faces/001.jpg,palsy,s01
    faces/002.jpg,healthy,s02
    videos/003.mp4,parkinsons,s03

Then:

    # Symmetry as a palsy screen (higher asymmetry = positive)
    python scripts/validate_dataset.py manifest.csv --metric symmetry \\
        --positive palsy --negative healthy

    # Expressivity as a hypomimia screen (LOWER expressivity = positive)
    python scripts/validate_dataset.py manifest.csv --metric expressivity \\
        --positive parkinsons --negative healthy --lower-is-positive

Outputs a JSON report and prints a summary. Frames failing the quality gate are
excluded and counted, so a poor result cannot be blamed on bad capture silently.

.. warning::
   A good AUC here does **not** make this a diagnostic device. It tells you the
   metric carries signal on *that* dataset, with that dataset's biases. Public
   facial-palsy and Parkinson's datasets are small, often posed, and demographic
   coverage is narrow. Report the dataset, its size, and its limits alongside
   any number you quote.

No dataset is bundled: these are clinical images and you must obtain them
yourself under their own licence and ethics terms.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from facemesh import FaceMeshDetector, ensure_model  # noqa: E402
from facemesh.medical import (  # noqa: E402
    MovementAnalyzer,
    QualityGate,
    symmetry_metrics,
)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# --- metrics -------------------------------------------------------------
def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via the rank-sum identity (equivalent to the Mann-Whitney U)."""
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Threshold maximizing Youden's J = sensitivity + specificity - 1."""
    best = {"threshold": float("nan"), "youden": -1.0}
    for thr in np.unique(scores):
        pred = scores >= thr
        tp = int(((pred == 1) & (labels == 1)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        j = sens + spec - 1
        if j > best["youden"]:
            best = {"threshold": float(thr), "youden": round(j, 4),
                    "sensitivity": round(sens, 4), "specificity": round(spec, 4),
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    return best


# --- extraction ----------------------------------------------------------
def _score_landmarks(face, w, h, metric, mover=None, now_s=0.0):
    if metric == "symmetry":
        return symmetry_metrics(face, w, h)["asymmetry"]
    if metric == "expressivity":
        return mover.update(face, w, h, now_s)["expressivity"]
    raise ValueError(f"unknown metric {metric!r}")


def score_image(det, gate, path: Path, metric: str) -> tuple[float | None, str]:
    img = cv2.imread(str(path))
    if img is None:
        return None, "unreadable"
    res = det.detect(img)
    if not res.face_landmarks:
        return None, "no face"
    face = res.face_landmarks[0]
    h, w = img.shape[:2]
    q = gate.check(face, w, h, 0.0, frame_bgr=img)
    if not q["ok"]:
        return None, ",".join(q["reasons"])
    if metric == "expressivity":
        return None, "expressivity needs video"
    return _score_landmarks(face, w, h, metric), "ok"


def score_video(det, gate, path: Path, metric: str, max_frames: int = 600) -> tuple[float | None, str]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, "unreadable"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mover = MovementAnalyzer(window_s=5.0)
    mover.set_task("dataset")
    vals, skipped, i = [], 0, 0
    while i < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        res = det.detect(frame, timestamp_ms=int(i / fps * 1000))
        if res.face_landmarks:
            face = res.face_landmarks[0]
            q = gate.check(face, w, h, i / fps, frame_bgr=frame)
            if q["ok"]:
                vals.append(_score_landmarks(face, w, h, metric, mover, i / fps))
            else:
                skipped += 1
        i += 1
    cap.release()
    if not vals:
        return None, f"no usable frames ({skipped} skipped)"
    # symmetry: the peak over the clip; expressivity: the peak reached
    return float(np.max(vals)), "ok"


def run(manifest: Path, metric: str, positive: str, negative: str,
        lower_is_positive: bool, model_path) -> dict:
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    root = manifest.parent
    gate = QualityGate()

    scores, labels, subjects = [], [], []
    excluded: dict[str, int] = defaultdict(int)
    per_class: dict[str, list] = defaultdict(list)

    mode = "video" if metric == "expressivity" else "image"
    running = "video" if mode == "video" else "image"
    with FaceMeshDetector(model_path, running_mode=running) as det:
        for row in rows:
            label = row["label"].strip()
            if label not in (positive, negative):
                continue
            path = (root / row["path"]).resolve()
            if not path.exists():
                excluded["missing file"] += 1
                continue
            ext = path.suffix.lower()
            if ext in VIDEO_EXT:
                s, why = score_video(det, gate, path, metric)
            elif ext in IMAGE_EXT:
                s, why = score_image(det, gate, path, metric)
            else:
                s, why = None, "unsupported type"
            if s is None:
                excluded[why] += 1
                continue
            scores.append(s)
            labels.append(1 if label == positive else 0)
            subjects.append(row.get("subject", ""))
            per_class[label].append(s)

    if not scores:
        return {"error": "no usable samples", "excluded": dict(excluded)}

    sc = np.asarray(scores, dtype=float)
    lb = np.asarray(labels, dtype=int)
    # Convention: higher score = more positive. Flip when lower means positive.
    oriented = -sc if lower_is_positive else sc

    auc = roc_auc(oriented, lb)
    thr = best_threshold(oriented, lb)
    return {
        "metric": metric,
        "positive": positive,
        "negative": negative,
        "lower_is_positive": lower_is_positive,
        "n_positive": int((lb == 1).sum()),
        "n_negative": int((lb == 0).sum()),
        "n_subjects": len({s for s in subjects if s}),
        "auc": round(auc, 4),
        "operating_point": thr,
        "class_means": {k: round(float(np.mean(v)), 5) for k, v in per_class.items()},
        "excluded": dict(excluded),
        "excluded_total": sum(excluded.values()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate a metric against a labelled dataset.")
    ap.add_argument("manifest", help="CSV with columns: path,label[,subject]")
    ap.add_argument("--metric", choices=["symmetry", "expressivity"], default="symmetry")
    ap.add_argument("--positive", required=True, help="label counted as positive")
    ap.add_argument("--negative", required=True, help="label counted as negative")
    ap.add_argument("--lower-is-positive", action="store_true",
                    help="set when a LOWER metric value indicates the condition "
                         "(e.g. reduced expressivity in hypomimia)")
    ap.add_argument("-o", "--output", default="dataset_validation.json")
    args = ap.parse_args(argv)

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"error: {manifest} not found", file=sys.stderr)
        return 2

    result = run(manifest, args.metric, args.positive, args.negative,
                 args.lower_is_positive, ensure_model())
    Path(args.output).write_text(json.dumps(result, indent=2))

    if "error" in result:
        print(f"error: {result['error']}; excluded={result['excluded']}", file=sys.stderr)
        return 1

    print(f"\n{args.metric}: {args.positive} (n={result['n_positive']}) vs "
          f"{args.negative} (n={result['n_negative']})")
    print(f"AUC                {result['auc']:.3f}")
    op = result["operating_point"]
    print(f"best threshold     {op['threshold']:.4f}")
    print(f"sensitivity        {op['sensitivity']:.3f}")
    print(f"specificity        {op['specificity']:.3f}")
    print(f"confusion (tp,fp,tn,fn) = ({op['tp']},{op['fp']},{op['tn']},{op['fn']})")
    print(f"class means        {result['class_means']}")
    if result["excluded_total"]:
        print(f"excluded           {result['excluded_total']}: {result['excluded']}")
    print(f"\nSaved -> {args.output}")
    print("Quote the dataset name, size and demographics next to this number. "
          "AUC on one dataset is not clinical validity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
