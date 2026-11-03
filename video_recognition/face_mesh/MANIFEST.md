# face-mesh v1.4.0 delta — "what can be added next"

Six of the seven roadmap items, built as **standalone tools that read your
exported session logs** (JSONL/CSV). Nothing here imports the `face-mesh`
package or touches `medical/`, so it can't destabilise `main` and doesn't depend
on your internal APIs. Drop these files into the repo root over the existing
tree.

## Files in this delta

```text
.github/workflows/ci.yml            # #2  run ruff + the full suite on every push/PR
scripts/session_io.py               #     shared schema-tolerant log reader (used by the below)
scripts/session_trends.py           # #7  temporal trend plots per metric, multi-face aware
scripts/rppg_confidence.py          # #4  honest confidence band per HR estimate
scripts/compare_reference_hr.py     # #5  A/B vs a reference device (MAE/RMSE/Bland-Altman)
scripts/symmetry_composite.py       # #3  experimental composite symmetry score (gated)
docs/rppg-bug-findings.md           # #6  writeup of the five rPPG bugs
tests/conftest.py                   #     puts scripts/ on the path for pytest
tests/test_session_analysis.py      #     8 tests
tests/test_rppg_and_symmetry.py     #     12 tests
```

20 new tests, all green. `ruff==0.15.20` clean (matches the pin in ci.yml).

## Install

```bash
# from the repo root, with the delta unzipped alongside:
rsync -a face-mesh-v1.4.0-delta/ .        # overlay onto the repo
ruff check .                              # should stay clean
pytest -q                                 # your 278 + these 20
```

## How to use each tool

```bash
# #7 trends — auto-detects metrics; --face picks one when several are tracked
python scripts/session_trends.py --input session.jsonl --out trends.png --window 30

# #4 rPPG confidence — labels each estimate high/moderate/low/reject
python scripts/rppg_confidence.py --input session.jsonl --out annotated.jsonl --summary

# #5 reference comparison — the one number worth defending
python scripts/compare_reference_hr.py --session session.jsonl \
    --reference chest_strap.csv --tolerance 2.0 --plot bland_altman.png

# #3 composite symmetry — EXPERIMENTAL, prints a "not a clinical grade" banner
python scripts/symmetry_composite.py --input session.jsonl
```

All four read whatever your export already writes. They auto-detect field names
(`hr`/`bpm`/`heart_rate`, `quality`/`snr`, `*_symmetry`, `synkinesis`, a time or
frame axis, a `face_id`). If your export uses different names, the `--fields`
flags and the `_*_KEYS` tuples at the top of each script are the one place to
adjust — no other change needed.

## The one item NOT in this delta

**#1 Multi-face medical protocol.** This one genuinely can't be built blind: it
means changing the per-face loop inside your `medical/` protocol runner so it
scores every tracked face instead of face 0 only. I need to see the actual
protocol/detector code to wire it without guessing your method signatures —
otherwise I'd be inventing an API, which is the mistake we already fixed once on
the warehouse paper.

Send me the `medical/` protocol module (or a zip of that subtree) and I'll add
it as v1.4.1. The analysis tools above are already multi-face aware, so the data
side is ready for it.

## Two notes worth acting on

- **`compare_reference_hr.py` is the highest-value tool here.** One honest MAE
  against a chest strap or pulse oximeter is worth more to an admissions
  committee than any new metric. Record one session facing a lamp with a
  reference device on, and you'll have a number you can defend.
- **The composite symmetry score prints a "NOT a clinical grade" banner by
  design.** Leave that in until the dataset-validation step exists — an
  uncalibrated composite that *looks* like a Sunnybrook grade is exactly the kind
  of confident-wrong output the bug writeup is about.
