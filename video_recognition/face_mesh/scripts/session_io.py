"""Schema-tolerant reader for face-mesh session logs.

The medical/main pipelines export per-frame records as JSONL or CSV. This module
reads either without assuming exact key names: it loads every record as a dict,
then lets callers pick fields by name or auto-detect numeric ones. Nothing here
imports the face-mesh package, so these tools work on any exported log.

Record shape assumed (all optional, auto-detected when present):
    frame        : int    frame index
    t / time / ts: float  seconds (falls back to frame index if absent)
    face_id / id : int    stable track id (default 0 when single-face)
    <metric>     : float  any numeric field (hr, resp, symmetry, tremor, ...)
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

_TIME_KEYS = ("t", "time", "ts", "timestamp", "sec", "seconds")
_FRAME_KEYS = ("frame", "frame_idx", "frame_index", "i", "n")
_FACE_KEYS = ("face_id", "id", "track_id", "face")


def _coerce(v: Any) -> Any:
    """Turn CSV strings into numbers where possible; leave the rest alone."""
    if isinstance(v, (int, float)) or v is None:
        return v
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    try:
        f = float(s)
        return int(f) if f.is_integer() and "." not in s and "e" not in s.lower() else f
    except ValueError:
        return v


def load_records(path: str | Path) -> list[dict]:
    """Load a session log (.jsonl / .json lines / .csv / .tsv) as a list of dicts."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"session log not found: {p}")
    suffix = p.suffix.lower()

    records: list[dict] = []
    if suffix in (".csv", ".tsv"):
        delim = "\t" if suffix == ".tsv" else ","
        with p.open(newline="") as fh:
            for row in csv.DictReader(fh, delimiter=delim):
                records.append({k: _coerce(v) for k, v in row.items()})
    else:  # treat everything else as JSON-lines
        with p.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
                elif isinstance(obj, list):  # a whole-session JSON array on one line
                    records.extend(x for x in obj if isinstance(x, dict))
    if not records:
        raise ValueError(f"no records parsed from {p}")
    return records


def _first_key(rec: dict, candidates: tuple[str, ...]) -> str | None:
    for k in candidates:
        if k in rec:
            return k
    return None


def time_axis(records: list[dict]) -> tuple[list[float], str]:
    """Return (values, label) for the best available time/frame axis."""
    tk = _first_key(records[0], _TIME_KEYS)
    if tk is not None:
        xs = [_num(r.get(tk)) for r in records]
        if any(x is not None for x in xs):
            return [x if x is not None else i for i, x in enumerate(xs)], "time (s)"
    fk = _first_key(records[0], _FRAME_KEYS)
    if fk is not None:
        xs = [_num(r.get(fk)) for r in records]
        if any(x is not None for x in xs):
            return [x if x is not None else i for i, x in enumerate(xs)], "frame"
    return list(range(len(records))), "frame"


def face_ids(records: list[dict]) -> list[int]:
    fk = _first_key(records[0], _FACE_KEYS)
    if fk is None:
        return [0]
    seen: list[int] = []
    for r in records:
        v = _num(r.get(fk))
        iv = int(v) if v is not None else 0
        if iv not in seen:
            seen.append(iv)
    return sorted(seen)


def filter_face(records: list[dict], face_id: int) -> list[dict]:
    fk = _first_key(records[0], _FACE_KEYS)
    if fk is None:
        return records
    return [r for r in records if int(_num(r.get(fk)) or 0) == face_id]


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and math.isnan(v)) else float(v)
    return None


def numeric_fields(records: list[dict], exclude: tuple[str, ...] = ()) -> list[str]:
    """Auto-detect metric columns: numeric in >=50% of records, not axis/id keys."""
    skip = set(_TIME_KEYS) | set(_FRAME_KEYS) | set(_FACE_KEYS) | set(exclude)
    keys: list[str] = []
    for r in records:
        for k in r:
            if k not in keys and k not in skip:
                keys.append(k)
    out = []
    for k in keys:
        vals = [_num(r.get(k)) for r in records]
        present = sum(v is not None for v in vals)
        if present >= max(1, len(records) // 2):
            out.append(k)
    return out


def series(records: list[dict], field: str) -> list[float | None]:
    return [_num(r.get(field)) for r in records]
