"""Export per-frame detection results to JSON Lines (.jsonl) or CSV (.csv).

JSONL keeps the nested structure (one object per frame, variable face count);
CSV flattens to one row per (frame, face) for easy loading into pandas/Excel.
Pick the format by file extension via :func:`create_exporter`.

JSONL example::

    {"frame":0,"faces":[{"ear":0.29,"blink":{"left":0,"right":0,"total":0,
     "rate_per_min":0.0},"head_pose":{"pitch":3.1,"yaw":-12.4,"roll":1.0}}]}
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def landmarks_to_list(face_landmarks, ndigits: int = 5) -> list[list[float]]:
    """Convert a face's landmarks to a compact ``[[x, y, z], ...]`` list."""
    return [
        [round(lm.x, ndigits), round(lm.y, ndigits), round(lm.z, ndigits)]
        for lm in face_landmarks
    ]


def flatten_face(face: dict) -> dict:
    """Flatten one nested face record into a single-level dict for CSV.

    Nested dicts become ``parent_child`` keys; a ``landmarks`` list becomes
    ``lm{i}_x/_y/_z`` columns.
    """
    flat: dict = {}
    for key, value in face.items():
        if key == "landmarks":
            for i, (x, y, z) in enumerate(value):
                flat[f"lm{i}_x"] = x
                flat[f"lm{i}_y"] = y
                flat[f"lm{i}_z"] = z
        elif isinstance(value, dict):
            for sub, subval in value.items():
                flat[f"{key}_{sub}"] = subval
        else:
            flat[key] = value
    return flat


class JsonlExporter:
    """Append-friendly JSONL writer: one JSON object per frame."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = self.path.open("w", encoding="utf-8")

    def write_frame(self, frame_idx: int, faces: list[dict]) -> None:
        record = {"frame": frame_idx, "faces": faces}
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "JsonlExporter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class CsvExporter:
    """CSV writer: one row per (frame, face). Header inferred on first face."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = self.path.open("w", encoding="utf-8", newline="")
        self._writer: csv.DictWriter | None = None

    def _ensure_header(self, sample_flat: dict) -> None:
        fields = ["frame", "face"] + list(sample_flat.keys())
        self._writer = csv.DictWriter(self._fh, fieldnames=fields, extrasaction="ignore")
        self._writer.writeheader()

    def write_frame(self, frame_idx: int, faces: list[dict]) -> None:
        if not faces:
            if self._writer is not None:
                self._writer.writerow({"frame": frame_idx, "face": ""})
            return
        for face_idx, face in enumerate(faces):
            flat = flatten_face(face)
            if self._writer is None:
                self._ensure_header(flat)
            row = {"frame": frame_idx, "face": face_idx, **flat}
            self._writer.writerow(row)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "CsvExporter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def create_exporter(path: str | Path):
    """Return a JSONL or CSV exporter based on the file extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return CsvExporter(path)
    return JsonlExporter(path)  # default: .jsonl / anything else


# Backwards-compatible alias (v0.2 name).
MetricsExporter = JsonlExporter
MetricsExporter.landmarks_to_list = staticmethod(landmarks_to_list)
