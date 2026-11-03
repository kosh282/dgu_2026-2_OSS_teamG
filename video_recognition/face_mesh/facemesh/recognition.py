"""Stable face identity across frames.

MediaPipe returns faces in no guaranteed order, so per-face metrics (blink
counts, drowsiness state) would jump between people as the ordering changes.
:class:`FaceTracker` assigns a persistent integer ID to each face by matching
detections to existing tracks on their landmark centroid — greedy nearest
assignment with a distance gate and track aging.

This is appearance-free tracking (position-based), which is robust for typical
webcam scenes without pulling in a heavy face-recognition model. It tolerates
brief dropouts by keeping tracks alive for a few frames.
"""

from __future__ import annotations

import numpy as np


def _centroid(face_landmarks) -> np.ndarray:
    xs = np.fromiter((lm.x for lm in face_landmarks), dtype=float)
    ys = np.fromiter((lm.y for lm in face_landmarks), dtype=float)
    return np.array([xs.mean(), ys.mean()])


class FaceTracker:
    """Assign persistent IDs to faces across frames.

    Args:
        max_distance: Max centroid distance (in normalized units) to consider
            a detection a continuation of an existing track.
        max_age: Drop a track after this many consecutive unseen frames.
    """

    def __init__(self, max_distance: float = 0.12, max_age: int = 15) -> None:
        self.max_distance = max_distance
        self.max_age = max_age
        self._next_id = 0
        self._tracks: dict[int, dict] = {}  # id -> {"centroid", "missed"}

    def update(self, faces) -> list[int]:
        """Return a list of track IDs aligned with ``faces``."""
        centroids = [_centroid(f) for f in faces]
        assigned: dict[int, int] = {}  # face_index -> track_id
        used_tracks: set[int] = set()

        # Candidate (distance, face_idx, track_id) pairs within the gate.
        candidates = []
        for fi, c in enumerate(centroids):
            for tid, tr in self._tracks.items():
                d = float(np.linalg.norm(c - tr["centroid"]))
                if d <= self.max_distance:
                    candidates.append((d, fi, tid))
        candidates.sort(key=lambda t: t[0])

        # Greedy nearest matching.
        for _, fi, tid in candidates:
            if fi in assigned or tid in used_tracks:
                continue
            assigned[fi] = tid
            used_tracks.add(tid)
            self._tracks[tid]["centroid"] = centroids[fi]
            self._tracks[tid]["missed"] = 0

        # Unmatched detections start new tracks.
        for fi, c in enumerate(centroids):
            if fi not in assigned:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"centroid": c, "missed": 0}
                assigned[fi] = tid
                used_tracks.add(tid)

        # Age and prune unmatched tracks.
        for tid in list(self._tracks):
            if tid not in used_tracks:
                self._tracks[tid]["missed"] += 1
                if self._tracks[tid]["missed"] > self.max_age:
                    del self._tracks[tid]

        return [assigned[i] for i in range(len(faces))]
