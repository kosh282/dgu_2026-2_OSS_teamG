"""Appearance-based face re-identification.

The position-based :class:`~facemesh.recognition.FaceTracker` keeps IDs stable
while people stay put, but it fails in exactly the cases that matter: two people
swap seats, someone leaves and returns, or you want to resume a session
tomorrow and attach it to the same person.

This module derives a compact **geometric face embedding** from the landmark
mesh itself — no extra model, no image pixels, no network. The embedding is the
set of normalized pairwise distances between stable anatomical landmarks,
computed in the pose-invariant 3D face frame. It captures face *shape*
(proportions of the skull, eye spacing, nose length) rather than expression.

.. warning::
   This is **not** a face-recognition system in the security sense. Geometric
   mesh embeddings are far weaker than a trained deep embedding (FaceNet,
   ArcFace): they degrade with expression, they can confuse similar-looking
   faces, and they carry the accuracy biases of the underlying landmark model.
   Use it to keep metrics attached to the right person in a cooperative
   session, not to verify identity.

.. note::
   The embedding is computed from **pixel** geometry (``x * width``,
   ``y * height``), which is what recovers true facial proportions from
   MediaPipe's normalized coordinates. A consequence is that a gallery must be
   built and queried with the **same camera aspect ratio**; enrolling at 16:9
   and identifying at 4:3 distorts the geometry and lowers similarity.

Privacy: embeddings are derived from geometry, are not reversible to an image,
and are stored only if you explicitly save a gallery.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import normalize_3d

# Stable skeletal landmarks — chosen to avoid the mouth and brows, which move
# with expression. Skull outline, eye corners, nose bridge, jaw.
STABLE_IDX = (
    10, 152, 168, 1,                       # forehead, chin, sellion, nose tip
    33, 133, 263, 362,                     # eye corners
    234, 454,                              # cheekbones / face width
    127, 356,                              # temples
    58, 288,                               # jaw angles
    93, 323,                               # lower cheeks
    168, 6, 197, 195,                      # nose bridge chain
)


def face_embedding(landmarks, width: int, height: int) -> np.ndarray:
    """A pose-invariant geometric embedding of face *shape*.

    Pairwise distances between stable landmarks, in the normalized 3D face
    frame, L2-normalized. Invariant to translation, rotation, and scale.
    """
    N = normalize_3d(landmarks, width, height)
    pts = N[list(STABLE_IDX)]
    # Upper-triangular pairwise distances.
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(len(pts), k=1)
    vec = dists[iu]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two embeddings, in ``[-1, 1]`` (1 = identical)."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class FaceGallery:
    """Enrolled identities, matched by embedding similarity.

    Args:
        threshold: Minimum cosine similarity to accept a match. Higher is
            stricter. Geometric embeddings are tightly clustered so this
            defaults high; a deep backend wants a much lower value (~0.36 for
            ArcFace). Pass ``backend`` to inherit its default.
        backend: An :class:`~facemesh.medical.embeddings.EmbeddingBackend`.
            Only its ``default_threshold`` and ``name`` are used here; the
            embeddings themselves are supplied by the caller.
    """

    def __init__(self, threshold: float | None = None, backend=None) -> None:
        self.backend_name = getattr(backend, "name", "geometric")
        if threshold is None:
            threshold = getattr(backend, "default_threshold", 0.9995)
        self.threshold = threshold
        self._people: dict[str, list[np.ndarray]] = {}

    @property
    def names(self) -> list[str]:
        return list(self._people)

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        """Add one sample for ``name``. Several samples improve robustness."""
        self._people.setdefault(name, []).append(np.asarray(embedding, dtype=float))

    def prototype(self, name: str) -> np.ndarray:
        """Mean embedding for an enrolled person."""
        return np.mean(self._people[name], axis=0)

    def identify(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Return ``(name, similarity)``; name is None below the threshold."""
        if not self._people:
            return None, 0.0
        best_name, best_score = None, -1.0
        for name in self._people:
            s = similarity(embedding, self.prototype(name))
            if s > best_score:
                best_name, best_score = name, s
        if best_score < self.threshold:
            return None, best_score
        return best_name, best_score

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 2,
            "threshold": self.threshold,
            "backend": self.backend_name,
            "people": {n: [e.tolist() for e in embs] for n, embs in self._people.items()},
        }
        path.write_text(json.dumps(data))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FaceGallery":
        data = json.loads(Path(path).read_text())
        g = cls(threshold=data.get("threshold", 0.9995))
        g.backend_name = data.get("backend", "geometric")
        for name, embs in data.get("people", {}).items():
            for e in embs:
                g.enroll(name, np.asarray(e, dtype=float))
        return g


class AppearanceTracker:
    """Track faces by appearance, falling back on position when unsure.

    Unlike the position-only tracker, an identity survives leaving the frame,
    swapping places with someone else, or a new session entirely (if you enroll
    into a saved gallery).

    Args:
        gallery: A pre-enrolled :class:`FaceGallery`, or None to build one
            on the fly (auto-enrolling each new face as ``person_N``).
        threshold: Similarity needed to call it the same person.
    """

    def __init__(self, gallery: FaceGallery | None = None,
                 threshold: float | None = None, auto_enroll: bool = True,
                 backend=None) -> None:
        self.backend = backend
        self.gallery = gallery or FaceGallery(threshold=threshold, backend=backend)
        self.auto_enroll = auto_enroll
        self._next = 0

    def _embed(self, face, width, height, frame_bgr=None):
        if self.backend is not None and frame_bgr is not None:
            return self.backend.embed(frame_bgr, face, width, height)
        if self.backend is not None and getattr(self.backend, "name", "") == "geometric":
            return self.backend.embed(None, face, width, height)
        return face_embedding(face, width, height)

    def update(self, faces, width: int, height: int, frame_bgr=None) -> list[str]:
        """Return a stable identity label per face, aligned with ``faces``."""
        labels: list[str] = []
        taken: set[str] = set()
        for face in faces:
            emb = self._embed(face, width, height, frame_bgr)
            name, score = self.gallery.identify(emb)
            if name is not None and name not in taken:
                labels.append(name)
                taken.add(name)
                continue
            if self.auto_enroll:
                new = f"person_{self._next}"
                self._next += 1
                self.gallery.enroll(new, emb)
                labels.append(new)
                taken.add(new)
            else:
                labels.append("unknown")
        return labels
