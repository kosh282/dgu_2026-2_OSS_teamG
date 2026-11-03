"""Pluggable face-embedding backends.

The default embedding in :mod:`identity` is *geometric*: pairwise distances
between stable landmarks. It is dependency-free and pose-invariant, but weak —
it degrades with expression, confuses similar face shapes, and is unsuitable for
anything resembling identity verification.

A **deep embedding** (ArcFace, FaceNet) is dramatically more discriminative
because it is trained on millions of identities. This module provides a common
interface so a deep backend can be dropped in without touching the gallery or
tracker code.

No model weights are bundled. Deep backends require you to install the runtime
and obtain the weights yourself, under their own licence:

    pip install onnxruntime            # then supply an ArcFace .onnx file
    pip install insightface            # downloads weights on first use

Usage::

    from facemesh.medical.embeddings import get_backend
    backend = get_backend("auto")                # geometric unless a deep one is set up
    emb = backend.embed(frame_bgr, landmarks, w, h)

.. warning::
   Deep face embeddings are biometric identifiers. They are far more powerful
   than the geometric default, and correspondingly more sensitive: they can
   identify people across sessions and devices, they carry documented
   demographic accuracy disparities, and in many jurisdictions they are
   regulated personal data. Obtain consent, store them securely, and do not
   deploy one because it is convenient.

SCREENING / EDUCATIONAL USE ONLY — not a diagnostic device.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class EmbeddingBackend(ABC):
    """Common interface for face embeddings."""

    #: Human-readable name.
    name: str = "base"
    #: Cosine similarity above which two embeddings are the same person.
    default_threshold: float = 0.9

    @abstractmethod
    def embed(self, frame_bgr, landmarks, width: int, height: int) -> np.ndarray:
        """Return a 1-D embedding for the face."""

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether this backend can run right now."""

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


class GeometricBackend(EmbeddingBackend):
    """Landmark-geometry embedding. Always available; weak but honest."""

    name = "geometric"
    default_threshold = 0.9995

    @property
    def available(self) -> bool:
        return True

    def embed(self, frame_bgr, landmarks, width: int, height: int) -> np.ndarray:
        from .identity import face_embedding  # local import avoids a cycle
        return face_embedding(landmarks, width, height)


def _aligned_crop(frame_bgr, landmarks, width: int, height: int, size: int = 112):
    """Similarity-align the face to a fixed square using the eye line.

    Deep face models expect a canonical crop; feeding raw boxes degrades them
    badly.
    """
    import cv2  # noqa: PLC0415

    right_eye = np.array([landmarks[33].x * width, landmarks[33].y * height])
    left_eye = np.array([landmarks[263].x * width, landmarks[263].y * height])
    nose = np.array([landmarks[1].x * width, landmarks[1].y * height])

    eye_center = (right_eye + left_eye) / 2.0
    dx, dy = left_eye - right_eye
    angle = np.degrees(np.arctan2(dy, dx))
    iod = float(np.linalg.norm(left_eye - right_eye))
    if iod <= 1:
        return None
    scale = (size * 0.38) / iod

    M = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale)
    # Put the eye centre a third of the way down the crop.
    M[0, 2] += size / 2 - eye_center[0]
    M[1, 2] += size * 0.38 - eye_center[1]
    _ = nose
    return cv2.warpAffine(frame_bgr, M, (size, size), flags=cv2.INTER_LINEAR)


class OnnxArcFaceBackend(EmbeddingBackend):
    """ArcFace-style embedding via an ONNX model you supply.

    Args:
        model_path: Path to an ONNX face-recognition model with a
            ``1x3x112x112`` input and a 512-d output (e.g. ArcFace r100).
    """

    name = "onnx-arcface"
    default_threshold = 0.36     # cosine; typical ArcFace operating point

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else None
        self._session = None

    @property
    def available(self) -> bool:
        if self.model_path is None or not self.model_path.exists():
            return False
        try:
            importlib.import_module("onnxruntime")
        except Exception:
            return False
        return True

    def _load(self):
        if self._session is None:
            import onnxruntime  # noqa: PLC0415
            self._session = onnxruntime.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"])
        return self._session

    def embed(self, frame_bgr, landmarks, width: int, height: int) -> np.ndarray:
        crop = _aligned_crop(frame_bgr, landmarks, width, height, 112)
        if crop is None:
            raise ValueError("could not align face crop")
        blob = crop.astype(np.float32)[:, :, ::-1]          # BGR -> RGB
        blob = (blob - 127.5) / 127.5
        blob = np.transpose(blob, (2, 0, 1))[None, ...]     # NCHW
        sess = self._load()
        out = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0]
        n = np.linalg.norm(out)
        return out / n if n > 0 else out


class InsightFaceBackend(EmbeddingBackend):
    """InsightFace's packaged ArcFace. Downloads weights on first use."""

    name = "insightface"
    default_threshold = 0.36

    def __init__(self, model_name: str = "buffalo_l") -> None:
        self.model_name = model_name
        self._app = None

    @property
    def available(self) -> bool:
        try:
            importlib.import_module("insightface")
            return True
        except Exception:
            return False

    def _load(self):
        if self._app is None:
            import insightface  # noqa: PLC0415
            self._app = insightface.app.FaceAnalysis(name=self.model_name)
            self._app.prepare(ctx_id=-1)
        return self._app

    def embed(self, frame_bgr, landmarks, width: int, height: int) -> np.ndarray:
        faces = self._load().get(frame_bgr)
        if not faces:
            raise ValueError("insightface found no face")
        emb = faces[0].normed_embedding
        return np.asarray(emb, dtype=float)


def get_backend(name: str = "auto", model_path: str | Path | None = None) -> EmbeddingBackend:
    """Return an embedding backend.

    Args:
        name: ``"geometric"``, ``"onnx"``, ``"insightface"``, or ``"auto"``.
            ``"auto"`` prefers a deep backend when one is usable, otherwise
            falls back to geometric.
        model_path: ONNX model file, required for the ``"onnx"`` backend.
    """
    name = name.lower()
    if name == "geometric":
        return GeometricBackend()
    if name == "onnx":
        b = OnnxArcFaceBackend(model_path)
        if not b.available:
            raise RuntimeError(
                "ONNX backend unavailable: install onnxruntime and pass a valid "
                "--embedding-model path to an ArcFace .onnx file.")
        return b
    if name == "insightface":
        b = InsightFaceBackend()
        if not b.available:
            raise RuntimeError("insightface is not installed: pip install insightface")
        return b
    if name == "auto":
        for candidate in (OnnxArcFaceBackend(model_path), InsightFaceBackend()):
            if candidate.available:
                return candidate
        return GeometricBackend()
    raise ValueError(f"unknown backend {name!r}")


def describe_backends(model_path: str | Path | None = None) -> dict[str, bool]:
    """Which backends are usable right now."""
    return {
        "geometric": True,
        "onnx-arcface": OnnxArcFaceBackend(model_path).available,
        "insightface": InsightFaceBackend().available,
    }
