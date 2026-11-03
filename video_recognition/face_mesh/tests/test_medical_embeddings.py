"""Pluggable embedding backends and the geometric backend's known limits."""
import numpy as np
import pytest

from facemesh.medical.embeddings import (
    EmbeddingBackend, GeometricBackend, InsightFaceBackend,
    OnnxArcFaceBackend, _aligned_crop, describe_backends, get_backend,
)
from facemesh.medical.identity import AppearanceTracker, FaceGallery


class LM:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def _person(seed, expression=0.0, n=478):
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 0.05, (n, 3))
    p[10] = [0, -0.35, 0]
    p[152] = [0, 0.40, 0]
    p[168] = [0, 0, 0]
    p[33] = [-0.10 - 0.01 * (seed % 3), 0, 0]
    p[263] = [0.10 + 0.01 * (seed % 3), 0, 0]
    p[1] = [0, 0.15 + 0.02 * (seed % 4), 0.04]
    p[234] = [-0.20 - 0.02 * (seed % 5), 0.05, 0]
    p[454] = [0.20 + 0.02 * (seed % 5), 0.05, 0]
    p[60:95] += np.array([0.0, expression * 0.06, 0.0])   # mouth moves with expression
    return [LM(*r) for r in p]


class StubDeep(EmbeddingBackend):
    """Simulates a model that keys on identity and ignores expression."""

    name = "stub-deep"
    default_threshold = 0.36

    @property
    def available(self):
        return True

    def embed(self, frame, lms, w, h):
        idx = [10, 152, 168, 33, 263, 1, 234, 454]
        v = np.array([[lms[i].x, lms[i].y, lms[i].z] for i in idx]).ravel()
        return v / np.linalg.norm(v)


def test_geometric_backend_always_available():
    assert GeometricBackend().available
    assert describe_backends()["geometric"] is True


def test_geometric_embedding_is_expression_sensitive():
    """Documented weakness: a smile can break a match at the default threshold."""
    geo = GeometricBackend()
    neutral = geo.embed(None, _person(1, 0.0), 640, 480)
    smiling = geo.embed(None, _person(1, 1.0), 640, 480)
    same_person_across_expression = geo.similarity(neutral, smiling)
    assert same_person_across_expression < geo.default_threshold


def test_geometric_still_separates_identity_more_than_expression():
    geo = GeometricBackend()
    neutral = geo.embed(None, _person(1, 0.0), 640, 480)
    smiling = geo.embed(None, _person(1, 1.0), 640, 480)
    other = geo.embed(None, _person(2, 0.0), 640, 480)
    assert geo.similarity(neutral, smiling) > geo.similarity(neutral, other)


def test_stub_deep_backend_is_expression_invariant():
    deep = StubDeep()
    same = deep.similarity(deep.embed(None, _person(1, 0.0), 640, 480),
                           deep.embed(None, _person(1, 1.0), 640, 480))
    other = deep.similarity(deep.embed(None, _person(1, 0.0), 640, 480),
                            deep.embed(None, _person(2, 0.0), 640, 480))
    assert same > other


def test_gallery_inherits_backend_threshold():
    assert FaceGallery(backend=GeometricBackend()).threshold == 0.9995
    g = FaceGallery(backend=StubDeep())
    assert g.threshold == 0.36 and g.backend_name == "stub-deep"


def test_gallery_records_backend_name(tmp_path):
    g = FaceGallery(backend=StubDeep())
    g.enroll("ana", np.ones(8) / np.sqrt(8))
    g.save(tmp_path / "g.json")
    assert FaceGallery.load(tmp_path / "g.json").backend_name == "stub-deep"


def test_tracker_uses_injected_backend():
    tr = AppearanceTracker(backend=StubDeep())
    frame = np.zeros((480, 640, 3), np.uint8)
    a = tr.update([_person(1), _person(2)], 640, 480, frame)
    b = tr.update([_person(2), _person(1)], 640, 480, frame)
    assert b == [a[1], a[0]]


def test_tracker_defaults_to_geometric_without_backend():
    tr = AppearanceTracker()
    assert tr.update([_person(1)], 640, 480) == ["person_0"]


def test_get_backend_auto_falls_back_to_geometric():
    assert get_backend("auto").name == "geometric"


def test_get_backend_geometric_explicit():
    assert isinstance(get_backend("geometric"), GeometricBackend)


def test_get_backend_onnx_without_model_raises():
    with pytest.raises(RuntimeError, match="ONNX backend unavailable"):
        get_backend("onnx", model_path=None)


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("nope")


def test_onnx_backend_unavailable_without_model():
    assert OnnxArcFaceBackend(None).available is False


def test_insightface_backend_reports_availability():
    assert isinstance(InsightFaceBackend().available, bool)


def test_aligned_crop_is_canonical_square():
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    lms = _person(1)
    lms[33] = LM(0.40, 0.42)
    lms[263] = LM(0.60, 0.42)
    lms[1] = LM(0.5, 0.55)
    crop = _aligned_crop(frame, lms, 640, 480, 112)
    assert crop is not None and crop.shape == (112, 112, 3)


def test_aligned_crop_rejects_degenerate_eyes():
    frame = np.zeros((480, 640, 3), np.uint8)
    lms = _person(1)
    lms[33] = LM(0.5, 0.5)
    lms[263] = LM(0.5, 0.5)      # eyes coincide
    assert _aligned_crop(frame, lms, 640, 480) is None
