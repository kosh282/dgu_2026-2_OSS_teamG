"""Shared test helpers: lightweight synthetic landmarks."""
from dataclasses import dataclass

import pytest


@dataclass
class LM:
    x: float = 0.5
    y: float = 0.5
    z: float = 0.0


@pytest.fixture
def lm_cls():
    """The lightweight landmark class, exposed as a fixture."""
    return LM


@pytest.fixture
def make_landmarks():
    def _make(n: int = 478):
        return [LM() for _ in range(n)]
    return _make


@pytest.fixture
def eye_setter():
    """Set a 6-point eye to a given openness (0=closed, 1=open)."""
    def _set(lm, idx, openness):
        p1, p2, p3, p4, p5, p6 = idx
        lm[p1].x, lm[p1].y = 0.0, 0.0
        lm[p4].x, lm[p4].y = 0.10, 0.0
        g = 0.03 * openness
        lm[p2].x, lm[p2].y = 0.03, g
        lm[p3].x, lm[p3].y = 0.06, g
        lm[p6].x, lm[p6].y = 0.03, -g
        lm[p5].x, lm[p5].y = 0.06, -g
    return _set

# v1.4.0 analysis tools: let tests import the standalone scripts by bare name
import sys as _sys
from pathlib import Path as _Path
_p = _Path(__file__).resolve().parent.parent / "scripts"
if str(_p) not in _sys.path:
    _sys.path.insert(0, str(_p))
