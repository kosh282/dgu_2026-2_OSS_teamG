"""Small utilities: an FPS meter and input-source resolution."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}


class FPSMeter:
    """Rolling-average frames-per-second meter."""

    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)
        self._last = time.perf_counter()

    def tick(self) -> float:
        """Record a frame and return the current smoothed FPS."""
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now
        avg = sum(self._times) / len(self._times) if self._times else 0.0
        return 1.0 / avg if avg > 0 else 0.0


def resolve_source(source: str) -> tuple[str, object]:
    """Classify a CLI source string.

    Returns a ``(kind, value)`` pair where ``kind`` is one of
    ``"webcam"``, ``"image"`` or ``"video"``.

    - Integer-like strings ("0", "1") -> webcam device index.
    - Paths with image extensions     -> single image.
    - Everything else                 -> video file / stream URL.
    """
    if source.isdigit():
        return "webcam", int(source)

    ext = Path(source).suffix.lower()
    if ext in _IMAGE_EXTS:
        return "image", source
    if ext in _VIDEO_EXTS:
        return "video", source
    # Fall back to treating it as a video source (e.g. an RTSP/HTTP stream).
    return "video", source
