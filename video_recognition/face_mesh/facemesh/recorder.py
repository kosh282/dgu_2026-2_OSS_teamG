"""On-demand recording of the annotated feed.

Toggle recording at runtime (the ``r`` key). Frames are written to a
timestamped MP4; the recorder tracks elapsed time so the UI can show a
REC indicator. Kept separate from ``--output`` (which records the whole run).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2


class VideoRecorder:
    """Start/stop writing annotated frames to a timestamped video.

    Args:
        out_dir: Directory for recordings.
        fps: Output frame rate.
        fourcc: Codec fourcc (default ``mp4v``).
        writer_factory: Injectable ``(path, fourcc, fps, size) -> writer`` for
            testing; defaults to ``cv2.VideoWriter``.
    """

    def __init__(self, out_dir: str | Path = ".", fps: float = 25.0,
                 fourcc: str = "mp4v", writer_factory=None) -> None:
        self.out_dir = Path(out_dir)
        self.fps = fps
        self.fourcc = fourcc
        self._writer_factory = writer_factory or self._default_writer
        self._writer = None
        self._start_t: float | None = None
        self.path: Path | None = None
        self.frames = 0

    @staticmethod
    def _default_writer(path, fourcc, fps, size):
        return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    def start(self, frame_size: tuple[int, int], now: float | None = None) -> Path:
        if self.is_recording:
            return self.path
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = self.out_dir / f"recording_{stamp}.mp4"
        self._writer = self._writer_factory(self.path, self.fourcc, self.fps, frame_size)
        self._start_t = now if now is not None else time.monotonic()
        self.frames = 0
        return self.path

    def write(self, frame) -> None:
        if self._writer is not None:
            self._writer.write(frame)
            self.frames += 1

    def elapsed(self, now: float | None = None) -> float:
        if self._start_t is None:
            return 0.0
        now = now if now is not None else time.monotonic()
        return now - self._start_t

    def stop(self) -> Path | None:
        path = self.path
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self._start_t = None
        return path

    def toggle(self, frame_size: tuple[int, int], now: float | None = None):
        """Start if stopped, stop if running. Returns (is_recording, path)."""
        if self.is_recording:
            p = self.stop()
            return False, p
        p = self.start(frame_size, now)
        return True, p
