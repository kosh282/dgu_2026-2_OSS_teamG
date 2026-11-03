"""Threaded webcam capture.

A dedicated thread continuously pulls frames so the main loop always processes
the *freshest* frame instead of blocking on camera I/O — which lifts effective
FPS and cuts input lag on a live webcam. Also handles a requested capture
resolution/FPS, optional mirroring, and auto-reconnect if the device drops.

Not used for video files: there you want every frame, in order, so the plain
``cv2.VideoCapture`` path is kept for those.
"""

from __future__ import annotations

import threading
import time

import cv2


class ThreadedCamera:
    """Background-threaded camera reader.

    Args:
        source: Device index or URL.
        width, height, fps: Requested capture properties (best effort).
        mirror: Horizontally flip frames (natural for a webcam).
        reconnect: Try to reopen the device if reads start failing.
        cap: Inject a pre-built capture object (used in tests).
    """

    def __init__(self, source=0, width=None, height=None, fps=None,
                 mirror=True, reconnect=True, cap=None) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.mirror = mirror
        self.reconnect = reconnect

        self._cap = cap if cap is not None else self._open()
        self._frame = None
        self._ok = False
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._fail_count = 0

    def _open(self):
        cap = cv2.VideoCapture(self.source)
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        return cap

    def isOpened(self) -> bool:
        return bool(self._cap and self._cap.isOpened())

    @property
    def frame_size(self) -> tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def start(self) -> "ThreadedCamera":
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                self._fail_count += 1
                if self.reconnect and self._fail_count > 30:
                    self._reopen()
                time.sleep(0.005)
                continue
            self._fail_count = 0
            if self.mirror:
                frame = cv2.flip(frame, 1)
            with self._lock:
                self._frame = frame
                self._ok = True

    def _reopen(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass
        self._cap = self._open()
        self._fail_count = 0

    def read(self):
        """Return ``(ok, frame)`` for the most recent frame (a copy)."""
        with self._lock:
            if not self._ok or self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self._cap.release()
        except Exception:
            pass

    def __enter__(self) -> "ThreadedCamera":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
