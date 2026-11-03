"""Detector wrapper around MediaPipe's Tasks ``FaceLandmarker``.

The legacy ``mp.solutions.face_mesh`` API was removed in recent MediaPipe
releases (0.10.30+), so this project targets the current Tasks API. The
wrapper hides the running-mode plumbing:

- ``image``       — single-shot ``detect``.
- ``video``       — ``detect_for_video`` with monotonic timestamps (tracking).
- ``live_stream`` — non-blocking ``detect_async``; results arrive on a
  callback thread and the most recent one is returned immediately, so the
  capture loop never stalls on inference (lower latency).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarkerResult,
    RunningMode,
)

_RUNNING_MODES = {
    "image": RunningMode.IMAGE,
    "video": RunningMode.VIDEO,
    "live_stream": RunningMode.LIVE_STREAM,
}

_DELEGATES = {
    "cpu": BaseOptions.Delegate.CPU,
    "gpu": BaseOptions.Delegate.GPU,
}

_EMPTY_RESULT = FaceLandmarkerResult([], [], [])


def _make_landmarker(kwargs: dict) -> FaceLandmarker:
    """Build the landmarker, falling back to CPU if a GPU delegate fails."""
    try:
        return FaceLandmarker.create_from_options(FaceLandmarkerOptions(**kwargs))
    except Exception:
        base = kwargs.get("base_options")
        if base is not None and base.delegate == BaseOptions.Delegate.GPU:
            import sys
            print("note: GPU delegate unavailable; falling back to CPU.", file=sys.stderr)
            base.delegate = BaseOptions.Delegate.CPU
            return FaceLandmarker.create_from_options(FaceLandmarkerOptions(**kwargs))
        raise


class FaceMeshDetector:
    """Detect 478 facial landmarks per face.

    Args:
        model_path: Path to a ``face_landmarker.task`` bundle.
        running_mode: ``"image"``, ``"video"`` or ``"live_stream"``.
        num_faces: Maximum number of faces to track.
        min_face_detection_confidence: Detection threshold in ``[0, 1]``.
        min_face_presence_confidence: Presence threshold in ``[0, 1]``.
        min_tracking_confidence: Tracking threshold in ``[0, 1]``.
        output_blendshapes: Also return 52 expression blendshape scores.
        output_transformation_matrixes: Also return 4x4 head-pose matrices.
    """

    def __init__(
        self,
        model_path: str | Path,
        running_mode: str = "video",
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        output_blendshapes: bool = False,
        output_transformation_matrixes: bool = False,
        delegate: str = "cpu",
    ) -> None:
        if running_mode not in _RUNNING_MODES:
            raise ValueError(
                f"running_mode must be one of {list(_RUNNING_MODES)}, got {running_mode!r}"
            )
        if delegate not in _DELEGATES:
            raise ValueError(f"delegate must be 'cpu' or 'gpu', got {delegate!r}")
        self.running_mode = running_mode
        self._last_ts_ms = -1

        # Live-stream results arrive asynchronously; guard the latest one.
        self._lock = threading.Lock()
        self._latest: FaceLandmarkerResult = _EMPTY_RESULT

        kwargs = dict(
            base_options=BaseOptions(
                model_asset_path=str(model_path), delegate=_DELEGATES[delegate]
            ),
            running_mode=_RUNNING_MODES[running_mode],
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=output_blendshapes,
            output_facial_transformation_matrixes=output_transformation_matrixes,
        )
        if running_mode == "live_stream":
            kwargs["result_callback"] = self._on_result

        self._landmarker = _make_landmarker(kwargs)

    def _on_result(self, result, output_image, timestamp_ms: int) -> None:
        """Callback invoked by MediaPipe on its own thread (live stream)."""
        with self._lock:
            self._latest = result

    def detect(self, frame_bgr, timestamp_ms: int | None = None) -> FaceLandmarkerResult:
        """Run detection on a single BGR frame (OpenCV's native format).

        In ``image``/``video`` mode this returns the result for *this* frame.
        In ``live_stream`` mode it submits the frame and returns the most
        recent result available (which may lag by a frame or two) without
        blocking on inference.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.running_mode == "image":
            return self._landmarker.detect(mp_image)

        # VIDEO / LIVE_STREAM both require strictly increasing timestamps.
        if timestamp_ms is None:
            timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self._last_ts_ms:
            timestamp_ms = self._last_ts_ms + 1
        self._last_ts_ms = timestamp_ms

        if self.running_mode == "video":
            return self._landmarker.detect_for_video(mp_image, timestamp_ms)

        # live_stream: fire and forget, return whatever the callback last stored.
        self._landmarker.detect_async(mp_image, timestamp_ms)
        with self._lock:
            return self._latest

    def close(self) -> None:
        """Release the underlying MediaPipe graph."""
        self._landmarker.close()

    def __enter__(self) -> "FaceMeshDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
