"""Model management.

MediaPipe's Tasks API needs a ``.task`` model bundle on disk. This module
downloads it once (on first run) from Google's official model storage and
caches it locally so subsequent runs are offline-friendly.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# Official MediaPipe face landmarker bundle (478 landmarks, incl. irises).
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Where the model is cached by default (project_root/models/...).
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    """Render a simple download progress bar on stderr."""
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(downloaded / total_size, 1.0)
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    mb = total_size / (1024 * 1024)
    sys.stderr.write(f"\r  [{bar}] {pct * 100:5.1f}%  ({mb:.1f} MB)")
    sys.stderr.flush()
    if pct >= 1.0:
        sys.stderr.write("\n")


def ensure_model(
    path: str | Path = DEFAULT_MODEL_PATH,
    url: str = DEFAULT_MODEL_URL,
    *,
    force: bool = False,
) -> Path:
    """Return a local path to the model, downloading it if necessary.

    Args:
        path: Where the model should live on disk.
        url:  Source URL for the model bundle.
        force: Re-download even if a cached copy exists.

    Returns:
        The resolved :class:`Path` to the model file.
    """
    path = Path(path).expanduser().resolve()
    if path.exists() and path.stat().st_size > 0 and not force:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"Downloading face landmarker model -> {path}", file=sys.stderr)
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    except Exception as exc:  # noqa: BLE001 - surface a friendly hint
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download the model from {url}.\n"
            f"Reason: {exc}\n"
            "If you are offline or behind a firewall, download it manually and "
            "pass its path with --model."
        ) from exc

    tmp.replace(path)
    return path
