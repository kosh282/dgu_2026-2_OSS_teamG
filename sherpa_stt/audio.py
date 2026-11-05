"""메모리 전용 오디오 청크. 계약 5절 `AudioChunk`에 대응하며 JSON으로 저장하지 않는다."""
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    session_id: str
    chunk_id: int
    start_ms: int
    end_ms: int
    sample_rate_hz: int
    channels: int
    sample_format: str
    samples: Any  # numpy 배열. (n,) 또는 (n, channels)

    def __post_init__(self):
        for name in ("chunk_id", "start_ms", "end_ms", "sample_rate_hz", "channels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name}은 0 이상의 정수여야 합니다")
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms는 start_ms보다 작을 수 없습니다")
        if self.sample_rate_hz == 0 or self.channels == 0:
            raise ValueError("sample_rate_hz와 channels는 양수여야 합니다")


class ChunkClock:
    """캡처 청크에 세션 공통 시각을 붙인다.

    영상 분석과 같은 `service.elapsed_ms(session_id)`를 기준으로 한다. 콜백 도착 지터로
    구간이 흔들리지 않도록 샘플 수로 시각을 이어 붙이고, 세션 시계와 `resync_ms` 이상
    뒤처지면(장치 지연·청크 누락) 세션 시계에 다시 맞춘다. 버퍼가 몰려 들어와 샘플 위치가
    세션 시계보다 앞서면 끝이 현재 시각이 되도록 끌어내린다. 끝 시각은 현재 세션 시각을
    넘지 않는다.
    """

    def __init__(self, service, session_id, sample_rate_hz, resync_ms=200):
        self._service = service
        self._session_id = session_id
        self._sample_rate_hz = sample_rate_hz
        self._resync_ms = resync_ms
        self._position_ms = None
        self.resync_count = 0

    def stamp(self, frames):
        now = self._service.elapsed_ms(self._session_id)
        duration = frames * 1000 / self._sample_rate_hz
        target = now - duration
        start = self._position_ms
        if start is None or start > target:
            start = target
        elif target - start > self._resync_ms:
            start = target
            self.resync_count += 1
        start = max(start, 0.0)
        end = min(start + duration, now)
        self._position_ms = start + duration
        return int(start), int(end)


def to_mono_float32(chunk):
    """sherpa-onnx 입력 형식(-1~1 범위 float32 mono)으로 변환한다."""
    samples = np.asarray(chunk.samples)
    if samples.ndim == 2:
        if samples.shape[1] != chunk.channels:
            raise ValueError("samples의 채널 수가 channels와 다릅니다")
    elif samples.ndim != 1 or chunk.channels != 1:
        raise ValueError("samples는 (n,) mono 또는 (n, channels) 배열이어야 합니다")
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    elif samples.dtype == np.int32:
        samples = samples.astype(np.float32) / 2147483648.0
    elif np.issubdtype(samples.dtype, np.floating):
        samples = samples.astype(np.float32, copy=False)
    else:
        raise ValueError(f"지원하지 않는 샘플 형식입니다: {samples.dtype}")
    if samples.ndim == 2:
        samples = samples.mean(axis=1, dtype=np.float32)
    return np.ascontiguousarray(samples)
