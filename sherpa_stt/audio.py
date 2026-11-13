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


def _dbfs(rms):
    return 20.0 * float(np.log10(max(rms, 1e-10)))


class AutoGain:
    """인식기 입력 전 말소리 크기를 일정하게 맞추는 자동 음량 조절.

    sherpa 한국어 모델은 말소리 RMS가 약 -40 dBFS보다 작으면 인식 결과가 사라진다(실측).
    마이크가 멀거나 장치 볼륨·영상 소리가 작을 때 이 범위로 떨어지므로, 말소리로 판단한
    청크의 크기를 추적해 `target_dbfs`에 가깝게 키운다.

    - 키우기만 하고 줄이지 않는다(0 ~ `max_gain_db`). 원래 큰 소리는 그대로 둔다.
    - 배경 잡음 바닥을 따로 추적해 잡음보다 `speech_margin_db` 이상 큰 청크만 말소리로 본다.
      무음 구간에서는 이득을 유지하므로 잡음을 따라 이득이 흔들리지 않는다.
    - 이득 변화는 청크 안에서 선형으로 이어 붙여 딸깍 소리가 나지 않게 한다.
    - 찌그러짐은 짧은 구간 리미터로 막는다. 키운 소리가 -1 dBFS를 넘는 5ms 구간만 즉시 줄이고
      약 50ms에 걸쳐 되돌린다. 예전처럼 청크의 최고값에 맞춰 청크 전체 이득을 깎으면, 평균은
      작고 파열음만 큰 노트북 마이크 말소리(-34 dBFS)가 +10dB 안팎밖에 커지지 않았다(실측).
    - 이번 청크가 이미 목표보다 6dB 이상 커질 만큼 크면 이득을 바로 낮춘다(조용한 첫머리 뒤 큰 소리).
    오디오는 메모리에서만 변환하며 저장하지 않는다. 값은 검증 전 실험값이다.
    """

    def __init__(self, target_dbfs=-20.0, max_gain_db=30.0, speech_margin_db=10.0,
                 floor_dbfs=-80.0, max_step_up_db=3.0, sample_rate_hz=16000,
                 ceiling_dbfs=-1.0, block_ms=5.0, release_db_per_50ms=6.0):
        self.target_dbfs = float(target_dbfs)
        self.max_gain_db = float(max_gain_db)
        self.speech_margin_db = float(speech_margin_db)
        self.floor_dbfs = float(floor_dbfs)
        self.max_step_up_db = float(max_step_up_db)
        self.noise_dbfs = None   # 배경 잡음 바닥 추정
        self.speech_dbfs = None  # 최근 말소리 크기 추정(원본 기준)
        self.gain_db = 0.0
        # 짧은 구간 리미터 상태(이전 청크에서 이어짐)
        self._ceiling = 10 ** (float(ceiling_dbfs) / 20)
        self._block = max(1, int(round(sample_rate_hz * block_ms / 1000)))
        self._release = 10 ** (float(release_db_per_50ms) * block_ms / 50 / 20)
        self._limiter_gain = 1.0
        self.limited_blocks = 0  # 리미터가 줄인 5ms 구간 수(진단용)

    def _limit_peaks(self, samples):
        """-1 dBFS를 넘는 5ms 구간만 줄이고 천천히 되돌린다. 청크 경계를 넘어 상태를 잇는다."""
        size = len(samples)
        blocks = -(-size // self._block)
        padded = np.zeros(blocks * self._block, dtype=np.float32)
        padded[:size] = np.abs(samples)
        peaks = padded.reshape(blocks, self._block).max(axis=1)
        needed = np.minimum(1.0, self._ceiling / np.maximum(peaks, 1e-9))
        gains = np.empty(blocks, dtype=np.float32)
        current = self._limiter_gain
        for index in range(blocks):
            current = min(needed[index], min(1.0, current * self._release))
            gains[index] = current
        self._limiter_gain = current
        self.limited_blocks += int(np.count_nonzero(gains < 0.999))
        return samples * np.repeat(gains, self._block)[:size]

    def process(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if not len(samples):
            return samples
        level = _dbfs(float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))))
        # 잡음 바닥: 더 작은 값이 오면 바로 내리고, 평소에는 천천히(청크당 0.1dB) 올린다.
        self.noise_dbfs = level if self.noise_dbfs is None else min(level, self.noise_dbfs + 0.1)
        is_speech = level > self.floor_dbfs and level >= self.noise_dbfs + self.speech_margin_db
        previous = self.gain_db
        if is_speech:
            if self.speech_dbfs is None:
                self.speech_dbfs = level
            else:
                weight = 0.3 if level > self.speech_dbfs else 0.1  # 커질 때는 빨리 따라가 잘림을 줄인다
                self.speech_dbfs += weight * (level - self.speech_dbfs)
            desired = min(max(self.target_dbfs - self.speech_dbfs, 0.0), self.max_gain_db)
            if previous == 0.0 and desired > 0.0 and self.speech_dbfs == level:
                self.gain_db = desired  # 첫 말소리는 바로 키워 첫 단어를 놓치지 않는다
            elif desired > previous:
                self.gain_db = min(desired, previous + self.max_step_up_db)
            else:
                self.gain_db = desired  # 소리가 커지면 바로 이득을 낮춘다
            # 이번 청크가 키운 뒤 목표보다 6dB 넘게 커지면 평균 추적을 기다리지 않고 바로 낮춘다.
            self.gain_db = max(0.0, min(self.gain_db, self.target_dbfs + 6.0 - level))
        start_gain = max(0.0, min(previous, self.target_dbfs + 6.0 - level)) if is_speech else previous
        if start_gain == self.gain_db == 0.0 and self._limiter_gain >= 1.0:
            return samples
        ramp = np.linspace(10 ** (start_gain / 20), 10 ** (self.gain_db / 20), len(samples), dtype=np.float32)
        return np.clip(self._limit_peaks(samples * ramp), -1.0, 1.0)

    @property
    def input_level_dbfs(self):
        return None if self.speech_dbfs is None else round(self.speech_dbfs, 1)


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
