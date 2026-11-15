"""말 멈춤 지점에서 오디오를 잘라 faster-whisper 입력 구간을 만든다.

고정 길이(예: 10초)로 자르면 단어가 구간 경계에서 잘리고, 말소리가 없는 구간에서
Whisper가 없는 문장("시청해 주셔서 감사합니다" 등)을 만들어 낸다. 이 분할기는 20ms 프레임
에너지를 배경 잡음 바닥과 비교해 말소리 여부를 추정하고,

- 최소 길이 이상 모였고 말이 `pause_ms` 이상 멈추면 그 지점에서 자른다.
- 최대 길이에 이르면 마지막 몇 초 안에서 가장 조용한 프레임에서 자른다.
- 말소리 프레임이 없는 구간은 인식기에 보내지 않는다.

에너지 기준이므로 큰 배경음악은 말소리로 볼 수 있다. 값은 검증 전 실험값이다.
오디오는 메모리에서만 다룬다.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SpeechSegment:
    audio: np.ndarray
    start_ms: int
    end_ms: int
    speech_frames: int


class PauseSegmenter:
    def __init__(self, sample_rate_hz=16000, min_seconds=3.0, max_seconds=12.0, pause_ms=500,
                 frame_ms=20, margin_db=8.0, floor_dbfs=-65.0, search_seconds=3.0, keep_silence_ms=300):
        if not 0 < min_seconds <= max_seconds:
            raise ValueError('0 < min_seconds <= max_seconds여야 합니다')
        self.sample_rate_hz = sample_rate_hz
        self._frame = max(1, int(sample_rate_hz * frame_ms / 1000))
        self._min = int(min_seconds * sample_rate_hz)
        self._max = int(max_seconds * sample_rate_hz)
        self._pause_frames = max(1, round(pause_ms / frame_ms))
        self._search = int(search_seconds * sample_rate_hz)
        self._keep = int(keep_silence_ms * sample_rate_hz / 1000)
        self._margin = margin_db
        self._floor = floor_dbfs
        self.noise_dbfs = None
        self._reset_buffer()

    def _reset_buffer(self):
        self._audio = np.zeros(0, np.float32)
        self._marks = []        # (버퍼 내 샘플 위치, 청크 시작 ms, 청크 끝 ms, 청크 샘플 수)
        self._levels = []       # 프레임별 dBFS
        self._speech = []       # 프레임별 말소리 여부
        self._pending = np.zeros(0, np.float32)  # 프레임 단위로 아직 나누지 못한 꼬리

    # 시각 변환 -------------------------------------------------------------
    def _time_at(self, index):
        for position, start_ms, end_ms, count in reversed(self._marks):
            if index >= position:
                ratio = min(1.0, (index - position) / count) if count else 0.0
                return int(round(start_ms + ratio * (end_ms - start_ms)))
        return self._marks[0][1] if self._marks else 0

    # 내부 ------------------------------------------------------------------
    def _measure(self, samples):
        data = np.concatenate([self._pending, samples])
        usable = len(data) // self._frame * self._frame
        self._pending = data[usable:]
        for frame in data[:usable].reshape(-1, self._frame):
            level = 20 * float(np.log10(max(float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))), 1e-10)))
            # 잡음 바닥: 더 작으면 바로 내리고 평소에는 프레임당 0.02dB(초당 1dB)씩 올린다.
            # 말하는 도중에 시작해도 첫 소리를 잡음으로 보지 않도록 `floor_dbfs`에서 출발한다.
            base = self._floor if self.noise_dbfs is None else self.noise_dbfs + 0.02
            self.noise_dbfs = min(level, base)
            self._levels.append(level)
            self._speech.append(level > self._floor and level >= self.noise_dbfs + self._margin)

    def _cut(self, frame_index):
        """`frame_index` 프레임 앞에서 잘라 앞부분을 구간으로 반환하고 뒷부분은 버퍼에 남긴다."""
        cut = min(len(self._audio), frame_index * self._frame)
        segment = SpeechSegment(self._audio[:cut].copy(), self._time_at(0), self._time_at(cut),
                                int(sum(self._speech[:frame_index])))
        rest_audio = self._audio[cut:]
        rest_marks = []
        for position, start_ms, end_ms, count in self._marks:
            if position + count <= cut:
                continue
            offset = max(0, cut - position)
            ratio = offset / count if count else 0.0
            rest_marks.append((max(0, position - cut), int(round(start_ms + ratio * (end_ms - start_ms))),
                               end_ms, count - offset))
        levels, speech, pending = self._levels[frame_index:], self._speech[frame_index:], self._pending
        self._audio, self._marks, self._levels, self._speech, self._pending = (
            rest_audio, rest_marks, levels, speech, pending)
        return segment if segment.speech_frames and len(segment.audio) else None

    def _trailing_silence(self):
        count = 0
        for value in reversed(self._speech):
            if value:
                break
            count += 1
        return count

    def _drop_leading_silence(self):
        """아직 말소리가 없으면 오래된 무음을 버리고 최근 `keep_silence_ms`만 남긴다."""
        if any(self._speech):
            return
        keep_frames = max(1, self._keep // self._frame)
        if len(self._speech) > keep_frames * 2:
            self._cut(len(self._speech) - keep_frames)

    # 공개 API --------------------------------------------------------------
    def push(self, samples, start_ms, end_ms):
        """연속된 청크 하나를 넣고 완성된 말소리 구간 목록을 반환한다."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not len(samples):
            return []
        self._marks.append((len(self._audio), int(start_ms), int(end_ms), len(samples)))
        self._audio = np.concatenate([self._audio, samples])
        self._measure(samples)
        self._drop_leading_silence()
        segments = []
        while True:
            size = len(self._audio)
            silence = self._trailing_silence()
            if size >= self._min and silence >= self._pause_frames and any(self._speech):
                # 멈춘 지점 뒤로 약간의 무음을 포함해 자르고 나머지는 다음 구간 앞부분으로 넘긴다.
                keep_frames = min(silence, max(1, self._keep // self._frame))
                piece = self._cut(len(self._speech) - silence + keep_frames)
            elif size >= self._max:
                frames = len(self._levels)
                window = max(1, min(frames, self._search // self._frame))
                low = frames - window
                quietest = low + int(np.argmin(self._levels[low:]))
                piece = self._cut(max(1, quietest))
            else:
                break
            if piece is not None:
                segments.append(piece)
        return segments

    def flush(self):
        """남은 오디오를 마지막 구간으로 반환하고 상태를 비운다(오디오 누락·종료 시)."""
        piece = self._cut(len(self._speech) + (1 if len(self._pending) else 0)) if len(self._audio) else None
        self._reset_buffer()
        return [piece] if piece is not None else []
