"""10초 단위 faster-whisper 전사를 공통 세션에 전달한다."""
from dataclasses import dataclass
from collections import deque
from pathlib import Path
import queue
import time
from uuid import uuid4

import numpy as np

from sherpa_stt.audio import to_mono_float32
from sherpa_stt.runner import LiveTranscription


@dataclass(frozen=True)
class WhisperConfig:
    model: str = 'base'
    sample_rate_hz: int = 16000
    chunk_ms: int = 100
    window_seconds: int = 10
    num_threads: int = 4

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError('Whisper 모델 이름이 필요합니다')
        for name in ('sample_rate_hz', 'chunk_ms', 'window_seconds', 'num_threads'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name}는 양의 정수여야 합니다')
        if self.sample_rate_hz != 16000:
            raise ValueError('faster-whisper 입력은 16kHz가 필요합니다')

    def public_info(self):
        return {'model_id': f'faster-whisper-{self.model}', 'model_version': '1.2.1',
                'sample_rate_hz': self.sample_rate_hz, 'window_seconds': self.window_seconds}


def load_whisper(config):
    from faster_whisper import WhisperModel
    return WhisperModel(config.model, device='cpu', compute_type='int8', cpu_threads=config.num_threads,
                        download_root=str(Path(__file__).resolve().parent / 'models'))


class WhisperLiveTranscription(LiveTranscription):
    """같은 캡처·세션 시계를 사용하며 겹치지 않는 오디오 구간을 한 번씩 인식한다."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._recent_levels = deque(maxlen=20)
        self._processed_windows = 0
        self._empty_windows = 0

    def stats(self):
        stats = super().stats()
        with self._stats_lock:
            peak = max(self._recent_levels) if self._recent_levels else None
            stats['capture_level_dbfs'] = (round(20 * float(np.log10(max(peak, 1e-10))), 1)
                                          if peak is not None else None)
            stats['processed_windows'] = self._processed_windows
            stats['empty_windows'] = self._empty_windows
        return stats

    def start(self):
        if self._thread is not None:
            raise RuntimeError('이미 시작했습니다')
        try:
            self._engine = self._engine or load_whisper(self._config)
            self._capture = self._capture or self._make_capture()
            self._capture.start()
        except Exception as error:
            self.last_error = error
            self._stop_capture()
            self._status('unavailable', f'전사를 시작하지 못했습니다: {error}', ['audio_unavailable'])
            return False
        self._status('running', 'faster-whisper 10초 구간 전사를 시작했습니다.')
        import threading
        self._thread = threading.Thread(target=self._loop, name='whisper-stt', daemon=True)
        self._thread.start()
        return True

    def _loop(self):
        chunks = self._capture.chunks
        parts, count, start_ms, end_ms, last_id = [], 0, None, None, None
        limit = self._config.window_seconds * self._config.sample_rate_hz

        def flush():
            nonlocal parts, count, start_ms, end_ms
            if not count:
                return
            samples = np.concatenate(parts)
            began = time.perf_counter()
            segments, _ = self._engine.transcribe(samples, language='ko', beam_size=1,
                                                   vad_filter=False, word_timestamps=False,
                                                   condition_on_previous_text=False)
            text = ''.join(segment.text for segment in segments).strip()
            elapsed = time.perf_counter() - began
            lag = max(0, self._service.elapsed_ms(self._session_id) - end_ms)
            if text:
                info = self._config.public_info()
                event = self._service.make_event(self._session_id, 'transcript_final',
                    {'utterance_id': str(uuid4()), 'revision': 1, 'text': text, 'is_edited': False},
                    start_ms, end_ms, 'speech', model_id=info['model_id'], model_version=info['model_version'])
                self._service.receive_event(event)
            with self._stats_lock:
                self._process_s += elapsed
                self._processed_windows += 1
                self._empty_windows += int(not text)
                self._audio_ms += count * 1000 / self._config.sample_rate_hz
                self._lags.append(lag)
                self._last_lag_ms = lag
                if len(self._lags) > 36000:
                    del self._lags[:6000]
            parts, count, start_ms, end_ms = [], 0, None, None

        try:
            while not self._stop.is_set() or not chunks.empty():
                try:
                    chunk = chunks.get(timeout=0.1)
                except queue.Empty:
                    error = getattr(self._capture, 'error', None)
                    if error is not None and not self._stop.is_set():
                        raise RuntimeError(f'오디오 입력 중단: {error}')
                    continue
                if last_id is not None and chunk.chunk_id != last_id + 1:
                    flush()
                    self._status('running', '오디오 청크 누락으로 전사 구간을 나눴습니다.', ['audio_unavailable'])
                last_id = chunk.chunk_id
                if chunk.sample_rate_hz != self._config.sample_rate_hz:
                    raise ValueError('faster-whisper 입력은 16kHz가 필요합니다')
                audio = to_mono_float32(chunk)
                if len(audio):
                    level = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
                    with self._stats_lock:
                        self._recent_levels.append(level)
                offset = 0
                while offset < len(audio):
                    take = min(limit - count, len(audio) - offset)
                    span = chunk.end_ms - chunk.start_ms
                    piece_start = chunk.start_ms + int(span * offset / len(audio))
                    piece_end = chunk.start_ms + int(span * (offset + take) / len(audio))
                    if start_ms is None:
                        start_ms = piece_start
                    parts.append(audio[offset:offset + take].copy())
                    count += take
                    end_ms = max(start_ms, piece_end)
                    offset += take
                    if count == limit:
                        flush()
            flush()
        except Exception as error:
            self.last_error = error
            self._safe_status('unavailable', f'faster-whisper 전사 오류: {error}', ['unknown'])
            self._stop_capture()

    def stop(self):
        self._stop_capture()
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self.last_error is None and not self._finished:
            self._finished = True
            self._status('stopped', 'faster-whisper 전사를 종료했습니다.')
