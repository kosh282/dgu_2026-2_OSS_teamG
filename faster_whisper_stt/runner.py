"""오디오 입력 → faster-whisper → 통합 서비스로 이어지는 말 멈춤 단위 전사 실행기."""
from dataclasses import dataclass
from collections import deque
from pathlib import Path
import queue
import re
import threading
import time
from uuid import uuid4

import numpy as np

from .audio import to_mono_float32
from .segmenter import PauseSegmenter

COMPONENT = 'stt'
SOURCES = {'loopback', 'microphone'}

# Whisper가 무음·잡음 구간에서 자주 만들어 내는 문장. 구간 전체가 이 문장 하나일 때만 버린다.
HALLUCINATED_PHRASES = frozenset(re.sub(r'[^0-9a-z가-힣]', '', text.lower()) for text in (
    '시청해주셔서 감사합니다', '시청해 주셔서 감사합니다', '시청해주셔서 고맙습니다', '오늘도 시청해주셔서 감사합니다',
    '구독과 좋아요', '구독과 좋아요 부탁드립니다', '다음 영상에서 만나요', '고맙습니다', '감사합니다',
    'MBC 뉴스', '자막 제공', '한국어 자막',
))


def _normalized(text):
    return re.sub(r'[^0-9a-z가-힣]', '', text.lower())


def _copies_hints(text, hint_terms):
    """출력이 용어 힌트 목록을 그대로 옮겨 적은 것인지(쉼표 항목 4개 이상, 75% 이상이 힌트)."""
    if not hint_terms:
        return False
    hints = {_normalized(term) for term in hint_terms}
    items = [_normalized(item) for item in re.split(r'[,、]', text) if _normalized(item)]
    return len(items) >= 4 and sum(item in hints for item in items) >= 0.75 * len(items)


def segment_text(segment, hint_terms=()):
    """Whisper 구간 결과에서 쓸 텍스트. 지어낸 문장·힌트 복사로 보이면 빈 문자열(실험 기준).

    실제로 같은 말을 반복했을 수 있으므로 텍스트는 고치지 않고 인식한 그대로 쓴다.
    구간 전체가 반복으로 채워진 경우(압축률 2.4 초과, faster-whisper의 반복 판정 기준)만 버린다.
    """
    text = (getattr(segment, 'text', '') or '').strip()
    normalized = _normalized(text)
    if not normalized or normalized in HALLUCINATED_PHRASES:
        return ''
    if getattr(segment, 'compression_ratio', 0.0) > 2.4:
        return ''
    if getattr(segment, 'no_speech_prob', 0.0) > 0.6 and getattr(segment, 'avg_logprob', 0.0) < -1.0:
        return ''
    if _copies_hints(text, hint_terms):
        return ''
    return text


def keep_segment(segment):
    """`segment_text`가 텍스트를 남기는지(이전 호출부 호환)."""
    return bool(segment_text(segment))


@dataclass(frozen=True)
class WhisperConfig:
    model: str = 'base'
    sample_rate_hz: int = 16000
    chunk_ms: int = 100
    # 한 구간의 최대 길이(초). 말이 멈추면 `min_segment_seconds` 이상에서 더 일찍 자른다.
    # Whisper는 구간 길이와 관계없이 30초 입력으로 계산하므로 짧게 자를수록 계산이 늘어난다.
    # 데모 영상 4분: 3~10초는 상대 CER 28.8%·RTF 0.65, 8~20초는 22.6%·0.34(스레드 2, 앞 문장 힌트).
    window_seconds: int = 20
    # 영상 분석과 CPU를 나눠 쓰도록 2개로 둔다(데모 영상 4분: 4개와 같은 정확도, 처리 시간 약 8% 증가).
    num_threads: int = 2
    min_segment_seconds: int = 8
    pause_ms: int = 500
    # 빔 5는 데모 영상에서 오차가 약간 줄었지만 처리 시간이 약 25% 늘어 기본은 1로 둔다.
    beam_size: int = 1
    # faster-whisper 내장 음성 구간 감지(Silero). 무음 구간의 지어낸 문장을 줄인다.
    vad_filter: bool = True
    # 직전 확정 문장 끝부분(최대 120자)을 다음 구간의 힌트로 넣어 문맥·용어를 잇는다.
    prompt_previous_text: bool = True
    # 용어 힌트를 쓰면 같은 말 반복("OnClick, OnClick, ...")이 생기고, faster-whisper 기본 온도 재시도
    # (0.0~1.0, 최대 6회)가 매번 최대 길이까지 다시 디코딩해 8초 구간에 47초가 걸렸다(실측).
    # 재시도 없이 온도 0만 쓰고 출력 길이를 제한한다. 인식한 텍스트는 고치지 않는다.
    temperatures: tuple = (0.0,)
    # 구간 1초당 최대 출력 토큰 수(+10). 데모 강의 말 속도(초당 약 4글자)보다 넉넉한 값.
    max_tokens_per_second: int = 6
    # 반복 n-gram 금지. 3으로 하면 정상적으로 다시 나온 말까지 바꿔 정확도가 떨어져(실측) 기본은 끈다(0).
    no_repeat_ngram_size: int = 0

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError('Whisper 모델 이름이 필요합니다')
        for name in ('sample_rate_hz', 'chunk_ms', 'window_seconds', 'num_threads',
                     'min_segment_seconds', 'pause_ms', 'beam_size', 'max_tokens_per_second'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name}는 양의 정수여야 합니다')
        if self.sample_rate_hz != 16000:
            raise ValueError('faster-whisper 입력은 16kHz가 필요합니다')
        if self.min_segment_seconds > self.window_seconds:
            raise ValueError('min_segment_seconds는 window_seconds 이하여야 합니다')
        for name in ('vad_filter', 'prompt_previous_text'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name}는 True/False여야 합니다')
        size = self.no_repeat_ngram_size
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError('no_repeat_ngram_size는 0 이상의 정수여야 합니다(0은 끔)')
        temps = self.temperatures
        if (not isinstance(temps, tuple) or not temps
                or any(isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t <= 1 for t in temps)):
            raise ValueError('temperatures는 0~1 사이 값의 비어 있지 않은 튜플이어야 합니다')

    def make_segmenter(self):
        return PauseSegmenter(self.sample_rate_hz, min_seconds=self.min_segment_seconds,
                              max_seconds=self.window_seconds, pause_ms=self.pause_ms)

    def transcribe_options(self, previous_text='', hotwords=None, audio_seconds=None):
        options = {'language': 'ko', 'beam_size': self.beam_size, 'vad_filter': self.vad_filter,
                   'word_timestamps': False, 'condition_on_previous_text': False,
                   'temperature': list(self.temperatures)}
        if self.no_repeat_ngram_size:
            options['no_repeat_ngram_size'] = self.no_repeat_ngram_size
        if audio_seconds is not None:
            options['max_new_tokens'] = int(audio_seconds * self.max_tokens_per_second) + 10
        if self.vad_filter:
            options['vad_parameters'] = {'min_silence_duration_ms': 500, 'speech_pad_ms': 300}
        if self.prompt_previous_text and previous_text:
            options['initial_prompt'] = previous_text[-120:]
        if hotwords:
            options['hotwords'] = hotwords
        return options

    def public_info(self):
        return {'model_id': f'faster-whisper-{self.model}', 'model_version': '1.2.1',
                'sample_rate_hz': self.sample_rate_hz, 'window_seconds': self.window_seconds,
                'segmentation': {'min_seconds': self.min_segment_seconds, 'pause_ms': self.pause_ms},
                'beam_size': self.beam_size, 'vad_filter': self.vad_filter,
                'prompt_previous_text': self.prompt_previous_text,
                'no_repeat_ngram_size': self.no_repeat_ngram_size, 'temperatures': list(self.temperatures)}


def load_whisper(config):
    from faster_whisper import WhisperModel
    return WhisperModel(config.model, device='cpu', compute_type='int8', cpu_threads=config.num_threads,
                        download_root=str(Path(__file__).resolve().parent / 'models'))


class WhisperLiveTranscription:
    """세션 하나의 전사를 별도 스레드에서 실행한다. 말 멈춤에서 자른 겹치지 않는 구간을 한 번씩 인식한다.

    - `source="loopback"`: `target_process_id`가 있으면 해당 프로세스 트리, 없으면 기본 출력 장치 소리
    - `source="microphone"`: 마이크 입력
    청크 시각은 영상 분석과 같은 `service.elapsed_ms(session_id)` 기준이다.
    사용 순서: `start()` → (수업 진행) → `stop()` → `service.stop_session()`.
    `stop()`이 남은 구간을 처리하므로 세션 종료보다 먼저 호출한다.

    말소리가 없는 구간은 인식하지 않고, 지어낸 문장으로 보이는 결과는 버린다(`segment_text`).
    `hint_terms`(강의 자료 용어 목록)를 주면 faster-whisper `hotwords`로 넣어 전문용어 인식을 돕는다.
    데모 영상 4분에서 핵심 용어 55회 중 맞힌 수가 39→52회로 늘었지만 처리 시간이 약 1.6배가 됐다.
    자료에 없는 말을 용어로 바꿔 적을 수 있다.
    """
    MAX_HINT_CHARS = 150

    def __init__(self, service, session_id, config, source='loopback', device=None, engine=None,
                 capture=None, target_process_id=None, hint_terms=None):
        if source not in SOURCES:
            raise ValueError(f'source는 {sorted(SOURCES)} 중 하나여야 합니다')
        self._service = service
        self._session_id = session_id
        self._config = config
        self.source = source
        self._device = device
        self._target_process_id = target_process_id
        self._engine = engine
        self._capture = capture
        self._thread = None
        self._stop = threading.Event()
        self._finished = False
        self._stats_lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._capture_stopped = False
        self._audio_ms = 0.0
        self._process_s = 0.0
        self._lags = []
        self._last_lag_ms = None
        self.last_error = None
        terms, length = [], 0
        for term in hint_terms or ():
            term = str(term).strip()
            if not term or term in terms:
                continue
            extra = len(term) + (2 if terms else 0)
            if length + extra > self.MAX_HINT_CHARS:
                break
            terms.append(term)
            length += extra
        self.hint_terms = tuple(terms)
        self._hotwords = ', '.join(terms) or None
        self._recent_levels = deque(maxlen=20)
        self._processed_windows = 0
        self._empty_windows = 0
        self._previous_text = ''

    # 상태 ---------------------------------------------------------------
    def _status(self, state, message, reasons=None):
        now = self._service.elapsed_ms(self._session_id)
        event = self._service.make_event(
            self._session_id, 'analysis_status', {'component': COMPONENT, 'state': state, 'message': message},
            now, now, 'speech', quality_status='not_observable' if state == 'unavailable' else 'valid',
            reason_codes=reasons)
        self._service.receive_event(event)

    def _safe_status(self, *args):
        try:
            self._status(*args)
        except Exception:
            pass

    def stats(self):
        """실시간성 계측(메모리 전용, 세션 JSON에 넣지 않음).

        - `rtf`: 인식 처리 시간 / 받은 오디오 길이. 1보다 작으면 실시간보다 빠르다.
        - `lag_ms_*`: 구간 끝 시각 → 해당 구간 인식 완료 시각(세션 시계).
        """
        with self._stats_lock:
            lags = sorted(self._lags)
            audio_ms, process_s = self._audio_ms, self._process_s
            peak = max(self._recent_levels) if self._recent_levels else None
            stats = {
                'source': self.source, 'audio_seconds': round(audio_ms / 1000, 2),
                'rtf': round(process_s * 1000 / audio_ms, 3) if audio_ms else None,
                'lag_ms_p50': lags[min(len(lags) - 1, int(len(lags) * 0.5))] if lags else None,
                'lag_ms_p95': lags[min(len(lags) - 1, int(len(lags) * 0.95))] if lags else None,
                'lag_ms_max': lags[-1] if lags else None,
                'lag_ms_latest': self._last_lag_ms,
                # 최근 약 2초 청크 RMS의 최대값. 말소리 여부 판정에는 쓰지 않는다.
                'capture_level_dbfs': (round(20 * float(np.log10(max(peak, 1e-10))), 1)
                                       if peak is not None else None),
                'processed_windows': self._processed_windows,
                'empty_windows': self._empty_windows,
                'hint_term_count': len(self.hint_terms),
            }
        capture = self._capture
        stats.update({
            'dropped_chunks': getattr(capture, 'dropped_chunks', 0),
            'input_overflows': getattr(capture, 'overflow_count', 0),
            'discontinuities': getattr(capture, 'discontinuity_count', 0),
            'queue_size': capture.chunks.qsize() if capture is not None else None,
        })
        return stats

    def is_alive(self):
        """전사 작업이 실제로 동작 중인지 관리자에게 알린다."""
        return self._thread is not None and self._thread.is_alive()

    # 실행 ---------------------------------------------------------------
    def _make_capture(self):
        if self.source == 'loopback':
            from .loopback import LoopbackCapture
            return LoopbackCapture(self._service, self._session_id, self._config.sample_rate_hz,
                                   self._config.chunk_ms, speaker=self._device,
                                   process_id=self._target_process_id)
        from .microphone import MicrophoneCapture
        return MicrophoneCapture(self._service, self._session_id, self._config.sample_rate_hz,
                                 self._config.chunk_ms, self._device)

    def _stop_capture(self):
        with self._capture_lock:
            if self._capture is not None and not self._capture_stopped:
                self._capture.stop()
                self._capture_stopped = True

    def start(self):
        """모델과 입력 장치를 준비한다. 실패하면 `unavailable` 상태를 알리고 False를 반환한다."""
        if self._thread is not None:
            raise RuntimeError('이미 시작했습니다')
        try:
            self._engine = self._engine or load_whisper(self._config)
            self._capture = self._capture or self._make_capture()
            self._capture.start()
        except Exception as error:  # 모델 없음, 장치 없음, 권한 거부, 드라이버 오류 등
            self.last_error = error
            try:
                self._stop_capture()
            except Exception:
                pass
            self._status('unavailable', f'전사를 시작하지 못했습니다: {error}', ['audio_unavailable'])
            return False
        label = '출력 음성(루프백)' if self.source == 'loopback' else '마이크'
        self._status('running', f'faster-whisper 말 멈춤 단위 전사를 시작했습니다. 입력: {label}')
        self._thread = threading.Thread(target=self._loop, name='whisper-stt', daemon=True)
        self._thread.start()
        return True

    def _transcribe(self, segment):
        """말소리 구간 하나를 인식해 확정 전사로 보낸다."""
        began = time.perf_counter()
        options = self._config.transcribe_options(self._previous_text, self._hotwords,
                                                  len(segment.audio) / self._config.sample_rate_hz)
        results, _ = self._engine.transcribe(segment.audio, **options)
        text = ' '.join(t for t in (segment_text(s, self.hint_terms) for s in results) if t).strip()
        if text:
            self._previous_text = text
        elapsed = time.perf_counter() - began
        lag = max(0, self._service.elapsed_ms(self._session_id) - segment.end_ms)
        if text:
            info = self._config.public_info()
            event = self._service.make_event(self._session_id, 'transcript_final',
                {'utterance_id': str(uuid4()), 'revision': 1, 'text': text, 'is_edited': False},
                segment.start_ms, max(segment.start_ms, segment.end_ms), 'speech',
                model_id=info['model_id'], model_version=info['model_version'])
            self._service.receive_event(event)
        with self._stats_lock:
            self._process_s += elapsed
            self._processed_windows += 1
            self._empty_windows += int(not text)
            self._lags.append(lag)
            self._last_lag_ms = lag
            if len(self._lags) > 36000:
                del self._lags[:6000]

    def _loop(self):
        chunks = self._capture.chunks
        segmenter = self._config.make_segmenter()
        last_id = None
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
                    for segment in segmenter.flush():
                        self._transcribe(segment)
                    self._status('running', '오디오 청크 누락으로 전사 구간을 나눴습니다.', ['audio_unavailable'])
                last_id = chunk.chunk_id
                if chunk.sample_rate_hz != self._config.sample_rate_hz:
                    raise ValueError('faster-whisper 입력은 16kHz가 필요합니다')
                audio = to_mono_float32(chunk)
                if not len(audio):
                    continue
                level = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
                with self._stats_lock:
                    self._recent_levels.append(level)
                    self._audio_ms += len(audio) * 1000 / self._config.sample_rate_hz
                for segment in segmenter.push(audio, chunk.start_ms, chunk.end_ms):
                    self._transcribe(segment)
            for segment in segmenter.flush():
                self._transcribe(segment)
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
