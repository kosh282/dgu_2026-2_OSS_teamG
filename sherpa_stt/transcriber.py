"""AudioChunk를 받아 `transcript_partial`/`transcript_final` 이벤트를 발행한다.

이벤트는 `IntegrationService.make_event()`로 만들고 `receive_event()`로 전달한다.
시간은 청크의 수업 상대 시각(`start_ms`, `end_ms`) 기준이며, 영상 분석과 같은
세션 공통 시계를 쓴다.
"""
import threading
from uuid import uuid4

from .audio import to_mono_float32

COMPONENT = "stt"
# 토큰 타임스탬프는 음절 시작 근처에 찍히므로 앞뒤로 여유를 둔다(실험값).
START_PAD_MS = 200
END_PAD_MS = 400


class SherpaTranscriber:
    """한 세션의 오디오 스트림 하나를 전사한다. 스레드 안전.

    - 발화 구간: 엔진이 토큰 타임스탬프를 주면 첫 토큰 - 200ms ~ 마지막 토큰 + 400ms를
      세션 시각으로 변환하고, 현재 세그먼트(직전 확정 이후 첫 청크 시작 ~ 마지막 청크 끝)
      안으로 제한한다. 타임스탬프가 없으면 세그먼트 전체를 쓴다. 단어 정렬이 아닌 추정값이다.
    - `utterance_id`는 텍스트가 처음 인식될 때 만든다. 빈 구간은 이벤트를 남기지 않는다.
    - 청크 번호가 빠지면(캡처가 청크를 버림) 무음으로 채우지 않고 현재 발화를 확정한 뒤
      `analysis_status`(사유 `audio_unavailable`)로 알린다. 번호가 이어지는 청크는 시각이
      건너뛰어도 같은 오디오로 이어서 인식한다. `max_gap_ms`는 시각이 과거로 되돌아간
      청크를 거부하는 허용 폭으로 쓴다.
    - 세션 종료(`stop_session`) 전에 `finish()`를 호출해야 마지막 발화가 확정된다.
    """

    def __init__(self, engine, service, session_id, model_id, model_version, max_gap_ms=500, auto_gain=None):
        self._engine = engine
        # 작은 말소리를 인식기 입력 전에 키운다(`audio.AutoGain`). None이면 원본 그대로 넣는다.
        self._auto_gain = auto_gain
        self._service = service
        self._session_id = session_id
        self._model_id = model_id
        self._model_version = model_version
        self._max_gap_ms = max_gap_ms
        self._lock = threading.Lock()
        self._stream = engine.create_stream()
        self._last_chunk_id = None
        self._last_end_ms = None
        self._segment_start_ms = None
        # 엔진에 넣은 오디오 위치(ms) ↔ 세션 시각 대응표: (오디오 시작, 길이, 세션 시작, 세션 끝)
        self._fed_ms = 0.0
        self._audio_map = []
        self._utterance_id = None
        self._revision = 0
        self._text = ""
        self._span = None
        self._finished = False

    # 이벤트 발행 -----------------------------------------------------------
    def _emit(self, event_type, payload, start_ms, end_ms, quality_status="valid", reason_codes=None):
        event = self._service.make_event(
            self._session_id, event_type, payload, start_ms, end_ms, "speech",
            quality_status=quality_status, reason_codes=reason_codes,
            model_id=self._model_id, model_version=self._model_version,
        )
        self._service.receive_event(event)
        return event

    def report_status(self, state, message, reason_codes=None, quality_status=None, start_ms=None, end_ms=None):
        """STT 구성요소 상태를 `analysis_status`로 알린다. 기본은 현재 시점 이벤트."""
        if start_ms is None:
            start_ms = end_ms = self._service.elapsed_ms(self._session_id)
        if quality_status is None:
            quality_status = "not_observable" if state == "unavailable" else "valid"
        return self._emit("analysis_status", {"component": COMPONENT, "state": state, "message": message},
                          start_ms, end_ms, quality_status=quality_status, reason_codes=reason_codes)

    # 시간 변환 -------------------------------------------------------------
    def _audio_to_session_ms(self, audio_ms):
        for fed_start, duration, start_ms, end_ms in reversed(self._audio_map):
            if audio_ms >= fed_start:
                ratio = min(1.0, (audio_ms - fed_start) / duration) if duration else 0.0
                return start_ms + ratio * (end_ms - start_ms)
        return self._audio_map[0][2] if self._audio_map else self._segment_start_ms

    def _interval(self, end_ms):
        """현재 발화의 [start_ms, end_ms] 추정."""
        start, end = self._segment_start_ms, end_ms
        if self._span is not None and self._audio_map:
            first_s, last_s = self._span
            start = max(start, int(self._audio_to_session_ms(first_s * 1000) - START_PAD_MS))
            end = min(end, int(self._audio_to_session_ms(last_s * 1000) + END_PAD_MS))
        start = min(start, end_ms)
        return start, max(start, end)

    def _read_result(self):
        reader = getattr(self._engine, "get_result", None)
        if reader is not None:
            return reader(self._stream)
        return self._engine.get_text(self._stream), None

    # 내부 상태 -------------------------------------------------------------
    def _update_partial(self, end_ms):
        text, span = self._read_result()
        if not text or text == self._text:
            return []
        if self._utterance_id is None:
            self._utterance_id = str(uuid4())
        self._revision += 1
        self._text, self._span = text, span
        start, end = self._interval(end_ms)
        payload = {"utterance_id": self._utterance_id, "revision": self._revision, "text": text}
        return [self._emit("transcript_partial", payload, start, end)]

    def _finalize(self, end_ms):
        events = []
        if self._utterance_id is not None and self._text:
            start, end = self._interval(end_ms)
            payload = {"utterance_id": self._utterance_id, "revision": self._revision,
                       "text": self._text, "is_edited": False}
            events.append(self._emit("transcript_final", payload, start, end))
        self._engine.reset(self._stream)
        self._segment_start_ms = None
        self._audio_map = []
        self._utterance_id = None
        self._revision = 0
        self._text = ""
        self._span = None
        return events

    # 공개 API -------------------------------------------------------------
    def accept_chunk(self, chunk):
        """청크 하나를 처리하고 이번 호출에서 발행한 이벤트 목록을 반환한다."""
        with self._lock:
            if self._finished:
                raise RuntimeError("finish() 이후에는 청크를 받을 수 없습니다")
            if chunk.session_id != self._session_id:
                raise ValueError("청크의 session_id가 전사기 세션과 다릅니다")
            if self._last_chunk_id is not None and chunk.chunk_id <= self._last_chunk_id:
                raise ValueError("chunk_id는 증가해야 합니다")
            if self._last_end_ms is not None and chunk.start_ms < self._last_end_ms - self._max_gap_ms:
                raise ValueError("청크 시각이 이전 청크보다 과거입니다")
            samples = to_mono_float32(chunk)
            events = []
            # 청크 번호가 이어지면 오디오는 끊기지 않은 것이다. CPU 부하로 콜백이 늦게·몰려서 오면
            # 시각만 건너뛰므로, 시간 공백만으로 발화를 끊고 인식기를 초기화하지 않는다
            # (끊으면 "지하철에서 다리" / "를 벌리고"처럼 문장이 조각난다. 지연 시뮬레이션 실측).
            # 실제 누락은 캡처가 청크를 버려 번호가 빠질 때다.
            missing_ids = self._last_chunk_id is not None and chunk.chunk_id > self._last_chunk_id + 1
            if self._last_end_ms is not None and missing_ids:
                gap_start = self._last_end_ms
                events += self._finalize(gap_start)
                events.append(self.report_status(
                    "running", f"오디오 입력 누락: 시간 공백 {max(0, chunk.start_ms - gap_start)}ms, 청크 번호 누락 {missing_ids}",
                    reason_codes=["audio_unavailable"], quality_status="not_observable",
                    start_ms=gap_start, end_ms=max(gap_start, chunk.start_ms)))
            self._last_chunk_id = chunk.chunk_id
            if self._segment_start_ms is None:
                self._segment_start_ms = chunk.start_ms
            # 지터로 약간 겹친 청크도 구간 시각이 거꾸로 가지 않게 한다.
            start_ms = max(chunk.start_ms, self._last_end_ms or 0)
            end_ms = max(chunk.end_ms, start_ms, self._segment_start_ms)
            self._last_end_ms = end_ms
            if len(samples):
                if self._auto_gain is not None:
                    samples = self._auto_gain.process(samples)
                duration = len(samples) * 1000 / chunk.sample_rate_hz
                self._audio_map.append((self._fed_ms, duration, min(start_ms, end_ms), end_ms))
                self._fed_ms += duration
                self._engine.accept_waveform(self._stream, chunk.sample_rate_hz, samples)
                self._engine.decode(self._stream)
            events += self._update_partial(end_ms)
            if self._engine.is_endpoint(self._stream):
                events += self._finalize(end_ms)
            return events

    def level_info(self):
        """최근 말소리 크기(원본, dBFS)와 적용 이득(dB). 진단용이며 세션 JSON에 넣지 않는다."""
        gain = self._auto_gain
        if gain is None:
            return {"input_level_dbfs": None, "gain_db": None}
        return {"input_level_dbfs": gain.input_level_dbfs, "gain_db": round(gain.gain_db, 1)}

    def finish(self):
        """입력 종료를 알리고 남은 발화를 확정한다. 여러 번 호출해도 된다."""
        with self._lock:
            if self._finished:
                return []
            self._finished = True
            if self._segment_start_ms is None:
                return []
            self._engine.input_finished(self._stream)
            self._engine.decode(self._stream)
            events = self._update_partial(self._last_end_ms)
            return events + self._finalize(self._last_end_ms)
