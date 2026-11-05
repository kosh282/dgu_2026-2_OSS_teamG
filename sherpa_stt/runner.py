"""오디오 입력 → sherpa-onnx → 통합 서비스로 이어지는 실시간 전사 실행기."""
import queue
import threading
import time

from .engine import SherpaEngine
from .transcriber import COMPONENT, SherpaTranscriber

SOURCES = {"loopback", "microphone"}


class LiveTranscription:
    """세션 하나의 실시간 전사를 별도 스레드에서 실행한다.

    - `source="loopback"`: 출력 장치로 재생되는 소리(수업 프로그램·영상 음성)
    - `source="microphone"`: 마이크 입력
    청크 시각은 영상 분석과 같은 `service.elapsed_ms(session_id)` 기준이다.

    사용 순서: `start()` → (수업 진행) → `stop()` → `service.stop_session()`.
    `stop()`이 마지막 발화를 확정하므로 세션 종료보다 먼저 호출한다.
    """

    def __init__(self, service, session_id, config, source="loopback", device=None, engine=None, capture=None):
        if source not in SOURCES:
            raise ValueError(f"source는 {sorted(SOURCES)} 중 하나여야 합니다")
        self._service = service
        self._session_id = session_id
        self._config = config
        self.source = source
        self._device = device
        self._engine = engine
        self._capture = capture
        self._transcriber = None
        self._thread = None
        self._stop = threading.Event()
        self._finished = False
        self._stats_lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._capture_stopped = False
        self._audio_ms = 0.0
        self._process_s = 0.0
        self._lags = []
        self.last_error = None

    # 상태 ---------------------------------------------------------------
    def _status(self, state, message, reasons=None):
        if self._transcriber is not None:
            return self._transcriber.report_status(state, message, reasons)
        now = self._service.elapsed_ms(self._session_id)
        event = self._service.make_event(
            self._session_id, "analysis_status", {"component": COMPONENT, "state": state, "message": message},
            now, now, "speech", quality_status="not_observable" if state == "unavailable" else "valid",
            reason_codes=reasons)
        self._service.receive_event(event)

    def stats(self):
        """실시간성 계측(메모리 전용, 세션 JSON에 넣지 않음).

        - `rtf`: 인식 처리 시간 / 처리한 오디오 길이. 1보다 작으면 실시간보다 빠르다.
        - `lag_ms_*`: 청크 끝 시각 → 해당 청크 처리 완료 시각(세션 시계).
        """
        with self._stats_lock:
            lags = sorted(self._lags)
            audio_ms, process_s = self._audio_ms, self._process_s
        def pct(p):
            return lags[min(len(lags) - 1, int(len(lags) * p))] if lags else None
        return {
            "source": self.source, "audio_seconds": round(audio_ms / 1000, 2),
            "rtf": round(process_s * 1000 / audio_ms, 3) if audio_ms else None,
            "lag_ms_p50": pct(0.5), "lag_ms_p95": pct(0.95), "lag_ms_max": lags[-1] if lags else None,
            "dropped_chunks": getattr(self._capture, "dropped_chunks", 0),
        }

    def is_alive(self):
        """전사 작업이 실제로 동작 중인지 관리자에게 알린다."""
        return self._thread is not None and self._thread.is_alive()

    # 실행 ---------------------------------------------------------------
    def _make_capture(self):
        if self.source == "loopback":
            from .loopback import LoopbackCapture
            return LoopbackCapture(self._service, self._session_id, self._config.sample_rate_hz,
                                   self._config.chunk_ms, speaker=self._device)
        from .microphone import MicrophoneCapture
        return MicrophoneCapture(self._service, self._session_id, self._config.sample_rate_hz,
                                 self._config.chunk_ms, self._device)

    def start(self):
        """모델과 입력 장치를 준비한다. 실패하면 `unavailable` 상태를 알리고 False를 반환한다."""
        if self._thread is not None:
            raise RuntimeError("이미 시작했습니다")
        try:
            engine = self._engine or SherpaEngine(self._config)
        except (ImportError, FileNotFoundError, RuntimeError, ValueError) as error:
            self.last_error = error
            self._status("unavailable", f"STT 모델을 불러오지 못했습니다: {error}", ["model_not_configured"])
            return False
        self._transcriber = SherpaTranscriber(engine, self._service, self._session_id, self._config.model_id,
                                              self._config.model_version, self._config.max_gap_ms)
        try:
            self._capture = self._capture or self._make_capture()
            self._capture.start()
        except Exception as error:  # 장치 없음, 권한 거부, 드라이버 오류 등
            self.last_error = error
            try:
                self._stop_capture()
            except Exception:
                pass
            self._capture = None
            self._status("unavailable", f"오디오 입력을 열지 못했습니다: {error}", ["audio_unavailable"])
            return False
        label = "출력 음성(루프백)" if self.source == "loopback" else "마이크"
        self._status("running", f"실시간 전사를 시작했습니다. 입력: {label}")
        self._thread = threading.Thread(target=self._loop, name="sherpa-stt", daemon=True)
        self._thread.start()
        return True

    def _loop(self):
        chunks = self._capture.chunks
        while not self._stop.is_set() or not chunks.empty():
            try:
                chunk = chunks.get(timeout=0.1)
            except queue.Empty:
                capture_error = getattr(self._capture, "error", None)
                if capture_error is not None and not self._stop.is_set():
                    self.last_error = capture_error
                    self._safe_status("unavailable", f"오디오 입력이 중단되었습니다: {capture_error}", ["audio_unavailable"])
                    self._stop_capture()
                    return
                continue
            started = time.perf_counter()
            try:
                self._transcriber.accept_chunk(chunk)
            except Exception as error:
                self.last_error = error
                self._safe_status("unavailable", f"전사 처리 중 오류: {error}", ["unknown"])
                self._stop_capture()
                return
            lag = self._service.elapsed_ms(self._session_id) - chunk.end_ms
            with self._stats_lock:
                self._process_s += time.perf_counter() - started
                self._audio_ms += chunk.end_ms - chunk.start_ms
                self._lags.append(lag)
                if len(self._lags) > 36000:  # 약 1시간 분량만 유지
                    del self._lags[:6000]

    def _safe_status(self, *args):
        try:
            self._status(*args)
        except Exception:
            pass

    def _stop_capture(self):
        with self._capture_lock:
            if self._capture is not None and not self._capture_stopped:
                self._capture.stop()
                self._capture_stopped = True

    def stop(self):
        """입력을 멈추고 남은 청크와 마지막 발화를 처리한다. 여러 번 호출해도 된다."""
        self._stop_capture()
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._transcriber is not None and self.last_error is None and not self._finished:
            self._finished = True
            try:
                self._transcriber.finish()
                self._status("stopped", "실시간 전사를 종료했습니다.")
            except Exception as error:  # 세션이 먼저 종료된 경우 등
                self.last_error = error
