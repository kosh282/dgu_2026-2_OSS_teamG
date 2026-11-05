"""한 대시보드 세션이 소유하는 영상 분석 작업."""
import threading
from integration import IntegrationService
from video_recognition.integrated_analysis import VisionPipeline
from video_recognition.window_capture import WindowCapture, FrameNotAvailable
from video_recognition.face_analysis import FaceAnalyzer
from video_recognition.pose_analysis import PoseAnalyzer
from video_recognition.webcam_test import DeveloperPreviewOutput
from engagement_model.runtime import EngagementRuntime, MODEL_ID, MODEL_VERSION
from video_recognition.preview import PreviewCapture
from sherpa_stt.config import SttConfig
from sherpa_stt.engine import SherpaEngine
from sherpa_stt.runner import LiveTranscription

# STT 입력 선택지. None은 STT를 쓰지 않는다.
STT_SOURCES = {"loopback": "수업 음성(스피커 출력)", "microphone": "마이크", None: "끄기"}

_stt_engine_lock = threading.Lock()
_stt_engine = None


def _load_stt_engine(config):
    """STT 모델을 한 번만 불러와 이후 수업에서 재사용한다(약 5초)."""
    global _stt_engine
    with _stt_engine_lock:
        if _stt_engine is None:
            _stt_engine = SherpaEngine(config)
        return _stt_engine


# 별도 웹캠을 열지 않고, 실제 분석기에 전달한 프레임을 그대로 보여 주는 창 이름이다.
# OpenCV 창이므로 Streamlit 페이지 안에 삽입되지 않고 Windows의 별도 창으로 열린다.
class LiveSession:
    """선택한 창과 실제 수업 세션의 시작 시점을 분리해서 관리한다."""

    def __init__(self, hwnd, stt_source="loopback"):
        # 창 연결만으로는 분석기·카메라 캡처·세션 타이머를 시작하지 않는다.
        # 교수자가 '수업 시작'을 눌렀을 때에만 start()가 실제 세션을 만든다.
        if stt_source not in STT_SOURCES:
            raise ValueError(f"알 수 없는 STT 입력입니다: {stt_source}")
        self.stt_source = stt_source
        self.stt = None
        self.stt_error = None
        self.service = IntegrationService()
        self.sid = None
        self.hwnd = hwnd
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.model_runtime = EngagementRuntime()
        # 개발자 미리보기는 필요할 때만 창을 갱신한다. 분석·세션 저장 데이터에는 영향을 주지 않는다.
        self.developer_preview = False
        # 웹캠 데모와 동일한 출력 객체·창 이름·오버레이·키 조작을 사용한다.
        self.developer_output = DeveloperPreviewOutput()
        self.thread = None

    def start(self, title):
        """수업 제목을 확정하고 타이머·공통 세션·영상 분석을 같은 순간에 시작한다."""
        if not isinstance(title, str) or not title.strip():
            raise ValueError('수업 제목을 입력하세요.')
        with self.lock:
            if self.thread is not None:
                raise RuntimeError('이미 시작했거나 종료된 수업 연결입니다. 새 수업 연결을 사용하세요.')
            if self.stt_source not in STT_SOURCES:
                raise ValueError(f"알 수 없는 STT 입력입니다: {self.stt_source}")
            self.model_runtime.reset()
            # STT 모델은 세션 시작 전에 불러 로드 시간이 수업 시간축에 섞이지 않게 한다.
            stt_config = stt_engine = None
            if self.stt_source is not None:
                stt_config = SttConfig()
                try:
                    stt_engine = _load_stt_engine(stt_config)
                except Exception as exc:  # 모델·패키지 없음. 영상 분석은 계속한다.
                    self.stt_error = str(exc)
            self.sid = self.service.start_session(
                title.strip(),
                config={
                    "config_version": self.model_runtime.config.config_version,
                    "participation_model": {
                        "model_id": MODEL_ID,
                        "model_version": MODEL_VERSION,
                        "output_mode": "developer_preview_only",
                    },
                    "stt_model": (None if stt_engine is None
                                  else {**stt_config.public_info(), "source": self.stt_source}),
                },
            )
            if stt_config is not None:
                # 영상 분석과 같은 service·sid를 써서 전사 구간이 같은 수업 시간축에 놓인다.
                # 모델 로드에 실패했으면 start()가 analysis_status(stt: unavailable)를 남긴다.
                self.stt = LiveTranscription(self.service, self.sid, stt_config,
                                             source=self.stt_source, engine=stt_engine)
                if not self.stt.start():
                    self.stt_error = self.stt_error or str(self.stt.last_error)
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def has_started(self):
        with self.lock:
            return self.thread is not None

    def is_running(self):
        with self.lock:
            return self.thread is not None and self.thread.is_alive()

    def elapsed_seconds(self):
        """프런트 타이머와 JSON 세션 시간이 반드시 같은 IntegrationService 시계를 쓴다."""
        with self.lock:
            sid = self.sid
        return self.service.elapsed_ms(sid) / 1000 if sid is not None else 0

    def _run(self):
        face = pose = capture = pipeline = None
        try:
            # 좌표는 개발자 창을 위해서만 메모리에 유지하며 JSON 이벤트나 최종 세션에는 넣지 않는다.
            face = FaceAnalyzer(show_landmarks=True)
            pose = PoseAnalyzer(show_landmarks=True)
            capture = PreviewCapture(WindowCapture(self.hwnd))
            pipeline = VisionPipeline(
                self.service, self.sid, capture, face,
                model_consumer=self.model_runtime, pose_analyzer=pose,
            )
            while not self.stop_event.is_set():
                with self.lock:
                    preview_enabled = self.developer_preview
                # 개발자 미리보기가 꺼져 있으면 랜드마크 좌표 복사와 오버레이 준비를 생략한다.
                face.show_landmarks = preview_enabled and self.developer_output.show_landmarks
                pose.show_landmarks = preview_enabled and self.developer_output.show_landmarks
                try:
                    result = pipeline.process_one()
                except FrameNotAvailable:
                    continue
                if result is None:
                    continue
                with self.lock:
                    self.latest = result
                    preview_enabled = self.developer_preview
                if preview_enabled:
                    controls = self._render_developer_preview(capture.last_frame, result, face, pose)
                    if controls['calibrate']:
                        # 키 입력도 분석기와 같은 스레드에서 처리해 보정 상태를 안전하게 바꾼다.
                        pipeline.head_calibration.start()
                    if controls['quit']:
                        with self.lock:
                            self.developer_preview = False
                else:
                    self._close_developer_windows()
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            try:
                if pipeline:
                    pipeline.close()
                else:
                    for resource in (capture, pose, face):
                        if resource:
                            resource.close()
            finally:
                self._close_developer_windows()
                # 영상 분석만 실패했으면 STT와 세션 시계는 교수자가 수업을 종료할 때까지 유지한다
                # (계약 8절: 한 스트림이 중단돼도 다른 스트림은 가능한 한 계속 동작).
                if self.stt is not None and self.stt.is_running():
                    self.stop_event.wait()
                # 마지막 발화를 확정하려면 세션 종료보다 먼저 STT를 멈춘다.
                self._stop_stt()
                self.service.stop_session(self.sid)

    def _stop_stt(self):
        if self.stt is None:
            return
        try:
            self.stt.stop()
        except Exception as exc:
            with self.lock:
                self.stt_error = str(exc)
        if self.stt.last_error is not None:
            with self.lock:
                self.stt_error = self.stt_error or str(self.stt.last_error)

    def stt_info(self):
        """대시보드 표시용 STT 입력·오류·실시간 계측. 계측값은 세션 JSON에 넣지 않는다."""
        stats = self.stt.stats() if self.stt is not None else None
        with self.lock:
            return {"source": self.stt_source, "label": STT_SOURCES[self.stt_source],
                    "error": self.stt_error, "stats": stats}

    def stop(self):
        if not self.has_started():
            raise RuntimeError('아직 수업을 시작하지 않았습니다.')
        self.stop_event.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('영상 분석이 아직 종료되지 않았습니다.')

    def set_developer_preview(self, enabled):
        """개발자용 원본 프레임·랜드마크 창의 표시 여부를 바꾼다.

        이 함수는 Streamlit 버튼에서 호출된다. 실제 OpenCV 갱신은 분석 스레드에서만
        수행해, 동시에 같은 창을 조작하지 않도록 한다.
        """
        with self.lock:
            self.developer_preview = bool(enabled)

    def developer_preview_enabled(self):
        """대시보드가 안전하게 현재 개발자 창의 요청 상태를 확인한다."""
        with self.lock:
            return self.developer_preview

    def _close_developer_windows(self):
        """Close the shared developer preview windows."""
        self.developer_output.close()

    def _render_developer_preview(self, image, result, face, pose):
        """Render the same preview and feature panel as the webcam demo."""
        return self.developer_output.render(
            image, result, face, pose, self.model_runtime.latest,
        )

    def snapshot(self):
        with self.lock:
            return self.latest, self.error, self.thread is not None and self.thread.is_alive()
