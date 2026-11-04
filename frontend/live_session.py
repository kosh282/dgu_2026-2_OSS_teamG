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


# 별도 웹캠을 열지 않고, 실제 분석기에 전달한 프레임을 그대로 보여 주는 창 이름이다.
# OpenCV 창이므로 Streamlit 페이지 안에 삽입되지 않고 Windows의 별도 창으로 열린다.
class LiveSession:
    """선택한 창과 실제 수업 세션의 시작 시점을 분리해서 관리한다."""

    def __init__(self, hwnd):
        # 창 연결만으로는 분석기·카메라 캡처·세션 타이머를 시작하지 않는다.
        # 교수자가 '수업 시작'을 눌렀을 때에만 start()가 실제 세션을 만든다.
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
            self.model_runtime.reset()
            self.sid = self.service.start_session(
                title.strip(),
                config={
                    "config_version": self.model_runtime.config.config_version,
                    "participation_model": {
                        "model_id": MODEL_ID,
                        "model_version": MODEL_VERSION,
                        "output_mode": "developer_preview_only",
                    },
                    "stt_model": None,
                },
            )
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
                self.service.stop_session(self.sid)

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
