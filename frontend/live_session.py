"""한 대시보드 세션이 소유하는 영상 분석 작업."""
import threading
import cv2
import numpy as np
from integration import IntegrationService
from video_recognition.integrated_analysis import VisionPipeline
from video_recognition.window_capture import WindowCapture, FrameNotAvailable
from video_recognition.face_analysis import FaceAnalyzer
from video_recognition.pose_analysis import PoseAnalyzer
from video_recognition.preview import PreviewCapture, draw_landmarks


# 별도 웹캠을 열지 않고, 실제 분석기에 전달한 프레임을 그대로 보여 주는 창 이름이다.
# OpenCV 창이므로 Streamlit 페이지 안에 삽입되지 않고 Windows의 별도 창으로 열린다.
DEBUG_CAPTURE_WINDOW = 'Developer preview - captured frame and landmarks'
DEBUG_INFO_WINDOW = 'Developer preview - MediaPipe features'


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
        # 개발자 미리보기는 필요할 때만 창을 갱신한다. 분석·세션 저장 데이터에는 영향을 주지 않는다.
        self.developer_preview = False
        self._show_landmarks = True
        # OpenCV 창은 한 번만 초기 크기를 정한다. 이후 사용자가 조절한 크기를 매 프레임 덮어쓰지 않는다.
        self._developer_windows_ready = False
        self.thread = None

    def start(self, title):
        """수업 제목을 확정하고 타이머·공통 세션·영상 분석을 같은 순간에 시작한다."""
        if not isinstance(title, str) or not title.strip():
            raise ValueError('수업 제목을 입력하세요.')
        with self.lock:
            if self.thread is not None:
                raise RuntimeError('이미 시작했거나 종료된 수업 연결입니다. 새 수업 연결을 사용하세요.')
            self.sid = self.service.start_session(title.strip())
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
            pipeline = VisionPipeline(self.service, self.sid, capture, face, pose_analyzer=pose)
            while not self.stop_event.is_set():
                try:
                    result = pipeline.process_one()
                except FrameNotAvailable:
                    continue
                with self.lock:
                    self.latest = result
                    preview_enabled = self.developer_preview
                    show_landmarks = self._show_landmarks
                if preview_enabled:
                    if self._render_developer_preview(capture.last_frame, result, face, pose, show_landmarks):
                        # 키 입력도 분석기와 같은 스레드에서 처리해 보정 상태를 안전하게 바꾼다.
                        pipeline.head_calibration.start()
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
        """창이 이미 닫힌 경우도 무시하고 두 개발자 창만 정리한다."""
        for name in (DEBUG_CAPTURE_WINDOW, DEBUG_INFO_WINDOW):
            try:
                cv2.destroyWindow(name)
            except cv2.error:
                pass
        self._developer_windows_ready = False

    @staticmethod
    def _feature_panel(result):
        """MediaPipe 원본이 아닌, 현재 파이프라인이 만든 특징값만 표시한다."""
        features = result['features']
        faces = features.get('faces', [])
        panel = np.zeros((440, 700, 3), dtype=np.uint8)
        lines = [
            'Developer view: the same frame sent to the analyzers',
            f"Capture: {result['start_ms']} ms | faces: {len(faces)} | bodies: {len(features.get('upper_bodies', []))}",
            f"Quality: {result['quality']['status']} | calibration: {features.get('head_calibration_status', 'waiting')}",
            'V: landmarks on/off   C: start 3-second front-facing calibration   Q: hide preview',
        ]
        for number, face in enumerate(faces[:2], start=1):
            angles = face.get('head_angles_degrees') or {}
            relative = face.get('head_angles_relative_degrees') or {}
            position = face.get('head_position') or {}
            lines.extend([
                f"Face {number}: eye openness = {face.get('eye_openness') if face.get('eye_openness') is not None else 'unavailable'}",
                (f"  raw pitch/yaw/roll: {angles.get('pitch', 0):.1f}, {angles.get('yaw', 0):.1f}, {angles.get('roll', 0):.1f} deg"
                 if angles else '  raw pitch/yaw/roll: unavailable'),
                (f"  relative pitch/yaw/roll: {relative.get('pitch', 0):.1f}, {relative.get('yaw', 0):.1f}, {relative.get('roll', 0):.1f} deg"
                 if relative else '  relative pitch/yaw/roll: not calibrated'),
                (f"  nose position: {position.get('x', 0):.3f}, {position.get('y', 0):.3f}"
                 if position else '  nose position: unavailable'),
            ])
        if not faces:
            lines.append('No face detected in the selected window.')
        if len(faces) > 2:
            lines.append(f'Additional faces: {len(faces) - 2}')
        for index, line in enumerate(lines):
            cv2.putText(panel, line, (14, 30 + index * 29), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (235, 235, 235), 1, cv2.LINE_AA)
        return panel

    def _render_developer_preview(self, image, result, face, pose, show_landmarks):
        """분석에 사용된 동일 이미지에만 점을 겹쳐 그린다. 원본을 저장·변경하지 않는다."""
        if image is None:
            return
        if not self._developer_windows_ready:
            # WINDOW_NORMAL은 창 테두리를 드래그해 확대·축소할 수 있게 한다.
            # 처음 열 때만 화면을 과도하게 가리지 않을 크기를 제안하고, 이후에는 사용자 크기를 보존한다.
            cv2.namedWindow(DEBUG_CAPTURE_WINDOW, cv2.WINDOW_NORMAL)
            cv2.namedWindow(DEBUG_INFO_WINDOW, cv2.WINDOW_NORMAL)
            height, width = image.shape[:2]
            scale = min(1.0, 720 / max(width, 1), 540 / max(height, 1))
            cv2.resizeWindow(DEBUG_CAPTURE_WINDOW, max(320, int(width * scale)), max(240, int(height * scale)))
            cv2.resizeWindow(DEBUG_INFO_WINDOW, 700, 440)
            self._developer_windows_ready = True
        frame = (draw_landmarks(image, face.last_landmarks, pose.last_landmarks)
                 if show_landmarks else image)
        cv2.imshow(DEBUG_CAPTURE_WINDOW, frame)
        cv2.imshow(DEBUG_INFO_WINDOW, self._feature_panel(result))
        key = cv2.waitKey(1) & 0xFF
        start_calibration = False
        with self.lock:
            if key in (ord('q'), ord('Q')):
                self.developer_preview = False
            elif key in (ord('v'), ord('V')):
                self._show_landmarks = not self._show_landmarks
            elif key in (ord('c'), ord('C')):
                # 개인 식별 없이, 현재 단일 얼굴이 이어진 3초만 임시 기준으로 사용한다.
                # 이 보정값은 메모리에만 존재하고 최종 결과나 세션 파일에는 저장되지 않는다.
                start_calibration = True
        return start_calibration

    def snapshot(self):
        with self.lock:
            return self.latest, self.error, self.thread is not None and self.thread.is_alive()
