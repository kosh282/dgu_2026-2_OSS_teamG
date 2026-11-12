"""한 대시보드 세션이 소유하는 영상 분석 작업."""
import threading
import time
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
from integration import IntegrationService
from engagement_model.gaussian import GaussianRuntime
from engagement_model.config import ModelConfig
from engagement_model.multi_face import MultiFaceGaussianRuntime
from engagement_model.window_publisher import WindowPublisher
from video_recognition.integrated_analysis import VisionPipeline
from video_recognition.window_capture import WindowCapture, FrameNotAvailable
from video_recognition.face_analysis import FaceAnalyzer
from video_recognition.pose_analysis import PoseAnalyzer
from video_recognition.face_reidentification import FaceReidentifier
from video_recognition.preview import (
    PreviewCapture, LandmarkOverlaySmoother, draw_landmarks, draw_face_postprocessing,
)


# 별도 웹캠을 열지 않고, 실제 분석기에 전달한 프레임을 그대로 보여 주는 창 이름이다.
# OpenCV 창이므로 Streamlit 페이지 안에 삽입되지 않고 Windows의 별도 창으로 열린다.
DEBUG_CAPTURE_WINDOW = 'Developer preview - captured frame and landmarks'
DEBUG_INFO_WINDOW = 'Developer preview - MediaPipe features'
RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0)
CHECKPOINT_INTERVAL_SECONDS = 15.0

_stt_engine_lock = threading.Lock()
_stt_engines = {}


def _load_stt_engine(config):
    """STT 모델은 설정별로 한 번만 불러와 이후 수업과 재시작에서 재사용한다(첫 로드 약 5초).

    인식기 상태는 스트림마다 따로 있으므로 엔진 하나를 여러 세션이 순서대로 써도 된다.
    """
    with _stt_engine_lock:
        engine = _stt_engines.get(config)
        if engine is None:
            from sherpa_stt.engine import SherpaEngine
            engine = _stt_engines[config] = SherpaEngine(config)
        return engine


class LiveSession:
    """선택한 창과 실제 수업 세션의 시작 시점을 분리해서 관리한다."""

    def __init__(self, hwnd, audio_source=None, *, video_enabled=True, model_enabled=True,
                 reidentification_enabled=False, recovery_root=None):
        # 창 연결만으로는 분석기·카메라 캡처·세션 타이머를 시작하지 않는다.
        # 교수자가 '수업 시작'을 눌렀을 때에만 start()가 실제 세션을 만든다.
        self.service = IntegrationService()
        self.sid = None
        self.hwnd = hwnd
        self.video_enabled = bool(video_enabled)
        self.model_enabled = bool(model_enabled and video_enabled)
        self.reidentification_enabled = bool(reidentification_enabled and video_enabled)
        self.reidentifier = None
        self.reidentification_error = None
        self._participant_rows = []
        self._active_capture = self._active_face = None
        if self.video_enabled and hwnd is None:
            raise ValueError('영상 분석을 사용하려면 수업 창을 선택하세요.')
        self.audio_source = audio_source
        self.transcription = None
        self._stt_engine = None
        self._stt_startup_error = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.model_runtime = None
        self.window_publisher = None
        self.model_latest = {}
        self.error = None
        # 개발자 미리보기는 필요할 때만 창을 갱신한다. 분석·세션 저장 데이터에는 영향을 주지 않는다.
        self.developer_preview = False
        self._show_landmarks = True
        self._show_tessellation = False
        self._debug_face_page = 0
        self._landmark_overlay_smoother = LandmarkOverlaySmoother()
        # OpenCV 창은 한 번만 초기 크기를 정한다. 이후 사용자가 조절한 크기를 매 프레임 덮어쓰지 않는다.
        self._developer_windows_ready = False
        self._developer_info_shape = None
        self.thread = None
        self.vision_thread = None
        self._window_process_id = None
        self.stt_thread = None
        self.recovery_root = Path(recovery_root) if recovery_root is not None else None
        self.checkpoint_error = None
        self.checkpoint_thread = None
        self._model_failed = False
        self._model_retry_count = 0
        self._model_retry_at = None
        self._calibration_request = threading.Event()
        self._calibration_blocked = False
        self._calibration_window_start = None
        self.performance_monitor = None

    def start(self, title):
        """수업 제목을 확정하고 타이머·공통 세션·영상 분석을 같은 순간에 시작한다."""
        if not isinstance(title, str) or not title.strip():
            raise ValueError('수업 제목을 입력하세요.')
        with self.lock:
            if self.thread is not None:
                raise RuntimeError('이미 시작했거나 종료된 수업 연결입니다. 새 수업 연결을 사용하세요.')
        stt_config = None
        self._stt_engine = None
        if self.audio_source is not None:
            try:
                from sherpa_stt.config import SttConfig
                stt_config = SttConfig()
                # 수업 시계가 시작되기 전에 모델을 불러, 로드 시간 동안 수업 앞부분 음성을 놓치지 않게 한다.
                # 실패하면 기존처럼 STT 작업이 스스로 로드를 시도하고 unavailable/재시작을 알린다.
                self._stt_engine = _load_stt_engine(stt_config)
            except Exception as exc:
                self._stt_startup_error = exc
        with self.lock:
            if self.thread is not None:
                raise RuntimeError('이미 시작했거나 종료된 수업 연결입니다. 새 수업 연결을 사용하세요.')
            self.sid = self.service.start_session(title.strip(), config={
                'config_version': ModelConfig.gaussian().config_version,
                'participation_model': self._model_identity() if self.model_enabled else None,
                # source: 루프백(스피커 출력) 또는 마이크. 장치 이름·경로는 넣지 않는다.
                'stt_model': {**stt_config.public_info(), 'source': self.audio_source} if stt_config else None,
            })
            self.stop_event.clear()
        if not self.video_enabled:
            self._component_status('vision', 'stopped', '영상 분석을 사용하지 않습니다.', 'vision')
        if not self.model_enabled:
            self._model_status('stopped', '실험 점수 모델을 사용하지 않습니다.')
        if self.audio_source is None:
            self._component_status('stt', 'stopped', '음성 전사를 사용하지 않습니다.', 'speech')
        self._stt_config = stt_config
        with self.lock:
            self.thread = threading.Thread(target=self._run, name='session-coordinator', daemon=True)
            self.thread.start()

    def _report_stt_unavailable(self, message, reason):
        event = self.service.make_event(
            self.sid, 'analysis_status',
            {'component': 'stt', 'state': 'unavailable', 'message': message},
            self.service.elapsed_ms(self.sid), self.service.elapsed_ms(self.sid), 'speech', quality_status='not_observable', reason_codes=[reason],
        )
        self.service.receive_event(event)

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
        """수업 생명주기는 각 분석 작업의 성공 여부와 별도로 유지한다."""
        if self.video_enabled:
            self.vision_thread = threading.Thread(target=self._supervise_vision, name='vision-analysis', daemon=True)
            self.vision_thread.start()
        if self.audio_source is not None:
            self.stt_thread = threading.Thread(target=self._supervise_stt, name='stt-supervisor', daemon=True)
            self.stt_thread.start()
        if self.recovery_root is not None:
            self.checkpoint_thread = threading.Thread(target=self._checkpoint_loop, name='session-checkpoint', daemon=True)
            self.checkpoint_thread.start()
        self.stop_event.wait()
        if self.vision_thread is not None:
            self.vision_thread.join()
        if self.stt_thread is not None:
            self.stt_thread.join()
        if self.checkpoint_thread is not None:
            self.checkpoint_thread.join()
        if self.performance_monitor is not None:
            self.performance_monitor.stop()
        self.service.stop_session(self.sid)

    def _checkpoint_loop(self):
        """진행 중 세션을 복구 전용 경로에 저장한다. 원본 미디어는 포함하지 않는다."""
        while not self.stop_event.is_set():
            try:
                self.service.save_session(self.sid, self.recovery_root / f'{self.sid}.json')
                with self.lock:
                    self.checkpoint_error = None
            except Exception as exc:
                with self.lock:
                    self.checkpoint_error = str(exc)
            if self.stop_event.wait(CHECKPOINT_INTERVAL_SECONDS):
                break

    def checkpoint_status(self):
        with self.lock:
            return self.checkpoint_error

    def diagnostics(self):
        """수집량과 지연은 개인 데이터 없이 현재 프로세스에서만 확인한다."""
        with self.lock:
            runner, latest = self.transcription, self.latest
        stt = runner.stats() if runner is not None else None
        age = max(0, self.service.elapsed_ms(self.sid) - latest['start_ms']) if latest else None
        return {'stt': stt, 'video_result_age_ms': age}

    def performance_snapshot(self):
        """로그용 허용 필드만 추출한다. 영상·전사·점수·식별자는 기록하지 않는다."""
        snapshot = self.service.get_dashboard_snapshot(self.sid)
        now = self.service.elapsed_ms(self.sid)
        with self.lock:
            latest, preview = self.latest, self.developer_preview
        features = latest.get('features', {}) if latest else {}
        sampling = features.get('sampling', {})
        stats = self.diagnostics()['stt'] or {}
        def event_age(events):
            created = [e.get('created_ms') for e in events if isinstance(e.get('created_ms'), int)]
            return max(0, now - max(created)) if created else None
        return {
            'session_elapsed_ms': now,
            'enabled': {'video': self.video_enabled, 'model': self.model_enabled,
                        'stt': self.audio_source is not None, 'preview': preview},
            'components': {key: snapshot['status'].get(key) for key in ('vision', 'model', 'stt', 'assistant')},
            'video': {'result_age_ms': max(0, now-latest['start_ms']) if latest else None,
                      'face_hz': sampling.get('actual_face_hz'), 'pose_hz': sampling.get('actual_pose_hz'),
                      'source_size': sampling.get('source_size'), 'analysis_size': sampling.get('analysis_size')},
            'stt': {key: stats.get(key) for key in ('source', 'rtf', 'lag_ms_latest', 'lag_ms_p50',
                'lag_ms_p95', 'lag_ms_max', 'queue_size', 'dropped_chunks', 'input_overflows', 'discontinuities')},
            'transcript': {'final_count': len(snapshot['transcripts']),
                           'partial_count': len(snapshot['partial_transcripts']),
                           'last_final_event_age_ms': event_age(snapshot['transcripts']),
                           'last_partial_event_age_ms': event_age(snapshot['partial_transcripts'])},
        }

    def start_performance_monitor(self, directory):
        from .performance_monitor import PerformanceMonitor
        if not self.is_running():
            raise RuntimeError('진행 중인 수업에서만 로그를 시작할 수 있습니다.')
        if self.performance_monitor is not None and self.performance_monitor.is_running():
            return self.performance_monitor.path
        monitor = PerformanceMonitor(self, directory)
        monitor.start()
        self.performance_monitor = monitor
        return monitor.path

    def stop_performance_monitor(self):
        if self.performance_monitor is not None:
            self.performance_monitor.stop()

    def _supervise_vision(self):
        for attempt in range(len(RETRY_DELAYS_SECONDS) + 1):
            with self.lock:
                self.error = None
            self._model_failed = False
            self._model_retry_count = 0
            self._model_retry_at = None
            try:
                self._run_vision()
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
            if self.stop_event.is_set():
                return
            if self.error is None:
                with self.lock:
                    self.error = '영상 분석이 예상과 달리 종료되었습니다.'
            if attempt == len(RETRY_DELAYS_SECONDS):
                self._component_status('vision', 'unavailable', f'재시도 횟수 초과: {self.error}', 'vision', ['retry_exhausted'])
                return
            delay = RETRY_DELAYS_SECONDS[attempt]
            self._component_status('vision', 'unavailable', f'{delay:g}초 후 재시작 ({attempt + 1}/{len(RETRY_DELAYS_SECONDS)}): {self.error}', 'vision', ['retrying'])
            if self.stop_event.wait(delay):
                return

    def _supervise_stt(self):
        for attempt in range(len(RETRY_DELAYS_SECONDS) + 1):
            runner = None
            try:
                if self._stt_config is None:
                    from sherpa_stt.config import SttConfig
                    self._stt_config = SttConfig()
                from sherpa_stt.runner import LiveTranscription
                if self._stt_engine is None:
                    # 시작 전 로드에 실패했으면 재시도마다 다시 불러 본다.
                    try:
                        self._stt_engine = _load_stt_engine(self._stt_config)
                    except Exception:
                        pass  # 아래 runner.start()가 같은 오류를 unavailable로 알린다.
                runner = LiveTranscription(self.service, self.sid, self._stt_config,
                                           source=self.audio_source, engine=self._stt_engine)
                with self.lock:
                    self.transcription = runner
                if not runner.start():
                    raise RuntimeError(str(runner.last_error or '전사 시작 실패'))
                self._stt_startup_error = None
                while not self.stop_event.wait(0.2):
                    if not runner.is_alive():
                        raise RuntimeError(str(runner.last_error or '전사 작업이 종료되었습니다'))
                return
            except Exception as exc:
                self._stt_startup_error = exc
                if runner is not None and runner.last_error is None:
                    runner.last_error = exc
            finally:
                if runner is not None:
                    try:
                        runner.stop()
                    except Exception as exc:
                        self._stt_startup_error = exc
                if not self.stop_event.is_set():
                    self.service.clear_partial_transcripts(self.sid)
            if self.stop_event.is_set():
                return
            if attempt == len(RETRY_DELAYS_SECONDS):
                self._report_stt_unavailable(f'STT 재시도 횟수 초과: {self._stt_startup_error}', 'retry_exhausted')
                return
            delay = RETRY_DELAYS_SECONDS[attempt]
            self._report_stt_unavailable(f'STT {delay:g}초 후 재시작 ({attempt + 1}/{len(RETRY_DELAYS_SECONDS)}): {self._stt_startup_error}', 'retrying')
            if self.stop_event.wait(delay):
                return

    def _run_vision(self):
        face = pose = window = capture = pipeline = None
        try:
            # 좌표는 개발자 창을 위해서만 메모리에 유지하며 JSON 이벤트나 최종 세션에는 넣지 않는다.
            face = FaceAnalyzer(show_landmarks=True)
            pose = PoseAnalyzer(show_landmarks=True)
            window = WindowCapture(self.hwnd, expected_process_id=self._window_process_id)
            if self._window_process_id is None:
                self._window_process_id = getattr(window, 'process_id', None)
            capture = PreviewCapture(window, keep_frame=False)
            self._active_capture, self._active_face = capture, face
            if self.reidentification_enabled:
                try:
                    self.reidentifier = FaceReidentifier()
                    self.reidentification_error = None
                except Exception as exc:
                    self.reidentification_error = f'재식별 모델 준비 실패: {exc}'
            if self.model_enabled:
                self._start_model()
            pipeline = VisionPipeline(
                self.service, self.sid, capture, face,
                model_consumer=(self._consume_frame if self.reidentification_enabled else self._consume_model)
                    if self.model_enabled or self.reidentification_enabled else None, pose_analyzer=pose,
            )
            if self.model_enabled and not self.reidentification_enabled:
                # 초기 기준은 첫 유효 얼굴을 기다려 수집한다. 정면 여부 자체는 확인할 수 없다.
                pipeline.head_calibration.start()
                if not self._model_failed:
                    self._calibration_blocked = True
                    self._calibration_window_start = None
                    self._model_status('unavailable', '정면 기준을 자동 수집합니다.', ['baseline_unavailable'])
            while not self.stop_event.is_set():
                self._maybe_restart_model()
                if self._calibration_request.is_set():
                    self._calibration_request.clear()
                    self._recalibrate_pipeline(pipeline)
                with self.lock:
                    preview_enabled = self.developer_preview
                    show_landmarks = self._show_landmarks
                # 개발자 미리보기가 꺼져 있으면 랜드마크 좌표 복사와 오버레이 준비를 생략한다.
                capture.keep_frame = preview_enabled or self.reidentification_enabled
                face.show_landmarks = preview_enabled and show_landmarks
                pose.show_landmarks = preview_enabled and show_landmarks
                try:
                    result = pipeline.process_one()
                except FrameNotAvailable:
                    with self.lock:
                        for row in self._participant_rows:
                            row.update(status='not_observable', current_direction_score=None, score=None,
                                       reason_codes=['capture_unavailable'])
                    self._advance_model()
                    continue
                if result is None:
                    self._advance_model()
                    continue
                with self.lock:
                    self.latest = result
                    if self.model_runtime is not None:
                        self.model_latest = self.model_runtime.latest
                    preview_enabled = self.developer_preview
                    show_landmarks = self._show_landmarks
                if preview_enabled and capture.last_frame is not None:
                    if self._render_developer_preview(capture.last_frame, result, face, pose, show_landmarks):
                        # 키 입력도 분석기와 같은 스레드에서 처리해 보정 상태를 안전하게 바꾼다.
                        self._recalibrate_pipeline(pipeline)
                else:
                    self._close_developer_windows()
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
            if self.model_enabled and not self._model_failed:
                self._model_status('unavailable', f'영상 입력이 중단되었습니다: {exc}', ['analysis_failed'])
        finally:
            try:
                if pipeline:
                    pipeline.close()
                else:
                    for resource in (capture or window, pose, face):
                        if resource:
                            resource.close()
            except Exception as exc:
                with self.lock:
                    if self.error is None:
                        self.error = f'영상 자원 종료 실패: {exc}'
            finally:
                self._close_developer_windows()
                self._landmark_overlay_smoother.reset()
                try:
                    if self.window_publisher is not None and not self._model_failed:
                        self.window_publisher.finish(self.service.elapsed_ms(self.sid))
                    if self.model_enabled and self.error is None and not self._model_failed:
                        self._model_status('stopped', '실험 모델 분석을 종료했습니다.')
                except Exception as exc:
                    with self.lock:
                        self.error = f'마지막 모델 구간 저장 실패: {exc}'
                if self.error is not None:
                    self._component_status('vision', 'unavailable', f'영상 분석 오류: {self.error}', 'vision', ['analysis_failed'])
                try:
                    if self.reidentifier is not None:
                        self.reidentifier.close()
                finally:
                    self.reidentifier = None
                    if self.reidentification_enabled and self.model_runtime is not None:
                        self.model_runtime.reset()
                    with self.lock:
                        self._participant_rows = []
                        if self.reidentification_enabled:
                            self.model_latest = {}
                    self._active_capture = self._active_face = None

    def _model_identity(self):
        return ({'model_id': 'multi-face-gaussian', 'model_version': 'multi-gaussian-0.1'}
                if self.reidentification_enabled else
                {'model_id': 'gaussian-head-direction', 'model_version': 'angle-time-0.3'})

    def _make_model(self):
        return MultiFaceGaussianRuntime() if self.reidentification_enabled else GaussianRuntime()

    def _consume_frame(self, feature):
        """같은 분석 프레임에서 ID를 갱신한 뒤 점수화한다. ID는 특징에 넣지 않는다."""
        identities = {}
        if self.reidentifier is not None:
            try:
                identities = self.reidentifier.identify(
                    self._active_capture.last_frame, feature['start_ms'], feature['features']['faces'],
                    boxes=self._active_face.last_face_boxes)
            except Exception as exc:
                self.reidentification_error = f'재식별 처리 실패: {exc}'
                self.reidentifier.close()
                self.reidentifier = None
        if self.model_enabled and self.model_runtime is not None:
            self.model_runtime.update_identities(identities)
            self._consume_model(feature)
            if not self._model_failed:
                with self.lock:
                    self._participant_rows = self.model_runtime.participants()
                    self.model_latest = dict(self.model_runtime.latest)
            if self.reidentification_error:
                self._model_status('unavailable', self.reidentification_error, ['reidentification_unavailable'])

    def _recalibrate_pipeline(self, pipeline):
        if self.reidentification_enabled and self.model_runtime is not None:
            now = self.service.elapsed_ms(self.sid)
            if self.window_publisher is not None:
                self.window_publisher.finish(now)
            self.model_runtime.recalibrate()
            self.window_publisher = WindowPublisher(self.service, self.sid, self.model_runtime, start_ms=now)
            self._calibration_blocked = True
            self._calibration_window_start = now
            with self.lock:
                self._participant_rows = self.model_runtime.participants()
        else:
            pipeline.head_calibration.start()

    def _consume_model(self, feature):
        if self._model_failed:
            return
        try:
            if self.reidentification_enabled:
                self.window_publisher(feature)
                self._finish_calibration_window()
                return
            calibration = feature.get('features', {}).get('head_calibration_status')
            if calibration != 'ready' and not self._calibration_blocked:
                self._calibration_blocked = True
                self._calibration_window_start = None
                self._model_status('unavailable', '정면 기준 수집 또는 재보정을 기다립니다.', ['baseline_unavailable'])
            elif calibration != 'ready':
                self._calibration_window_start = None
            elif calibration == 'ready' and self._calibration_blocked and self._calibration_window_start is None:
                # 이전 기준으로 수집한 점수와 새로운 기준의 점수가 한 구간에서 섞이지 않게 한다.
                self.model_runtime = self._make_model()
                self._calibration_window_start = feature['start_ms']
                self.window_publisher = WindowPublisher(self.service, self.sid, self.model_runtime,
                                                        start_ms=self._calibration_window_start)
            self.window_publisher(feature)
            self._finish_calibration_window()
        except Exception as exc:
            self._fail_model(exc)

    def _finish_calibration_window(self):
        if (self._calibration_blocked and self._calibration_window_start is not None
                and self.window_publisher._last_end > self._calibration_window_start):
            self._calibration_blocked = False
            self._calibration_window_start = None
            self._model_status('running', '정면 기준을 적용한 실험 구간 집계를 시작했습니다.')

    def _advance_model(self):
        if self.window_publisher is None or self._model_failed:
            return
        try:
            self.window_publisher.advance(self.service.elapsed_ms(self.sid))
        except Exception as exc:
            self._fail_model(exc)

    def _fail_model(self, error):
        self._model_failed = True
        with self.lock:
            self._participant_rows = []
            self.model_latest = {}
        if self._model_retry_count < len(RETRY_DELAYS_SECONDS):
            delay = RETRY_DELAYS_SECONDS[self._model_retry_count]
            self._model_retry_count += 1
            self._model_retry_at = time.monotonic() + delay
            message = f'모델 {delay:g}초 후 재시작 ({self._model_retry_count}/{len(RETRY_DELAYS_SECONDS)}): {error}'
            reason = 'retrying'
        else:
            self._model_retry_at = None
            message = f'모델 재시도 횟수 초과: {error}'
            reason = 'retry_exhausted'
        self._model_status('unavailable', message, [reason])

    def _start_model(self):
        try:
            self.model_runtime = self._make_model()
            self.window_publisher = WindowPublisher(
                self.service, self.sid, self.model_runtime,
                start_ms=self.service.elapsed_ms(self.sid),
            )
            self._model_failed = False
            self._model_retry_at = None
            self._calibration_blocked = False
            self._calibration_window_start = None
            self._model_status('running', 'ID별 Gaussian 실험 모델이 실행 중입니다.'
                               if self.reidentification_enabled else '단일 얼굴 Gaussian 실험 모델이 실행 중입니다.')
        except Exception as exc:
            self._fail_model(exc)

    def _maybe_restart_model(self):
        if (self._model_failed and self._model_retry_at is not None
                and time.monotonic() >= self._model_retry_at and not self.stop_event.is_set()):
            self._start_model()

    def _component_status(self, component, state, message, module, reasons=None):
        now = self.service.elapsed_ms(self.sid)
        event = self.service.make_event(
            self.sid, 'analysis_status', {'component': component, 'state': state, 'message': message},
            now, now, module, quality_status='not_observable' if state == 'unavailable' else 'valid',
            reason_codes=reasons,
        )
        self.service.receive_event(event)

    def _model_status(self, state, message, reasons=None):
        now = self.service.elapsed_ms(self.sid)
        event = self.service.make_event(
            self.sid, 'analysis_status', {'component': 'model', 'state': state, 'message': message},
            now, now, 'model', quality_status='not_observable' if state == 'unavailable' else 'valid',
            reason_codes=reasons, **self._model_identity(),
        )
        self.service.receive_event(event)

    def stop(self):
        if not self.has_started():
            raise RuntimeError('아직 수업을 시작하지 않았습니다.')
        self.stop_event.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError('분석 작업을 종료하는 중입니다. 잠시 후 다시 시도하세요.')

    def request_head_calibration(self):
        """UI에서는 요청만 전달하고 실제 보정 객체는 영상 스레드에서 변경한다."""
        if not self.model_enabled or not self.is_running():
            return False
        with self.lock:
            self.latest = None
            self._calibration_request.set()
        self._model_status('unavailable', '정면 기준 재보정 요청을 처리합니다.', ['baseline_unavailable'])
        return True

    def calibration_state(self):
        with self.lock:
            if self._calibration_request.is_set():
                return {'status': 'collecting', 'last_reason_code': None, 'progress': 0.0}
            if not self.latest:
                return {'status': 'starting', 'last_reason_code': None, 'progress': 0.0}
            if self.reidentification_enabled:
                rows = self._participant_rows
                ready = sum(row.get('calibration_status') == 'ready' for row in rows
                            if row.get('status') not in {'missing', 'not_observable', 'unavailable'})
                return {'status': 'ready' if ready else 'collecting', 'last_reason_code': None,
                        'progress': ready / len(rows) if rows else 0.0}
            return dict(self.latest.get('features', {}).get('head_calibration_diagnostics') or {})

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
        self._developer_info_shape = None

    @staticmethod
    def _feature_panel(result, face_page=0):
        """MediaPipe 원본이 아닌, 현재 파이프라인이 만든 특징값만 표시한다."""
        features = result['features']
        faces = features.get('faces', [])
        page_size = 2
        page_count = max(1, (len(faces) + page_size - 1) // page_size)
        face_page = min(max(0, face_page), page_count - 1)
        start = face_page * page_size
        shown_faces = faces[start:start + page_size]
        sampling = features.get('sampling', {})
        source_size = sampling.get('source_size', {})
        analysis_size = sampling.get('analysis_size', {})
        lines = [
            'Developer view: the same frame sent to the analyzers',
            f"Capture: {result['start_ms']} ms | faces: {len(faces)} | bodies: {len(features.get('upper_bodies', []))}",
            f"Quality: {result['quality']['status']} | calibration: {features.get('head_calibration_status', 'waiting')}",
            f"Postprocess: {features.get('face_postprocessing_version', 'unavailable')} | temporary position-only tracking",
            (f"Analysis Hz: face {sampling.get('target_face_hz', 0):.0f}/"
             f"{sampling.get('actual_face_hz') or 0:.1f}, pose "
             f"{sampling.get('target_pose_hz') or 0:.0f}/{sampling.get('actual_pose_hz') or 0:.1f}"),
            (f"Frame: {source_size.get('width', 0)}x{source_size.get('height', 0)} -> "
             f"{analysis_size.get('width', 0)}x{analysis_size.get('height', 0)} | "
             f"pose age {sampling.get('pose_observation_age_ms') if sampling.get('pose_observation_age_ms') is not None else 'n/a'} ms"),
            f'Faces {start + 1}-{min(start + page_size, len(faces))} / {len(faces)} | page {face_page + 1}/{page_count}',
            'V: contour/pose on-off  M: dense mesh on-off  [ ]: face page  C: calibrate  Q: hide',
        ]
        for number, face in enumerate(shown_faces, start=1):
            angles = face.get('head_angles_degrees') or {}
            smoothed = face.get('head_angles_smoothed_degrees') or {}
            relative = face.get('head_angles_relative_degrees') or {}
            position = face.get('head_position') or {}
            left_ear = face.get('eye_aspect_ratio_left')
            right_ear = face.get('eye_aspect_ratio_right')
            blink = face.get('blink_metrics') or {}
            movement = face.get('head_angle_change_candidate') or {}
            track_key = face.get('transient_track_key', '?')
            tracking_quality = face.get('tracking_quality', 'unavailable')
            blink_state = (f"both={blink.get('closed', False)} "
                           f"left={blink.get('left_closed', False)} "
                           f"right={blink.get('right_closed', False)}")
            blink_counts = (f"left/right/total={blink.get('left_count', 0)}/"
                            f"{blink.get('right_count', 0)}/{blink.get('total_count', 0)}")
            blink_window = (f"window={blink.get('count_in_window', 0)}/"
                            f"{blink.get('count_window_ms', 0)}ms "
                            f"rate={blink.get('blinks_per_minute', 0):.1f}/min "
                            f"event={blink.get('event_completed', False)}")
            delta = movement.get('delta_degrees') or {}
            lines.extend([
                f"Face {start + number} / temporary T{track_key} | tracking={tracking_quality} | pp={face.get('postprocessing_version', 'n/a')}",
                f"  EAR L/R={left_ear if left_ear is not None else 'n/a'} / {right_ear if right_ear is not None else 'n/a'} | eye openness={face.get('eye_openness') if face.get('eye_openness') is not None else 'n/a'}",
                (f"  raw pitch/yaw/roll: {angles.get('pitch', 0):.1f}, {angles.get('yaw', 0):.1f}, {angles.get('roll', 0):.1f} deg"
                 if angles else '  raw pitch/yaw/roll: unavailable'),
                (f"  smoothed pitch/yaw/roll: {smoothed.get('pitch', 0):.1f}, {smoothed.get('yaw', 0):.1f}, {smoothed.get('roll', 0):.1f} deg"
                 if smoothed else '  smoothed pitch/yaw/roll: unavailable'),
                (f"  angle delta P/Y/R: {delta.get('pitch', 0):.2f}, {delta.get('yaw', 0):.2f}, {delta.get('roll', 0):.2f} deg / {movement.get('interval_ms')} ms"
                 if movement else '  angle delta candidate: waiting for next observation'),
                f"  blink state: {blink_state}",
                f"  blink counts: {blink_counts}",
                f"  blink recent: {blink_window}",
                (f"  relative pitch/yaw/roll: {relative.get('pitch', 0):.1f}, {relative.get('yaw', 0):.1f}, {relative.get('roll', 0):.1f} deg"
                 if relative else '  relative pitch/yaw/roll: not calibrated'),
                (f"  nose position: {position.get('x', 0):.3f}, {position.get('y', 0):.3f}"
                 if position else '  nose position: unavailable'),
            ])
        if not faces:
            lines.append('No face detected in the selected window.')
        panel = np.zeros((max(190, 42 + len(lines) * 24), 980, 3), dtype=np.uint8)
        for index, line in enumerate(lines):
            cv2.putText(panel, line, (14, 27 + index * 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.46, (235, 235, 235), 1, cv2.LINE_AA)
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
            self._developer_windows_ready = True
        frame = image
        if show_landmarks:
            features = result['features']
            with self.lock:
                show_tessellation = self._show_tessellation
                face_page = self._debug_face_page
            # 영상 분석용 원본 랜드마크는 건드리지 않고, 개발자 화면 복사본만 시간축으로 부드럽게 한다.
            overlay_landmarks = self._landmark_overlay_smoother.smooth(
                face.last_landmarks, features.get('faces', []), result['start_ms'])
            frame = draw_landmarks(
                frame, overlay_landmarks, pose.last_landmarks,
                face_mesh=True, tessellation=show_tessellation, show_points=False,
            )
        else:
            with self.lock:
                face_page = self._debug_face_page
        frame = draw_face_postprocessing(frame, result['features'].get('faces', []))
        if self.reidentifier is not None:
            frame = self.reidentifier.render(frame, result['features'].get('faces', []), boxes=face.last_face_boxes)
        panel = self._feature_panel(result, face_page=face_page)
        panel_shape = (panel.shape[1], panel.shape[0])
        if panel_shape != self._developer_info_shape:
            # 얼굴 수가 바뀔 때만 내용에 맞춰 조정해 사용자가 직접 바꾼 창 크기는 유지한다.
            cv2.resizeWindow(DEBUG_INFO_WINDOW, *panel_shape)
            self._developer_info_shape = panel_shape
        cv2.imshow(DEBUG_CAPTURE_WINDOW, frame)
        cv2.imshow(DEBUG_INFO_WINDOW, panel)
        key = cv2.waitKey(1) & 0xFF
        start_calibration = False
        with self.lock:
            if key in (ord('q'), ord('Q')):
                self.developer_preview = False
            elif key in (ord('v'), ord('V')):
                self._show_landmarks = not self._show_landmarks
            elif key in (ord('m'), ord('M')):
                self._show_tessellation = not self._show_tessellation
            elif key == ord('['):
                self._debug_face_page = max(0, self._debug_face_page - 1)
            elif key == ord(']'):
                face_count = len(result['features'].get('faces', []))
                page_count = max(1, (face_count + 1) // 2)
                self._debug_face_page = (self._debug_face_page + 1) % page_count
            elif key in (ord('c'), ord('C')):
                # 개인 식별 없이, 현재 단일 얼굴이 이어진 3초만 임시 기준으로 사용한다.
                # 이 보정값은 메모리에만 존재하고 최종 결과나 세션 파일에는 저장되지 않는다.
                start_calibration = True
        return start_calibration

    def snapshot(self):
        with self.lock:
            return self.latest, self.error, self.thread is not None and self.thread.is_alive()

    def model_snapshot(self):
        """현재 분석 특징에서 계산한 최신 실험 점수의 복사본을 반환한다."""
        with self.lock:
            return dict(self.model_latest)

    def participant_snapshot(self):
        """개인 표는 메모리 복사본만 제공한다. 입력이 끊기면 현재 점수를 지운다."""
        with self.lock:
            rows = deepcopy(self._participant_rows)
            timestamp = self.latest.get('start_ms') if self.latest else None
        stale = (timestamp is None or self.sid is None or
                 self.service.elapsed_ms(self.sid) - timestamp > 1500 or self._model_failed)
        if stale:
            for row in rows:
                row.update(status='not_observable', current_direction_score=None, score=None,
                           reason_codes=['observation_unavailable'])
        return rows

    def transcription_error(self):
        """오디오 초기화 또는 전사 중 발생한 오류를 화면에 전달한다."""
        with self.lock:
            runner = self.transcription
            startup_error = self._stt_startup_error
        error = (runner.last_error if runner is not None else None) or startup_error
        return str(error) if error is not None else None
