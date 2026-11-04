"""수업 창 캡처·공통 얼굴 분석·모델 입력 및 통합 서비스 연결."""
from dataclasses import dataclass
from collections import deque
from pathlib import Path
import json
import math

from .face_analysis import DEFAULT_MODEL, FaceAnalyzer
from .face_postprocessing import FacePostProcessor, POSTPROCESSING_VERSION
from .head_calibration import HeadCalibration
from .pose_analysis import PoseAnalyzer
from .window_capture import FrameNotAvailable, WindowCapture

FEATURE_VERSION = 'experimental-0.8'
DEFAULT_FACE_ANALYSIS_HZ = 10.0
DEFAULT_POSE_ANALYSIS_HZ = 5.0
DEFAULT_MAX_ANALYSIS_DIMENSION = 1920


def _without_transient_tracking(value):
    """저장용 복사본에서 화면/런타임 전용 얼굴 추적 키를 재귀적으로 뺀다."""
    if isinstance(value, dict):
        return {
            key: _without_transient_tracking(item)
            for key, item in value.items() if key != 'transient_track_key'
        }
    if isinstance(value, list):
        return [_without_transient_tracking(item) for item in value]
    return value


def feature_quality(faces, bodies, pose_enabled, pose_observation_age_ms=None,
                    max_pose_age_ms=500):
    """프레임별 결측을 요약한다. valid는 측정 정확도가 검증됐다는 뜻이 아니다."""
    face_incomplete = any(
        face.get('eye_openness') is None
        or face.get('head_angles_degrees') is None
        or face.get('quality', {}).get('status', 'valid') != 'valid'
        for face in faces
    )
    body_incomplete = not pose_enabled or not bodies or any(
        body.get('shoulder_center') is None for body in bodies
    )
    reasons = []
    if not faces:
        reasons.append('face_not_found')
    if face_incomplete or body_incomplete:
        reasons.append('insufficient_observation')
    if (pose_enabled and pose_observation_age_ms is not None
            and pose_observation_age_ms > max_pose_age_ms):
        reasons.append('stale_pose_observation')
    if not faces and not bodies:
        status = 'not_observable'
    else:
        status = 'partial' if reasons else 'valid'
    return {'status': status, 'reason_codes': reasons}


def list_windows():
    """선택 가능한 표시 중인 최상위 Windows 창의 (핸들, 제목) 목록."""
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    # 원본 웹캠 고정 입력 대신 사용자가 수업 창을 선택할 수 있게 목록을 만든다.
    found = []

    @callback_type
    def collect(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length:
                title = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, title, length + 1)
                found.append((hwnd, title.value))
        return True
    if not user32.EnumWindows(collect, 0):
        raise OSError(ctypes.get_last_error(), '창 목록 조회에 실패했습니다')
    return found


@dataclass
class VideoFrame:
    """캡처한 이미지와 공통 세션 시각을 메모리에서 함께 전달한다."""
    session_id: str
    frame_id: int
    capture_ms: int
    width: int
    height: int
    color_format: str
    image: object


class VisionPipeline:
    """호출자가 수업 세션과 창/분석기의 생명주기를 관리한다."""
    def __init__(
        self, service, session_id, capture, analyzer,
        model_consumer=None, intermediate_path=None, pose_analyzer=None,
        face_postprocessor=None,
        face_analysis_hz=DEFAULT_FACE_ANALYSIS_HZ,
        pose_analysis_hz=DEFAULT_POSE_ANALYSIS_HZ,
        max_analysis_dimension=DEFAULT_MAX_ANALYSIS_DIMENSION,
    ):
        self.service, self.session_id = service, session_id
        self.capture, self.analyzer = capture, analyzer
        self.model_consumer = model_consumer
        self.pose_analyzer = pose_analyzer
        # 후처리기는 주입 가능하게 두어 모델 확정 시 구현체 교체가 쉽도록 한다.
        self.face_postprocessor = face_postprocessor or FacePostProcessor()
        if face_analysis_hz <= 0 or pose_analysis_hz <= 0:
            raise ValueError('분석 주기는 0보다 커야 합니다.')
        self.face_analysis_hz = float(face_analysis_hz)
        self.pose_analysis_hz = float(pose_analysis_hz)
        self.max_analysis_dimension = max_analysis_dimension
        self.face_interval_ms = round(1000 / self.face_analysis_hz)
        self.pose_interval_ms = round(1000 / self.pose_analysis_hz)
        self.last_face_analysis_ms = None
        self.last_pose_analysis_ms = None
        self.latest_faces = []
        self.latest_bodies = []
        self.latest_pose_movement = None
        self.face_sample_times = deque(maxlen=10)
        self.pose_sample_times = deque(maxlen=10)
        self.previous_positions = {}
        self.head_calibration = HeadCalibration()
        self.intermediate_path = Path(intermediate_path) if intermediate_path is not None else None
        self.frame_id = 0
        self.state = None

    def _analysis_image(self, image):
        """창 원본은 보존하고 모델 입력에만 종횡비 유지 축소를 적용한다."""
        height, width = image.shape[:2]
        limit = self.max_analysis_dimension
        if limit is None or limit <= 0 or max(width, height) <= limit:
            return image
        import cv2
        scale = limit / max(width, height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _sample_rate(timestamps):
        if len(timestamps) < 2:
            return None
        elapsed_ms = timestamps[-1] - timestamps[0]
        if elapsed_ms <= 0:
            return None
        return round((len(timestamps) - 1) * 1000 / elapsed_ms, 2)

    def _movement(self, key, items, position_key, capture_ms):
        """단일 검출만 임시 비교한다. 사람 식별이나 추적을 보장하지 않는다."""
        position = items[0].get(position_key) if len(items) == 1 else None
        previous = self.previous_positions.pop(key, None)
        movement = None
        if position is not None:
            if previous is not None:
                old_time, old_position = previous
                delta_ms = capture_ms - old_time
                if 0 < delta_ms <= 1000:
                    movement = {
                        'distance_normalized': math.hypot(
                            position['x'] - old_position['x'], position['y'] - old_position['y']),
                        'interval_ms': delta_ms,
                    }
            self.previous_positions[key] = (capture_ms, position.copy())
        return movement

    def _status(self, state, message, reason=None):
        # 통합부에 상태가 바뀐 경우에만 알리고 반복 이벤트는 만들지 않는다.
        if self.state == state:
            return
        now = self.service.elapsed_ms(self.session_id)
        event = self.service.make_event(
            self.session_id,
            'analysis_status',
            {'component': 'vision', 'state': state, 'message': message},
            now,
            now,
            'vision',
            quality_status='not_observable' if reason else 'valid',
            reason_codes=[reason] if reason else [],
        )
        self.service.receive_event(event)
        self.state = state

    def process_one(self):
        try:
            # Keep the captured frame for preview, but send a bounded-size copy to the models.
            image = self.capture.read()
            capture_ms = self.service.elapsed_ms(self.session_id)
            source_height, source_width = image.shape[:2]
            analysis_image = self._analysis_image(image)
            height, width = analysis_image.shape[:2]
            frame = VideoFrame(
                self.session_id, self.frame_id, capture_ms, width, height, 'BGR', analysis_image,
            )
            self.frame_id += 1

            # Read every newest capture, while limiting expensive face inference to its target rate.
            if (self.last_face_analysis_ms is not None
                    and capture_ms - self.last_face_analysis_ms < self.face_interval_ms):
                return None

            faces = self.analyzer.analyze(frame)
            self.last_face_analysis_ms = capture_ms
            self.face_sample_times.append(capture_ms)
            faces = self.face_postprocessor.process(
                faces, getattr(self.analyzer, 'last_face_centers', []), capture_ms)
            # 정면 기준과 모델 입력이 같은 스무딩 각도를 쓰도록 후처리를 먼저 적용한다.
            calibration_status = self.head_calibration.update(faces, capture_ms)
            calibration_diagnostics = self.head_calibration.diagnostics(capture_ms)

            # Run pose less often and reuse its latest observation between pose samples.
            pose_due = self.pose_analyzer is not None and (
                self.last_pose_analysis_ms is None
                or capture_ms - self.last_pose_analysis_ms >= self.pose_interval_ms
            )
            if pose_due:
                bodies = self.pose_analyzer.analyze(frame)
                self.last_pose_analysis_ms = capture_ms
                self.pose_sample_times.append(capture_ms)
                self.latest_bodies = bodies
                self.latest_pose_movement = self._movement(
                    'body', bodies, 'shoulder_center', capture_ms)
            else:
                bodies = self.latest_bodies

            head_movement = self._movement('head', faces, 'head_position', capture_ms)
            # Pose motion is new only on pose samples; do not repeat the same candidate at face rate.
            body_movement = self.latest_pose_movement if pose_due else None
            pose_age_ms = (
                capture_ms - self.last_pose_analysis_ms
                if self.last_pose_analysis_ms is not None else None
            )
            max_pose_age_ms = max(self.pose_interval_ms * 2, 500)
            feature_window = {
                'session_id': self.session_id,
                'start_ms': capture_ms,
                'end_ms': capture_ms,
                'features': {
                    'faces': faces,
                    'upper_bodies': bodies,
                    'head_movement_candidate': head_movement,
                    'upper_body_movement_candidate': body_movement,
                    'head_calibration_status': calibration_status,
                    'head_calibration_diagnostics': calibration_diagnostics,
                    'face_postprocessing_version': POSTPROCESSING_VERSION,
                    'sampling': {
                        'target_face_hz': self.face_analysis_hz,
                        'actual_face_hz': self._sample_rate(self.face_sample_times),
                        'target_pose_hz': self.pose_analysis_hz if self.pose_analyzer else None,
                        'actual_pose_hz': self._sample_rate(self.pose_sample_times),
                        'pose_observation_age_ms': pose_age_ms,
                        'source_size': {'width': source_width, 'height': source_height},
                        'analysis_size': {'width': width, 'height': height},
                    },
                },
                'quality': feature_quality(
                    faces, bodies, self.pose_analyzer is not None,
                    pose_age_ms, max_pose_age_ms,
                ),
                'feature_version': FEATURE_VERSION,
            }
            # Optional intermediate rows use the same face sampling rate as the model input.
            if self.intermediate_path is not None:
                with self.intermediate_path.open('a', encoding='utf-8') as handle:
                    stored_window = _without_transient_tracking(feature_window)
                    line = json.dumps(stored_window, ensure_ascii=False, allow_nan=False)
                    handle.write(line + '\n')
            if self.model_consumer is not None:
                self.model_consumer(feature_window)
            self._status('running', '영상 분석 중입니다')
            return feature_window
        except FrameNotAvailable:
            raise
        except Exception:
            self._status('unavailable', '영상 처리 또는 결과 전달에 실패했습니다', 'unknown')
            raise

    def close(self):
        try:
            try:
                if hasattr(self.capture, 'close'):
                    self.capture.close()
            finally:
                try:
                    self.analyzer.close()
                finally:
                    if self.pose_analyzer is not None:
                        self.pose_analyzer.close()
        finally:
            if hasattr(self.face_postprocessor, 'reset'):
                self.face_postprocessor.reset()
            self._status('stopped', '영상 분석을 종료했습니다')


def main():
    """저장소 루트에서 -m 옵션으로 실행하는 간단한 연결 예제."""
    import argparse
    import time
    from integration import IntegrationService
    parser = argparse.ArgumentParser(description='수업 창 영상 분석')
    parser.add_argument('--list', action='store_true', help='선택 가능한 창 목록 출력')
    parser.add_argument('--hwnd', type=int, help='분석할 창의 핸들')
    parser.add_argument('--model', default=str(DEFAULT_MODEL), help='MediaPipe 모델 파일')
    parser.add_argument('--intermediate', help='선택적 중간 결과 JSONL 경로')
    parser.add_argument('--session', default='session.json', help='최종 세션 JSON 경로')
    parser.add_argument('--quiet', action='store_true', help='특징값 터미널 표시 끄기')
    parser.add_argument('--face-hz', type=float, default=DEFAULT_FACE_ANALYSIS_HZ,
                        help='Target face analysis rate (default: 10 Hz)')
    parser.add_argument('--pose-hz', type=float, default=DEFAULT_POSE_ANALYSIS_HZ,
                        help='Target pose analysis rate (default: 5 Hz)')
    parser.add_argument('--max-analysis-dimension', type=int,
                        default=DEFAULT_MAX_ANALYSIS_DIMENSION,
                        help='Maximum model input dimension; 0 disables resizing')
    parser.add_argument('--no-preview', action='store_true', help='캡처 미리보기 창 끄기')
    parser.add_argument('--calibrate', action='store_true', help='시작 시 단일 얼굴 정면 기준 수집')
    args = parser.parse_args()
    if args.list:
        for hwnd, title in list_windows():
            print(hwnd, title)
        return
    if args.hwnd is None:
        parser.error('--list로 창을 확인하고 --hwnd를 지정하세요')
    if args.intermediate and Path(args.intermediate).resolve() == Path(args.session).resolve():
        parser.error('중간 결과와 최종 세션 파일은 서로 다른 경로여야 합니다')
    service = IntegrationService()
    sid = service.start_session('온라인 수업')
    # 참여 모델이 정해지면 model_consumer에 실제 모델 입력 함수를 연결한다.
    analyzer = FaceAnalyzer(args.model, show_landmarks=not args.no_preview)
    try:
        pose_analyzer = PoseAnalyzer(show_landmarks=not args.no_preview)
    except Exception:
        analyzer.close()
        raise
    try:
        capture = WindowCapture(args.hwnd)
    except Exception:
        analyzer.close()
        pose_analyzer.close()
        raise
    preview_window = 'Class capture - V: landmarks, Q: quit'
    if not args.no_preview:
        import cv2
        from .preview import PreviewCapture, draw_landmarks
        # 분석기에 전달한 프레임 자체를 보관해 별도 재캡처에 따른 차이를 없앤다.
        capture = PreviewCapture(capture)
    pipeline = VisionPipeline(
        service, sid, capture, analyzer,
        intermediate_path=args.intermediate, pose_analyzer=pose_analyzer,
        face_analysis_hz=args.face_hz,
        pose_analysis_hz=args.pose_hz,
        max_analysis_dimension=args.max_analysis_dimension,
    )
    if args.calibrate:
        pipeline.head_calibration.start()
    try:
        if not args.no_preview:
            import ctypes
            cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(preview_window, 600, 420)
            screen_width = ctypes.windll.user32.GetSystemMetrics(0)
            cv2.moveWindow(preview_window, max(0, screen_width - 620), 40)
            print('C: 단일 얼굴 정면 보정 시작 (3초), V: 기준점 전환, Q: 종료')
        next_display = 0.0
        show_overlay = True
        while True:
            try:
                result = pipeline.process_one()
            except FrameNotAvailable:
                if not args.no_preview:
                    key = cv2.waitKey(1) & 0xFF
                    # 정지 화면에서도 창 닫기와 종료 키를 처리한다.
                    if key in (ord('q'), ord('Q')) or cv2.getWindowProperty(preview_window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key in (ord('c'), ord('C')):
                        pipeline.head_calibration.start()
                continue
            if result is None:
                if not args.no_preview:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), ord('Q')) or cv2.getWindowProperty(preview_window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key in (ord('v'), ord('V')):
                        show_overlay = not show_overlay
                    if key in (ord('c'), ord('C')):
                        pipeline.head_calibration.start()
                continue
            if not args.no_preview:
                frame = capture.last_frame
                if show_overlay:
                    frame = draw_landmarks(frame, analyzer.last_landmarks, pose_analyzer.last_landmarks)
                else:
                    frame = frame.copy()
                status = result['features']['head_calibration_status']
                cv2.putText(frame, f'Head calibration: {status} (C to start)', (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(preview_window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')) or cv2.getWindowProperty(preview_window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord('v'), ord('V')):
                    show_overlay = not show_overlay
                if key in (ord('c'), ord('C')):
                    pipeline.head_calibration.start()
            if not args.quiet and time.monotonic() >= next_display:
                print(json.dumps(result, ensure_ascii=False), flush=True)
                next_display = time.monotonic() + 2
    except KeyboardInterrupt:
        print('영상 분석을 종료합니다.')
    finally:
        try:
            pipeline.close()
        finally:
            if not args.no_preview:
                cv2.destroyAllWindows()
            service.stop_session(sid)
            service.save_session(sid, args.session)
            print('세션 저장:', args.session)


if __name__ == '__main__':
    main()
