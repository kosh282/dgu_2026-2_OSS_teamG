"""웹캠 창을 수업 창처럼 캡처해 MediaPipe 결과를 눈으로 확인한다."""
import argparse
import ctypes
from ctypes import wintypes
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import cv2
from integration import IntegrationService
from .face_analysis import FaceAnalyzer
from .integrated_analysis import VisionPipeline, WindowCapture
from .preview import PreviewCapture, LandmarkOverlaySmoother
from .developer_view import render_feature_panel, render_video_frame
from engagement_model.window_publisher import WindowPublisher
from .group_developer_view import make_group_runtime, render_group_panel
from .window_capture import FrameNotAvailable

CAMERA_WINDOW = 'Webcam source - keep visible'
RESULT_WINDOW = 'MediaPipe results - press Q to quit'
CAPTURE_WINDOW = 'Captured frame and landmarks - V: toggle'
DEFAULT_LOG_DIR = Path(__file__).resolve().parent / 'demo_logs'


def open_demo_log(log_dir, session_id, model_runtime):
    """분석 원본 영상은 저장하지 않고, 프레임별 특징·모델 판정만 JSONL로 기록한다."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = log_dir / f'webcam-{timestamp}-{session_id[:8]}.jsonl'
    handle = path.open('w', encoding='utf-8', newline='\n')
    handle.write(json.dumps({
        'record_type': 'run_metadata',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'session_id': session_id,
        'model_id': model_runtime.model_id,
        'model_version': model_runtime.model_version,
        'config': asdict(model_runtime.config),
        'score_display_hold_seconds': getattr(model_runtime, 'display_score_hold_seconds', None),
        'note': '영상 프레임은 저장하지 않으며, 이 파일은 데모 진단용 특징/판정 기록입니다.',
    }, ensure_ascii=False, allow_nan=False) + '\n')
    handle.flush()
    return handle, path


def write_frame_log(handle, result, model_result):
    """한 분석 프레임에서 각도 입력과 실제/화면용 점수를 구분해 기록한다."""
    features = result.get('features', {})
    faces = []
    for face in features.get('faces', []):
        faces.append({
            'transient_track_key': face.get('transient_track_key'),
            'quality': face.get('quality'),
            'raw_angles_degrees': face.get('head_angles_degrees'),
            'smoothed_angles_degrees': face.get('head_angles_smoothed_degrees'),
            'relative_angles_degrees': face.get('head_angles_relative_degrees'),
            'relative_raw_angles_degrees': face.get('head_angles_relative_raw_degrees'),
            'model_input_angle_source': face.get('head_angle_model_input_source'),
        })
    record = {
        'record_type': 'frame',
        'logged_at_utc': datetime.now(timezone.utc).isoformat(),
        'capture_ms': result.get('start_ms'),
        'feature_version': result.get('feature_version'),
        'frame_quality': result.get('quality'),
        'calibration': features.get('head_calibration_diagnostics'),
        'face_count': len(faces),
        'faces': faces,
        'model': model_result,
    }
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    handle.flush()

def result_panel(window, landmarks=None, face_page=0, model_result=None):
    """Compatibility wrapper around the shared developer panel."""
    return render_feature_panel(window, face_page=face_page, model_result=model_result)


class DeveloperPreviewOutput:
    """웹캠 데모와 프런트 개발자 모드가 공유하는 두 창 출력·키보드 처리기."""

    def __init__(self, max_preview_size=(720, 540)):
        self.max_preview_size = max_preview_size
        self.show_landmarks = True
        self.show_tessellation = False
        self.face_page = 0
        self.overlay_smoother = LandmarkOverlaySmoother()
        self.windows_ready = False
        self.result_panel_shape = None

    def render(self, frame, result, face_analyzer, model_result):
        """웹캠 데모에서 사용하던 캡처 프리뷰와 텍스트 결과를 함께 그린다."""
        if frame is None:
            return {"key": -1, "quit": False, "calibrate": False,
                    "show_landmarks": self.show_landmarks}

        if not self.windows_ready:
            cv2.namedWindow(CAPTURE_WINDOW, cv2.WINDOW_NORMAL)
            cv2.namedWindow(RESULT_WINDOW, cv2.WINDOW_NORMAL)
            height, width = frame.shape[:2]
            max_width, max_height = self.max_preview_size
            scale = min(1.0, max_width / max(width, 1), max_height / max(height, 1))
            cv2.resizeWindow(
                CAPTURE_WINDOW,
                max(320, int(width * scale)), max(240, int(height * scale)),
            )
            self.windows_ready = True

        preview = render_video_frame(
            frame, result, face_analyzer.last_landmarks,
            self.overlay_smoother, show_overlay=self.show_landmarks,
            show_tessellation=self.show_tessellation,
        )
        panel = result_panel(result, face_page=self.face_page, model_result=model_result)
        panel_shape = (panel.shape[1], panel.shape[0])
        if panel_shape != self.result_panel_shape:
            cv2.resizeWindow(RESULT_WINDOW, *panel_shape)
            self.result_panel_shape = panel_shape
        cv2.imshow(CAPTURE_WINDOW, preview)
        cv2.imshow(RESULT_WINDOW, panel)

        key = cv2.waitKey(1) & 0xFF
        return self.handle_key(key, result)

    def poll_key(self, result=None):
        """Pump OpenCV window events before capturing the displayed webcam window."""
        return self.handle_key(cv2.waitKey(1) & 0xFF, result)

    def handle_key(self, key, result=None):
        """Handle keys received before capture or after drawing the result."""
        if key in (ord('v'), ord('V')):
            self.show_landmarks = not self.show_landmarks
        elif key in (ord('m'), ord('M')):
            self.show_tessellation = not self.show_tessellation
        elif key == ord('['):
            self.face_page = max(0, self.face_page - 1)
        elif key == ord(']'):
            face_count = len((result or {}).get('features', {}).get('faces', []))
            page_count = max(1, (face_count + 1) // 2)
            self.face_page = (self.face_page + 1) % page_count

        return {
            "key": key,
            "quit": key in (ord('q'), ord('Q')),
            "calibrate": key in (ord('c'), ord('C')),
            "show_landmarks": self.show_landmarks,
        }

    def close(self):
        """공유 출력 창을 닫는다. 여러 번 호출해도 안전하다."""
        if self.windows_ready:
            for name in (CAPTURE_WINDOW, RESULT_WINDOW):
                try:
                    cv2.destroyWindow(name)
                except cv2.error:
                    pass
        self.windows_ready = False
        self.result_panel_shape = None
        self.overlay_smoother.reset()


class FrontendStylePreview(DeveloperPreviewOutput):
    """프런트 개발자 모드와 같은 모델 패널로 웹캠 분석을 표시한다."""

    WINDOW = 'Webcam developer preview - multi-face Gaussian'

    def __init__(self):
        super().__init__()
        # webcam_demo는 Mac 데모를 통해 이 모듈을 참조하므로 실행 시점에 불러온다.
        from engagement_model.webcam_demo import PoseInputDiagnostics
        self._diagnostics = PoseInputDiagnostics()

    def render(self, frame, result, face_analyzer, model):
        if frame is None:
            return {'key': -1, 'quit': False, 'calibrate': False,
                    'show_landmarks': self.show_landmarks}
        if not self.windows_ready:
            cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW, 1050, 645)
            self.windows_ready = True
        panel = render_group_panel(frame, result, face_analyzer, model, self._diagnostics,
            self.overlay_smoother, show_overlay=self.show_landmarks, page=self.face_page)
        cv2.imshow(self.WINDOW, panel)
        return self.handle_key(cv2.waitKey(1) & 0xFF, result)

    def handle_key(self, key, result=None):
        # 통합 패널의 페이지는 얼굴 수가 아닌 진단 페이지 기준이다.
        if key == ord('['):
            self.face_page -= 1
        elif key == ord(']'):
            self.face_page += 1
        else:
            return super().handle_key(key, result)
        return {'key': key, 'quit': False, 'calibrate': False,
                'show_landmarks': self.show_landmarks}

    def close(self):
        if self.windows_ready:
            try:
                cv2.destroyWindow(self.WINDOW)
            except cv2.error:
                pass
        self.windows_ready = False
        self.overlay_smoother.reset()
        self._diagnostics.reset()


class DemoMultiModel:
    """프런트와 같은 다인 런타임 및 구간 발행기를 웹캠 프레임에 연결한다."""

    def __init__(self, service, sid, capture, face, reidentifier=None):
        self.service, self.sid, self.capture, self.face = service, sid, capture, face
        self.reidentifier = reidentifier
        self.runtime = make_group_runtime(reidentifier is not None)
        self.publisher = WindowPublisher(service, sid, self.runtime,
                                         start_ms=service.elapsed_ms(sid))

    def __call__(self, feature):
        if self.reidentifier is not None:
            identities = self.reidentifier.identify(
                self.capture.last_frame, feature['start_ms'], feature['features']['faces'],
                boxes=self.face.last_face_boxes)
            self.runtime.update_identities(identities)
        return self.publisher(feature)

    def recalibrate(self):
        now = self.service.elapsed_ms(self.sid)
        self.publisher.finish(now)
        self.runtime.recalibrate()
        self.publisher = WindowPublisher(self.service, self.sid, self.runtime, start_ms=now)


def write_group_frame_log(handle, result, model_result):
    """다인 진단 로그에는 개인별 얼굴 특징·추적 ID를 저장하지 않는다."""
    handle.write(json.dumps({
        'record_type': 'frame', 'capture_ms': result.get('start_ms'),
        'frame_quality': result.get('quality'),
        'face_count': len(result.get('features', {}).get('faces', [])),
        'model': {key: model_result.get(key) for key in
                  ('status', 'score', 'current_direction_score', 'detected_count',
                   'observable_count', 'scoreable_count', 'reason_codes')},
    }, ensure_ascii=False, allow_nan=False) + '\n')
    handle.flush()


def log_group_transition(model, previous_signature):
    """개별 ID를 기록하지 않고 다인 보정·집계 상태 변화만 표시한다."""
    active = [row for row in model.participants() if row.get('status') != 'missing']
    ready = sum(row.get('calibration_status') == 'ready' for row in active)
    latest = model.latest
    signature = (len(active), ready, latest.get('status'),
                 tuple(latest.get('reason_codes') or ()))
    if signature != previous_signature:
        print(f"[vision-demo] tracked={len(active)} calibrated={ready} "
              f"scored={latest.get('scoreable_count', 0)} status={latest.get('status')} "
              f"reasons={','.join(latest.get('reason_codes') or ()) or 'none'}", flush=True)
    return signature


def log_debug_transition(features, model_result, previous_signature):
    """터미널에는 보정/모델 상태가 바뀔 때만 진단 한 줄을 남긴다."""
    calibration = features.get("head_calibration_diagnostics", {})
    status = calibration.get("status", "waiting")
    elapsed_ms = calibration.get("elapsed_ms", 0)
    progress_tick = int(elapsed_ms / 1000) if status == "collecting" else None
    reasons = tuple(model_result.get("reason_codes", []))
    signature = (
        status, calibration.get("last_reason_code"), progress_tick,
        model_result.get("status"), model_result.get("orientation_state"), reasons,
        model_result.get("display_state"),
    )
    if signature == previous_signature:
        return previous_signature

    score = model_result.get("score")
    score_text = f"{score:.1f}" if isinstance(score, (int, float)) else "unavailable"
    display_score = model_result.get("display_score")
    display_state = model_result.get("display_state", "unavailable")
    if display_state == "held" and isinstance(display_score, (int, float)):
        display_text = (
            f"{display_score:.1f} held/{model_result.get('display_age_ms')}ms"
        )
    elif display_state == "expired":
        display_text = f"expired/{model_result.get('display_age_ms')}ms"
    else:
        display_text = f"{score_text} ({display_state})"
    print(
        "[vision-demo] "
        f"calibration={status} "
        f"samples={calibration.get('sample_count', 0)}/{calibration.get('required_samples', 0)} "
        f"time={elapsed_ms / 1000.0:.1f}/{calibration.get('required_duration_ms', 0) / 1000.0:.1f}s "
        f"progress={calibration.get('progress', 0.0) * 100:.0f}% "
        f"calibration_issue={calibration.get('last_reason_code') or 'none'} | "
        f"model={model_result.get('status')} "
        f"orientation={model_result.get('orientation_state')} "
        f"score={score_text} display={display_text} reasons={','.join(reasons) or 'none'}",
        flush=True,
    )
    return signature


def main():
    parser = argparse.ArgumentParser(description='웹캠 창으로 MediaPipe와 통합 경로 시험')
    parser.add_argument('--camera', type=int, default=0, help='웹캠 번호 (기본 0)')
    parser.add_argument('--duration', type=float, default=0, help='자동 종료 시간(초), 0은 수동 종료')
    parser.add_argument('--log-dir', type=Path, default=DEFAULT_LOG_DIR,
                        help='프레임별 JSONL 로그 폴더 (기본: 저장소 루트/demo_logs)')
    parser.add_argument('--reidentify', action='store_true',
                        help='동의한 인물의 세션 내 재식별 실험 (첫 실행 시 비상업 연구용 모델 다운로드)')
    args = parser.parse_args()
    if args.duration < 0:
        parser.error('--duration은 0 이상이어야 합니다')

    reidentifier = None
    if args.reidentify:
        from .face_reidentification import FaceReidentifier
        try:
            reidentifier = FaceReidentifier()
        except Exception as exc:
            parser.exit(1, f'재식별 초기화 실패: {exc}\n')
        print('재식별 실험: ID/임베딩은 메모리에만 유지하며 JSON에 저장하지 않습니다.')

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        if reidentifier is not None:
            reidentifier.close()
        raise SystemExit('웹캠을 열 수 없습니다. 연결과 카메라 접근 권한을 확인하세요.')

    analyzer = pipeline = service = sid = log_handle = demo_model = None
    developer_output = FrontendStylePreview()
    try:
        ok, image = camera.read()
        if not ok:
            raise RuntimeError('웹캠 화면을 읽을 수 없습니다.')
        cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CAMERA_WINDOW, 640, 480)
        cv2.moveWindow(CAMERA_WINDOW, 20, 40)
        cv2.imshow(CAMERA_WINDOW, image)
        cv2.waitKey(100)
        # 수업 창 목록은 자신의 프로세스를 제외하므로 데모 창은 제목으로 직접 찾는다.
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        user32.FindWindowW.restype = wintypes.HWND
        camera_window = user32.FindWindowW(None, CAMERA_WINDOW)
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        owner = wintypes.DWORD()
        if (not camera_window or not user32.GetWindowThreadProcessId(camera_window, ctypes.byref(owner))
                or owner.value != os.getpid()):
            raise RuntimeError('웹캠 창을 찾지 못했습니다.')

        service = IntegrationService()
        model_runtime = make_group_runtime(reidentifier is not None)
        sid = service.start_session(
            'Webcam engagement demo',
            config={
                'config_version': model_runtime.config.config_version,
                'participation_model': {
                    'model_id': model_runtime.model_id,
                    'model_version': model_runtime.model_version,
                },
                'stt_model': None,
            },
        )
        analyzer = FaceAnalyzer(show_landmarks=True)
        preview_capture = PreviewCapture(WindowCapture(camera_window))
        demo_model = DemoMultiModel(service, sid, preview_capture, analyzer, reidentifier)
        model_runtime = demo_model.runtime
        log_handle, log_path = open_demo_log(args.log_dir, sid, model_runtime)
        print(f'테스트 로그: {log_path.resolve()}')
        pipeline = VisionPipeline(
            service, sid, preview_capture, analyzer,
            model_consumer=demo_model,
        )
        print('다인 보정은 자동 시작합니다. 정면을 유지하세요. C: 재보정, V: 오버레이, Q: 종료')
        started = time.monotonic()
        last_debug_signature = None
        while not args.duration or time.monotonic() - started < args.duration:
            ok, image = camera.read()
            if not ok:
                raise RuntimeError('웹캠 화면을 읽을 수 없습니다.')
            cv2.imshow(CAMERA_WINDOW, image)
            # Pump OpenCV events before window capture so the visible source frame is current.
            source_controls = developer_output.poll_key()
            if source_controls["calibrate"]:
                demo_model.recalibrate()
            if source_controls["quit"]:
                break
            analyzer.show_landmarks = developer_output.show_landmarks
            try:
                result = pipeline.process_one()
            except FrameNotAvailable:
                demo_model.publisher.advance(service.elapsed_ms(sid))
                continue
            if result is None:
                demo_model.publisher.advance(service.elapsed_ms(sid))
                continue
            last_debug_signature = log_group_transition(model_runtime, last_debug_signature)
            write_group_frame_log(log_handle, result, model_runtime.latest)
            display_frame = preview_capture.last_frame
            if reidentifier is not None:
                display_frame = reidentifier.render(
                    display_frame, result['features']['faces'],
                    boxes=getattr(analyzer, 'last_face_boxes', []),
                )
            controls = developer_output.render(
                display_frame, result, analyzer,
                model_runtime,
            )
            if controls['calibrate']:
                demo_model.recalibrate()
            if controls['quit']:
                break
    except KeyboardInterrupt:
        print('테스트를 종료합니다.')
    finally:
        if reidentifier is not None:
            reidentifier.close()
        if pipeline is not None:
            pipeline.close()
        elif analyzer is not None:
            analyzer.close()
        if demo_model is not None:
            demo_model.publisher.finish(service.elapsed_ms(sid))
        if service is not None and sid is not None:
            service.stop_session(sid)
        if log_handle is not None:
            log_handle.write(json.dumps({
                'record_type': 'run_end',
                'ended_at_utc': datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False) + '\n')
            log_handle.close()
        developer_output.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
