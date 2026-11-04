"""웹캠 창을 수업 창처럼 캡처해 MediaPipe 결과를 눈으로 확인한다."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
from integration import IntegrationService
from .face_analysis import FaceAnalyzer
from .integrated_analysis import VisionPipeline, WindowCapture, list_windows
from .pose_analysis import PoseAnalyzer
from .preview import PreviewCapture, LandmarkOverlaySmoother
from .developer_view import render_feature_panel, render_video_frame
from engagement_model.runtime import EngagementRuntime, MODEL_ID, MODEL_VERSION
from engagement_model.config import ModelConfig
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
        'model_id': MODEL_ID,
        'model_version': MODEL_VERSION,
        'config': asdict(model_runtime.config),
        'score_display_hold_seconds': model_runtime.display_score_hold_seconds,
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
    args = parser.parse_args()
    if args.duration < 0:
        parser.error('--duration은 0 이상이어야 합니다')

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        raise SystemExit('웹캠을 열 수 없습니다. 연결과 카메라 접근 권한을 확인하세요.')

    analyzer = pipeline = service = sid = log_handle = None
    try:
        ok, image = camera.read()
        if not ok:
            raise RuntimeError('웹캠 화면을 읽을 수 없습니다.')
        cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CAMERA_WINDOW, 640, 480)
        cv2.moveWindow(CAMERA_WINDOW, 20, 40)
        cv2.imshow(CAMERA_WINDOW, image)
        cv2.waitKey(100)
        matching = [handle for handle, title in list_windows() if title == CAMERA_WINDOW]
        if not matching:
            raise RuntimeError('웹캠 창을 찾지 못했습니다.')

        service = IntegrationService()
        model_runtime = EngagementRuntime(config=ModelConfig())
        sid = service.start_session(
            'Webcam engagement demo',
            config={
                'config_version': model_runtime.config.config_version,
                'participation_model': {
                    'model_id': MODEL_ID,
                    'model_version': MODEL_VERSION,
                },
                'stt_model': None,
            },
        )
        log_handle, log_path = open_demo_log(args.log_dir, sid, model_runtime)
        print(f'테스트 로그: {log_path.resolve()}')
        analyzer = FaceAnalyzer(show_landmarks=True)
        preview_capture = PreviewCapture(WindowCapture(matching[-1]))
        pose_analyzer = PoseAnalyzer(show_landmarks=True)
        pipeline = VisionPipeline(
            service, sid, preview_capture, analyzer,
            model_consumer=model_runtime, pose_analyzer=pose_analyzer,
        )
        cv2.namedWindow(RESULT_WINDOW, cv2.WINDOW_NORMAL)
        cv2.moveWindow(RESULT_WINDOW, 700, 40)
        cv2.namedWindow(CAPTURE_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CAPTURE_WINDOW, 480, 360)
        cv2.moveWindow(CAPTURE_WINDOW, 20, 560)
        print('C: 정면 보정 시작 (3초), V: 기준점 표시 전환, Q 또는 Ctrl+C: 종료')
        started = time.monotonic()
        show_overlay = True
        show_tessellation = False
        face_page = 0
        overlay_smoother = LandmarkOverlaySmoother()
        latest_face_count = 0
        result_panel_shape = None
        last_debug_signature = None
        while not args.duration or time.monotonic() - started < args.duration:
            ok, image = camera.read()
            if not ok:
                raise RuntimeError('웹캠 화면을 읽을 수 없습니다.')
            cv2.imshow(CAMERA_WINDOW, image)
            # 캡처 대상 창을 표시한 뒤 실제 수업 창과 같은 경로로 읽는다.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            if key in (ord('v'), ord('V')):
                show_overlay = not show_overlay
            if key in (ord('m'), ord('M')):
                show_tessellation = not show_tessellation
            if key == ord('['):
                face_page = max(0, face_page - 1)
            if key == ord(']'):
                face_page = (face_page + 1) % max(1, (latest_face_count + 1) // 2)
            if key in (ord('c'), ord('C')):
                pipeline.head_calibration.start()
            try:
                result = pipeline.process_one()
            except FrameNotAvailable:
                continue
            if result is None:
                continue
            latest_face_count = len(result['features'].get('faces', []))
            last_debug_signature = log_debug_transition(
                result['features'], model_runtime.latest, last_debug_signature,
            )
            write_frame_log(log_handle, result, model_runtime.latest)
            # 분석에 사용한 동일 프레임의 복사본에만 표시한다. V로 원본을 비교한다.
            preview = render_video_frame(
                preview_capture.last_frame, result, analyzer.last_landmarks,
                pose_analyzer.last_landmarks, overlay_smoother,
                show_overlay=show_overlay, show_tessellation=show_tessellation,
            )
            cv2.imshow(CAPTURE_WINDOW, preview)
            panel = result_panel(result, face_page=face_page, model_result=model_runtime.latest)
            panel_shape = (panel.shape[1], panel.shape[0])
            if panel_shape != result_panel_shape:
                cv2.resizeWindow(RESULT_WINDOW, *panel_shape)
                result_panel_shape = panel_shape
            cv2.imshow(RESULT_WINDOW, panel)
    except KeyboardInterrupt:
        print('테스트를 종료합니다.')
    finally:
        if pipeline is not None:
            pipeline.close()
        elif analyzer is not None:
            analyzer.close()
        if service is not None:
            service.stop_session(sid)
        if log_handle is not None:
            log_handle.write(json.dumps({
                'record_type': 'run_end',
                'ended_at_utc': datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False) + '\n')
            log_handle.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
