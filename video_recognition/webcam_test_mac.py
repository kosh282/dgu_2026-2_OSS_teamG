"""macOS에서 창 캡처 없이 웹캠 프레임을 직접 분석하는 데모.

저장소 루트에서 python -m video_recognition.webcam_test_mac 으로 실행한다.
출력·보정·점수·진단 로그는 기존 Windows 데모의 공통 구현을 사용한다.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2
from integration import IntegrationService
from .face_analysis import FaceAnalyzer
from .integrated_analysis import VisionPipeline
from .preview import PreviewCapture
from .webcam_test import (
    DEFAULT_LOG_DIR, DeveloperPreviewOutput, log_debug_transition,
    open_demo_log, write_frame_log,
)
from engagement_model.gaussian import GaussianRuntime


class WebcamCapture:
    """OpenCV 카메라를 파이프라인의 이미지 반환 인터페이스에 맞춘다."""

    def __init__(self, camera):
        self.camera = camera

    def read(self):
        ok, image = self.camera.read()
        if not ok or image is None:
            raise RuntimeError('웹캠 화면을 읽을 수 없습니다.')
        return image

    def close(self):
        self.camera.release()


def main():
    parser = argparse.ArgumentParser(description='macOS 웹캠 직접 입력으로 MediaPipe와 점수 시험')
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

    analyzer = pipeline = service = sid = log_handle = None
    developer_output = DeveloperPreviewOutput()
    try:
        service = IntegrationService()
        model_runtime = GaussianRuntime()
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
        log_handle, log_path = open_demo_log(args.log_dir, sid, model_runtime)
        print(f'테스트 로그: {log_path.resolve()}')
        analyzer = FaceAnalyzer(show_landmarks=True)
        preview_capture = PreviewCapture(WebcamCapture(camera))
        pipeline = VisionPipeline(
            service, sid, preview_capture, analyzer,
            model_consumer=model_runtime,
        )
        print('C: 정면 보정 시작 (3초), V: 기준점 표시 전환, Q 또는 Ctrl+C: 종료')
        started = time.monotonic()
        last_debug_signature = None
        while not args.duration or time.monotonic() - started < args.duration:
            # 샘플링으로 분석을 건너뛰어도 창 이벤트와 키 입력은 처리한다.
            source_controls = developer_output.poll_key()
            if source_controls["calibrate"]:
                pipeline.head_calibration.start()
            if source_controls["quit"]:
                break
            analyzer.show_landmarks = developer_output.show_landmarks
            result = pipeline.process_one()
            if result is None:
                continue
            last_debug_signature = log_debug_transition(
                result['features'], model_runtime.latest, last_debug_signature,
            )
            write_frame_log(log_handle, result, model_runtime.latest)
            display_frame = preview_capture.last_frame
            if reidentifier is not None:
                display_frame = reidentifier.annotate(
                    display_frame, result['start_ms'], result['features']['faces'],
                    boxes=getattr(analyzer, 'last_face_boxes', []),
                )
            controls = developer_output.render(
                display_frame, result, analyzer,
                model_runtime.latest,
            )
            if controls['calibrate']:
                pipeline.head_calibration.start()
            if controls['quit']:
                break
    except KeyboardInterrupt:
        print('테스트를 종료합니다.')
    finally:
        if reidentifier is not None:
            reidentifier.close()
        if pipeline is not None:
            pipeline.close()
        else:
            if analyzer is not None:
                analyzer.close()
        if service is not None:
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
