"""웹캠 창을 수업 창처럼 캡처해 MediaPipe 결과를 눈으로 확인한다."""
import argparse
import time

import cv2
import numpy as np

from integration import IntegrationService
from .face_analysis import FaceAnalyzer
from .integrated_analysis import VisionPipeline, WindowCapture, list_windows

CAMERA_WINDOW = 'Webcam source - keep visible'
RESULT_WINDOW = 'MediaPipe results - press Q to quit'


def result_panel(window, landmarks):
    """OpenCV는 한글 표시를 지원하지 않아 화면에는 영문 라벨을 사용한다."""
    faces = window['features']['faces']
    canvas = np.zeros((640, 740, 3), dtype=np.uint8)
    lines = [
        'MediaPipe face landmarks (x, y normalized; z relative)',
        f'Capture time: {window["start_ms"]} ms   Faces: {len(faces)}',
        f'Quality: {window["quality"]["status"]}',
        'Raw coordinates: nose index 1 and chin index 152',
    ]
    if not faces:
        lines.append('No face detected. Face the camera and keep source window visible.')
    # 화면에는 최대 세 얼굴의 원본 기준점 일부를 보여준다. 전체 좌표는 메모리에만 유지한다.
    for index, face in enumerate(faces[:3]):
        points = landmarks[index]
        nose, chin = points[1], points[152]
        angles = face['head_angles_degrees']
        eye = face['eye_openness']
        lines.extend([
            f'Face {index + 1}: {len(points)} landmarks (order may change)',
            f'  nose: {nose[0]:.3f}, {nose[1]:.3f}, {nose[2]:.3f}',
            f'  chin: {chin[0]:.3f}, {chin[1]:.3f}, {chin[2]:.3f}',
            '  Derived features (not direct MediaPipe outputs):',
            f'  eye openness: {eye:.3f}' if eye is not None else '  eye openness: unavailable',
            (f'  pitch/yaw/roll: {angles["pitch"]:.1f}, {angles["yaw"]:.1f}, {angles["roll"]:.1f} deg'
             if angles else '  head angles: unavailable'),
        ])
    if len(faces) > 3:
        lines.append(f'Additional faces detected: {len(faces) - 3}')
    for line_number, line in enumerate(lines):
        cv2.putText(canvas, line, (15, 28 + line_number * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description='웹캠 창으로 MediaPipe와 통합 경로 시험')
    parser.add_argument('--camera', type=int, default=0, help='웹캠 번호 (기본 0)')
    parser.add_argument('--duration', type=float, default=0, help='자동 종료 시간(초), 0은 수동 종료')
    args = parser.parse_args()
    if args.duration < 0:
        parser.error('--duration은 0 이상이어야 합니다')

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        raise SystemExit('웹캠을 열 수 없습니다. 연결과 카메라 접근 권한을 확인하세요.')

    analyzer = pipeline = service = sid = None
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
        sid = service.start_session('웹캠 영상 분석 테스트')
        analyzer = FaceAnalyzer(show_landmarks=True)
        pipeline = VisionPipeline(service, sid, WindowCapture(matching[-1]), analyzer)
        cv2.namedWindow(RESULT_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(RESULT_WINDOW, 740, 640)
        cv2.moveWindow(RESULT_WINDOW, 700, 40)
        print('웹캠 창을 가리지 말고 카메라를 바라보세요. 종료: Q 또는 Ctrl+C')
        started = time.monotonic()
        while not args.duration or time.monotonic() - started < args.duration:
            ok, image = camera.read()
            if not ok:
                raise RuntimeError('웹캠 화면을 읽을 수 없습니다.')
            cv2.imshow(CAMERA_WINDOW, image)
            # 캡처 대상 창을 표시한 뒤 실제 수업 창과 같은 경로로 읽는다.
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            result = pipeline.process_one()
            cv2.imshow(RESULT_WINDOW, result_panel(result, analyzer.last_landmarks))
    except KeyboardInterrupt:
        print('테스트를 종료합니다.')
    finally:
        if pipeline is not None:
            pipeline.close()
        elif analyzer is not None:
            analyzer.close()
        if service is not None:
            service.stop_session(sid)
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
