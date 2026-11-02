"""수업 창 캡처·공통 얼굴 분석·모델 입력 및 통합 서비스 연결."""
from dataclasses import dataclass
from pathlib import Path
import json

from .face_analysis import DEFAULT_MODEL, FaceAnalyzer


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


class WindowCapture:
    """Windows 창의 화면상 사각 영역을 캡처한다. 가려진 부분은 그대로 가려진다."""
    def __init__(self, hwnd):
        import ctypes
        from ctypes import wintypes
        self.user32 = ctypes.WinDLL('user32', use_last_error=True)
        self.user32.IsWindow.argtypes = [wintypes.HWND]
        self.user32.IsWindow.restype = wintypes.BOOL
        self.user32.IsIconic.argtypes = [wintypes.HWND]
        self.user32.IsIconic.restype = wintypes.BOOL
        self.user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.user32.GetClientRect.restype = wintypes.BOOL
        self.user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        self.user32.ClientToScreen.restype = wintypes.BOOL
        self.hwnd = hwnd

    def read(self):
        from ctypes import wintypes, byref
        from PIL import ImageGrab
        import numpy as np
        if not self.user32.IsWindow(self.hwnd) or self.user32.IsIconic(self.hwnd):
            raise RuntimeError('수업 창이 닫혔거나 최소화되었습니다')
        # 창 테두리를 제외한 내용 영역을 화면 좌표로 변환해 캡처한다.
        # 화면에 보이는 픽셀을 읽기 때문에 다른 창에 가려진 부분도 포함된다.
        rect, point = wintypes.RECT(), wintypes.POINT(0, 0)
        if not self.user32.GetClientRect(self.hwnd, byref(rect)) or not self.user32.ClientToScreen(self.hwnd, byref(point)):
            raise RuntimeError('수업 창 영역을 확인할 수 없습니다')
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width <= 0 or height <= 0:
            raise RuntimeError('수업 창 크기가 올바르지 않습니다')
        bounds = (point.x, point.y, point.x + width, point.y + height)
        image = ImageGrab.grab(bbox=bounds, all_screens=True).convert('RGB')
        # 공통 입력은 OpenCV와 동일한 BGR 순서로 전달한다.
        return np.ascontiguousarray(np.asarray(image)[:, :, ::-1])


class VisionPipeline:
    """호출자가 수업 세션과 창/분석기의 생명주기를 관리한다."""
    def __init__(
        self, service, session_id, capture, analyzer,
        model_consumer=None, intermediate_path=None,
    ):
        self.service, self.session_id = service, session_id
        self.capture, self.analyzer = capture, analyzer
        self.model_consumer = model_consumer
        self.intermediate_path = Path(intermediate_path) if intermediate_path is not None else None
        self.frame_id = 0
        self.state = None

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
            image = self.capture.read()
            # 원본의 프레임마다 33ms 증가하는 가상 시각 대신 실제 공통 시계를 사용한다.
            capture_ms = self.service.elapsed_ms(self.session_id)
            height, width = image.shape[:2]
            frame = VideoFrame(
                self.session_id, self.frame_id, capture_ms, width, height, 'BGR', image,
            )
            self.frame_id += 1
            faces = self.analyzer.analyze(frame)
            # 한 프레임은 시점 데이터다. 모델 담당자가 창 길이와 특징 의미를 확정해야 한다.
            # 얼굴 특징은 메모리의 모델 입력이며 세션 JSON에 직접 넣지 않는다.
            partial = any(
                face.get('eye_openness') is None or face.get('head_angles_degrees') is None
                for face in faces
            )
            quality = 'not_observable' if not faces else ('partial' if partial else 'valid')
            reasons = ['face_not_found'] if not faces else (['unknown'] if partial else [])
            feature_window = {
                'session_id': self.session_id,
                'start_ms': capture_ms,
                'end_ms': capture_ms,
                'features': {'faces': faces},
                'quality': {'status': quality, 'reason_codes': reasons},
                'feature_version': 'experimental-0.2',
            }
            # 선택적 중간 기록은 세션 파일과 별도로 둔다. 자동 삭제되지는 않는다.
            if self.intermediate_path is not None:
                with self.intermediate_path.open('a', encoding='utf-8') as handle:
                    line = json.dumps(feature_window, ensure_ascii=False, allow_nan=False)
                    handle.write(line + '\n')
            # 모델 미확정: 연결 함수가 제공된 경우에만 특징을 메모리로 넘긴다.
            if self.model_consumer is not None:
                self.model_consumer(feature_window)
            self._status('running', '영상 분석 중입니다')
            return feature_window
        except Exception:
            self._status('unavailable', '영상 처리 또는 결과 전달에 실패했습니다', 'unknown')
            raise

    def close(self):
        try:
            self.analyzer.close()
        finally:
            self._status('stopped', '영상 분석을 종료했습니다')


def main():
    """저장소 루트에서 -m 옵션으로 실행하는 간단한 연결 예제."""
    import argparse
    from integration import IntegrationService
    parser = argparse.ArgumentParser(description='수업 창 영상 분석')
    parser.add_argument('--list', action='store_true', help='선택 가능한 창 목록 출력')
    parser.add_argument('--hwnd', type=int, help='분석할 창의 핸들')
    parser.add_argument('--model', default=str(DEFAULT_MODEL), help='MediaPipe 모델 파일')
    parser.add_argument('--intermediate', help='선택적 중간 결과 JSONL 경로')
    parser.add_argument('--session', default='session.json', help='최종 세션 JSON 경로')
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
    pipeline = VisionPipeline(
        service, sid, WindowCapture(args.hwnd), FaceAnalyzer(args.model),
        intermediate_path=args.intermediate,
    )
    try:
        while True:
            pipeline.process_one()
    except KeyboardInterrupt:
        print('영상 분석을 종료합니다.')
    finally:
        pipeline.close()
        service.stop_session(sid)
        service.save_session(sid, args.session)
        print('세션 저장:', args.session)


if __name__ == '__main__':
    main()
