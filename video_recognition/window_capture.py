"""Windows Graphics Capture로 지정한 창의 프레임을 받는다."""
import ctypes
from ctypes import wintypes
import threading
import time


class FrameNotAvailable(Exception):
    """창이 변하지 않아 아직 새 프레임이 없는 정상적인 대기 상태."""


class WindowCapture:
    """창 핸들에 연결하며 새 프레임만 전달한다. 여러 프레임이 오면 최신 프레임만 유지한다."""

    def __init__(self, hwnd, timeout=0.5):
        from windows_capture import WindowsCapture
        self.user32 = ctypes.WinDLL('user32', use_last_error=True)
        self.user32.IsWindow.argtypes = [wintypes.HWND]
        self.user32.IsWindow.restype = wintypes.BOOL
        self.user32.IsIconic.argtypes = [wintypes.HWND]
        self.user32.IsIconic.restype = wintypes.BOOL
        if not self.user32.IsWindow(hwnd):
            raise ValueError('선택한 창을 찾을 수 없습니다. 창 목록을 다시 확인하세요.')
        self.hwnd = hwnd
        self.timeout = timeout
        self.condition = threading.Condition()
        self.latest_frame = None
        self.generation = 0
        self.consumed = 0
        self.closed = False
        self.stopped = False
        self.started_at = time.monotonic()
        self.session = WindowsCapture(window_hwnd=hwnd, cursor_capture=False)

        @self.session.event
        def on_frame_arrived(frame, capture_control):
            # 라이브러리의 프레임 버퍼는 콜백 이후 재사용될 수 있어 즉시 복사한다.
            import numpy as np
            image = np.ascontiguousarray(frame.frame_buffer[:, :, :3])
            with self.condition:
                self.latest_frame = image
                self.generation += 1
                self.condition.notify_all()

        @self.session.event
        def on_closed():
            with self.condition:
                self.closed = True
                self.condition.notify_all()

        self.control = self.session.start_free_threaded()

    def read(self):
        deadline = time.monotonic() + self.timeout
        with self.condition:
            if self.closed or self.stopped or not self.user32.IsWindow(self.hwnd):
                raise RuntimeError('선택한 창이 닫혔거나 캡처가 종료되었습니다')
            if self.user32.IsIconic(self.hwnd):
                raise RuntimeError('선택한 창이 최소화되었습니다. 창을 복원해 주세요')
            while self.generation == self.consumed:
                if self.closed or self.stopped or not self.user32.IsWindow(self.hwnd):
                    raise RuntimeError('선택한 창이 닫혔거나 캡처가 종료되었습니다')
                if self.user32.IsIconic(self.hwnd):
                    raise RuntimeError('선택한 창이 최소화되었습니다. 창을 복원해 주세요')
                if self.control.is_finished():
                    # 캡처 스레드 오류는 wait에서 원래 예외로 전달된다.
                    self.control.wait()
                    raise RuntimeError('창 캡처가 예상치 못하게 종료되었습니다')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if self.generation == 0 and time.monotonic() - self.started_at >= 10:
                        raise TimeoutError('창에서 첫 프레임을 받지 못했습니다. 캡처 지원과 권한을 확인하세요')
                    raise FrameNotAvailable('창에서 아직 새 프레임이 없습니다')
                self.condition.wait(min(remaining, 0.1))
            self.consumed = self.generation
            return self.latest_frame

    def close(self):
        with self.condition:
            if self.stopped:
                return
            self.stopped = True
            self.condition.notify_all()
        self.control.stop()
