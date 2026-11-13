"""Windows Graphics Capture로 지정한 창의 프레임을 받는다."""
import ctypes
from ctypes import wintypes
import threading
import time


class FrameNotAvailable(Exception):
    """창이 변하지 않아 아직 새 프레임이 없는 정상적인 대기 상태."""


class WindowCapture:
    """창 핸들에 연결하며 새 프레임만 전달한다. 여러 프레임이 오면 최신 프레임만 유지한다.

    같은 창의 캡처는 프로세스 안에서 재사용한다. `WindowCapture(hwnd)`를 다시 만들면 동작 중인
    기존 캡처를 돌려주고, `close()`는 네이티브 캡처를 멈추지 않고 일시정지만 한다.
    windows_capture 2.0.1은 MediaPipe가 쓰이는 프로세스에서 캡처를 멈췄다가 다시 만들면
    access violation으로 프로세스 전체를 종료시킨다(Windows 실측: 대시보드의 두 번째 수업,
    또는 준비 화면 미리보기 뒤 첫 수업). 닫는 순서나 실행 스레드를 바꿔도 재현됐고, 캡처를
    재사용하면 수업 16회 연속 정상이었다. 창이 닫혔거나 캡처가 끝난 경우에만 새로 만든다.
    """

    _shared = {}
    _shared_lock = threading.Lock()

    def __new__(cls, hwnd=None, timeout=0.5, expected_process_id=None):
        # hwnd 없이 만드는 경우(테스트에서 __init__을 거치지 않는 생성)는 재사용하지 않는다.
        if hwnd is None:
            instance = super().__new__(cls)
            instance._reuse_pending = False
            return instance
        with cls._shared_lock:
            existing = cls._shared.get(hwnd)
            if existing is not None and existing._reusable():
                existing._reuse_pending = True
                return existing
            instance = super().__new__(cls)
            instance._reuse_pending = False
            return instance

    def __init__(self, hwnd, timeout=0.5, expected_process_id=None):
        if self._reuse_pending:
            self._reuse_pending = False
            self._resume(timeout, expected_process_id)
            return
        from windows_capture import WindowsCapture
        self.user32 = ctypes.WinDLL('user32', use_last_error=True)
        self.user32.IsWindow.argtypes = [wintypes.HWND]
        self.user32.IsWindow.restype = wintypes.BOOL
        self.user32.IsIconic.argtypes = [wintypes.HWND]
        self.user32.IsIconic.restype = wintypes.BOOL
        if not self.user32.IsWindow(hwnd):
            raise ValueError('선택한 창을 찾을 수 없습니다. 창 목록을 다시 확인하세요.')
        self.user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        owner = wintypes.DWORD()
        if not self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner)) or not owner.value:
            raise ValueError('선택한 창의 프로세스를 확인할 수 없습니다.')
        self.process_id = owner.value
        if expected_process_id is not None and self.process_id != expected_process_id:
            raise ValueError('선택한 창의 프로세스가 변경되었습니다. 수업 창을 다시 선택하세요.')
        self.hwnd = hwnd
        self.timeout = timeout
        self.condition = threading.Condition()
        self.latest_frame = None
        self.generation = 0
        self.consumed = 0
        self.closed = False
        self.stopped = False
        # 재사용 대기 중(close 이후)에는 프레임 복사를 건너뛰어 CPU를 아낀다.
        self.paused = False
        self.started_at = time.monotonic()
        self.session = WindowsCapture(window_hwnd=hwnd, cursor_capture=False)

        @self.session.event
        def on_frame_arrived(frame, capture_control):
            if self.paused:
                return
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
        with type(self)._shared_lock:
            type(self)._shared[hwnd] = self

    def _reusable(self):
        try:
            return (not self.closed and not self.control.is_finished()
                    and bool(self.user32.IsWindow(self.hwnd)))
        except AttributeError:  # 초기화가 끝나지 않은 객체
            return False

    def _resume(self, timeout, expected_process_id):
        """일시정지된 캡처를 새 사용자에게 넘긴다. 이전 사용의 오래된 프레임은 버린다."""
        if expected_process_id is not None and self.process_id != expected_process_id:
            raise ValueError('선택한 창의 프로세스가 변경되었습니다. 수업 창을 다시 선택하세요.')
        with self.condition:
            self.timeout = timeout
            self.consumed = self.generation
            self.stopped = False
            self.paused = False
            self.started_at = time.monotonic()

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
        """사용을 끝낸다. 네이티브 캡처는 다음 사용을 위해 일시정지 상태로 남긴다(클래스 설명 참고)."""
        with self.condition:
            if self.stopped:
                return
            self.stopped = True
            self.paused = True
            self.condition.notify_all()
