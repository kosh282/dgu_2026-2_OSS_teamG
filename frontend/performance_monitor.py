"""수업 내용 없이 프로세스와 실시간 처리 지표만 남기는 선택형 진단 로그."""
import ctypes
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def process_memory_bytes():
    """Windows 프로세스 작업 집합. 사용할 수 없는 환경에서는 null이다."""
    if os.name != 'nt':
        return None
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    try:
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        data = Counters()
        data.cb = ctypes.sizeof(data)
        return int(data.WorkingSetSize) if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(data), data.cb) else None
    except (OSError, AttributeError):
        return None


class PerformanceMonitor:
    """일정 간격 JSONL 기록. 실패는 진단에만 표시하고 수업을 중단하지 않는다."""
    def __init__(self, live, directory, interval_s=5):
        self.live = live
        self.directory = Path(directory)
        self.interval_s = interval_s
        self.path = self.directory / f"performance-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}.jsonl"
        self.error = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        # 실패 여부를 시작 버튼에서 확인하고, 우연한 기존 파일 덮어쓰기를 막는다.
        with self.path.open('x', encoding='utf-8'):
            pass
        self._thread = threading.Thread(target=self._run, name='performance-monitor', daemon=True)
        self._thread.start()

    def _run(self):
        last_wall, last_cpu = time.monotonic(), time.process_time()
        try:
            with self.path.open('a', encoding='utf-8') as handle:
                while not self._stop.wait(self.interval_s):
                    now, cpu = time.monotonic(), time.process_time()
                    elapsed = now - last_wall
                    row = self.live.performance_snapshot()
                    row.update({'log_version': 1, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                                'sample_interval_s': round(elapsed, 3),
                                # 100%는 CPU 논리 코어 하나를 가득 사용한 값이다.
                                'process_cpu_percent_one_core': round(100 * (cpu-last_cpu) / elapsed, 1) if elapsed > 0 else None,
                                'process_working_set_bytes': process_memory_bytes()})
                    last_wall, last_cpu = now, cpu
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    handle.flush()
        except Exception:
            self.error = '성능 로그 기록 실패: 경로와 디스크 상태를 확인하세요.'

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()
