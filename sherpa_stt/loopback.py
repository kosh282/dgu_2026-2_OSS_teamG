"""WASAPI 오디오 루프백 입력을 `AudioChunk` 메모리 큐로 전달한다.

선택 프로세스 ID가 있으면 해당 프로세스 트리만 캡처하고, 없으면 콘솔 데모용 SoundCard
기본 출력 루프백을 사용한다. 캡처 오디오는 파일에 저장하지 않는다.
"""
import queue
import subprocess
import sys
import threading
import warnings
from pathlib import Path

from .audio import AudioChunk, ChunkClock


class ProcessLoopbackRecorder:
    """선택 프로세스와 자식 프로세스의 WASAPI 출력을 PCM16 mono로 받는다."""

    def __init__(self, process_id, sample_rate_hz=16000):
        if sample_rate_hz != 16000:
            raise ValueError("프로세스 루프백은 현재 16kHz 입력만 지원합니다")
        self.process_id = int(process_id)
        self.sample_rate_hz = sample_rate_hz
        self.discontinuity_count = 0
        self._process = None
        self._ready = threading.Event()
        self._status_lock = threading.Lock()
        self._error = None

    def __enter__(self):
        if sys.platform != "win32":
            raise RuntimeError("창 단위 시스템 오디오 캡처는 Windows에서만 지원합니다")
        helper = Path(__file__).resolve().parent / "native" / "process_loopback.exe"
        if not helper.is_file():
            raise RuntimeError(
                "창 오디오 캡처 도우미가 없습니다. sherpa_stt/native/build_process_loopback.ps1을 실행하세요"
            )
        self._process = subprocess.Popen(
            [str(helper), str(self.process_id)], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        threading.Thread(target=self._read_status, name="process-loopback-status", daemon=True).start()
        if not self._ready.wait(10):
            self.stop()
            raise RuntimeError("선택한 창의 오디오 캡처를 시작하지 못했습니다 (10초 시간 초과)")
        if self._error:
            message = self._error
            self.stop()
            raise RuntimeError(f"선택한 창의 오디오 캡처를 시작하지 못했습니다: {message}")
        return self

    def _read_status(self):
        for raw in self._process.stderr:
            line = raw.decode("utf-8", errors="replace").strip()
            if line == "READY":
                self._ready.set()
            elif line == "DISCONTINUITY":
                self.discontinuity_count += 1
            elif line.startswith("ERROR"):
                with self._status_lock:
                    self._error = line
                self._ready.set()

    def record(self, numframes):
        if self._process is None:
            raise RuntimeError("오디오 캡처가 시작되지 않았습니다")
        remaining = numframes * 2
        blocks = []
        while remaining:
            block = self._process.stdout.read(remaining)
            if not block:
                with self._status_lock:
                    error = self._error
                code = self._process.poll()
                raise RuntimeError(error or f"프로세스 오디오 스트림이 종료되었습니다 (code={code})")
            blocks.append(block)
            remaining -= len(block)
        import numpy as np
        return np.frombuffer(b"".join(blocks), dtype=np.int16).astype(np.float32).reshape(-1, 1) / 32768.0

    def stop(self):
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()

    def __exit__(self, *_):
        self.stop()


def _co_initialize():
    # 새 스레드에서 WASAPI를 쓰려면 COM 초기화가 필요하다.
    try:
        import ctypes
        ctypes.windll.ole32.CoInitializeEx(None, 0)
    except (AttributeError, OSError):
        pass


class LoopbackCapture:
    """`MicrophoneCapture`와 같은 인터페이스(`chunks`, `start`, `stop`)를 가진 루프백 입력."""

    def __init__(self, service, session_id, sample_rate_hz=16000, chunk_ms=100, speaker=None,
                 process_id=None, max_queue=600):
        self._session_id = session_id
        self._sample_rate_hz = sample_rate_hz
        self._blocksize = max(1, sample_rate_hz * chunk_ms // 1000)
        self._speaker = speaker
        self._process_id = process_id
        self._clock = ChunkClock(service, session_id, sample_rate_hz)
        self._queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self._recorder = None
        self._next_id = 0
        self.error = None
        self.dropped_chunks = 0
        self.discontinuity_count = 0

    @property
    def chunks(self):
        return self._queue

    def _open_recorder(self):
        if self._process_id is not None:
            return ProcessLoopbackRecorder(self._process_id, self._sample_rate_hz)
        import soundcard as sc
        speaker = sc.default_speaker() if self._speaker is None else sc.get_speaker(self._speaker)
        device = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        return device.recorder(samplerate=self._sample_rate_hz, channels=1, blocksize=self._blocksize)

    def _run(self):
        _co_initialize()
        try:
            with self._open_recorder() as recorder:
                self._recorder = recorder
                self._ready.set()
                while not self._stop.is_set():
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        data = recorder.record(numframes=self._blocksize)
                    self.discontinuity_count = getattr(recorder, "discontinuity_count", self.discontinuity_count)
                    # SoundCard는 데이터 불연속을 경고로 알린다.
                    if any("discontinuity" in str(w.message).lower() for w in caught):
                        self.discontinuity_count += 1
                    frames = len(data)
                    if frames == 0:
                        continue
                    start_ms, end_ms = self._clock.stamp(frames)
                    chunk = AudioChunk(self._session_id, self._next_id, start_ms, end_ms,
                                       self._sample_rate_hz, 1, "float32", data[:, 0].copy())
                    self._next_id += 1
                    try:
                        self._queue.put_nowait(chunk)
                    except queue.Full:
                        self.dropped_chunks += 1
        except Exception as error:  # 장치 없음, 장치 변경, 드라이버 오류 등
            if not self._stop.is_set():
                self.error = error
        finally:
            self._recorder = None
            self._ready.set()

    def start(self, timeout=12.0):
        if self._process_id is None:
            try:
                import soundcard  # noqa: F401
            except ImportError as error:
                raise ImportError("SoundCard가 설치되지 않았습니다. sherpa_stt/requirements.txt를 설치하세요") from error
        self._thread = threading.Thread(target=self._run, name="sherpa-stt-loopback", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            self.stop()
            raise RuntimeError(f"루프백 장치 준비 시간 초과 ({timeout:g}초)")
        if self.error is not None:
            raise RuntimeError(f"루프백 장치를 열지 못했습니다: {self.error}")

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def stop(self):
        self._stop.set()
        if self._recorder is not None:
            # SoundCard 녹음기는 stop이 없으며 컨텍스트를 벗어날 때 닫힌다.
            stop_recorder = getattr(self._recorder, 'stop', None)
            if callable(stop_recorder):
                stop_recorder()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


def list_output_devices():
    """루프백으로 받을 수 있는 출력 장치 이름 목록."""
    import soundcard as sc
    return [speaker.name for speaker in sc.all_speakers()]
