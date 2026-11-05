"""Windows 출력 장치 루프백 입력: 수업 프로그램·영상에서 재생되는 소리를 받는다.

SoundCard(WASAPI loopback)를 사용한다. 선택한 창의 소리만 분리하지 않고
해당 출력 장치(기본: 기본 스피커)로 나가는 **모든 소리**를 받는다.
샘플은 메모리 큐로만 전달하고 파일로 저장하지 않는다.
"""
import queue
import threading
import warnings

from .audio import AudioChunk, ChunkClock


def _co_initialize():
    # 새 스레드에서 WASAPI를 쓰려면 COM 초기화가 필요하다.
    try:
        import ctypes
        ctypes.windll.ole32.CoInitializeEx(None, 0)
    except (AttributeError, OSError):
        pass


class LoopbackCapture:
    """`MicrophoneCapture`와 같은 인터페이스(`chunks`, `start`, `stop`)를 가진 루프백 입력."""

    def __init__(self, service, session_id, sample_rate_hz=16000, chunk_ms=100, speaker=None, max_queue=600):
        self._session_id = session_id
        self._sample_rate_hz = sample_rate_hz
        self._blocksize = max(1, sample_rate_hz * chunk_ms // 1000)
        self._speaker = speaker
        self._clock = ChunkClock(service, session_id, sample_rate_hz)
        self._queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self._next_id = 0
        self.error = None
        self.dropped_chunks = 0
        self.discontinuity_count = 0

    @property
    def chunks(self):
        return self._queue

    def _open_recorder(self):
        import soundcard as sc
        speaker = sc.default_speaker() if self._speaker is None else sc.get_speaker(self._speaker)
        device = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        return device.recorder(samplerate=self._sample_rate_hz, channels=1, blocksize=self._blocksize)

    def _run(self):
        _co_initialize()
        try:
            with self._open_recorder() as recorder:
                self._ready.set()
                while not self._stop.is_set():
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        data = recorder.record(numframes=self._blocksize)
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
            self.error = error
        finally:
            self._ready.set()

    def start(self, timeout=5.0):
        try:
            import soundcard  # noqa: F401
        except ImportError as error:
            raise ImportError("SoundCard가 설치되지 않았습니다. sherpa_stt/requirements.txt를 설치하세요") from error
        self._thread = threading.Thread(target=self._run, name="sherpa-stt-loopback", daemon=True)
        self._thread.start()
        self._ready.wait(timeout)
        if self.error is not None:
            raise RuntimeError(f"루프백 장치를 열지 못했습니다: {self.error}")

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


def list_output_devices():
    """루프백으로 받을 수 있는 출력 장치 이름 목록."""
    import soundcard as sc
    return [speaker.name for speaker in sc.all_speakers()]
