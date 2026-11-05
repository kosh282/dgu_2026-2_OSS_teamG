"""sounddevice 기반 마이크 입력. 샘플은 메모리 큐로만 전달하고 파일로 저장하지 않는다."""
import queue

from .audio import AudioChunk, ChunkClock


class MicrophoneCapture:
    """입력 장치에서 일정 길이의 mono float32 청크를 받아 `AudioChunk`로 만든다.

    청크 시각은 `ChunkClock`이 세션 공통 시계 기준으로 붙인다.
    """

    def __init__(self, service, session_id, sample_rate_hz=16000, chunk_ms=100, device=None, max_queue=600):
        self._session_id = session_id
        self._sample_rate_hz = sample_rate_hz
        self._blocksize = max(1, sample_rate_hz * chunk_ms // 1000)
        self._device = device
        self._clock = ChunkClock(service, session_id, sample_rate_hz)
        self._queue = queue.Queue(maxsize=max_queue)
        self._stream = None
        self._next_id = 0
        self.dropped_chunks = 0
        self.overflow_count = 0

    @property
    def chunks(self):
        """소비자가 `get()`으로 꺼낼 `AudioChunk` 큐."""
        return self._queue

    def _callback(self, indata, frames, time_info, status):
        if status and getattr(status, "input_overflow", False):
            self.overflow_count += 1
        start_ms, end_ms = self._clock.stamp(frames)
        chunk = AudioChunk(self._session_id, self._next_id, start_ms, end_ms, self._sample_rate_hz,
                           1, "float32", indata[:, 0].copy())
        self._next_id += 1
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            # 소비가 밀리면 버린다. 버린 구간은 전사기에서 공백으로 감지된다.
            self.dropped_chunks += 1

    def start(self):
        try:
            import sounddevice as sd
        except ImportError as error:
            raise ImportError("sounddevice가 설치되지 않았습니다. sherpa_stt/requirements.txt를 설치하세요") from error
        self._stream = sd.InputStream(samplerate=self._sample_rate_hz, channels=1, dtype="float32",
                                      blocksize=self._blocksize, device=self._device, callback=self._callback)
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None


def list_input_devices():
    """입력 가능한 장치의 (번호, 이름) 목록."""
    import sounddevice as sd
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d.get("max_input_channels", 0) > 0]
