import queue
import threading
import unittest

import numpy as np

from faster_whisper_stt.audio import AudioChunk, ChunkClock, to_mono_float32
from faster_whisper_stt.microphone import MicrophoneCapture
from faster_whisper_stt.runner import WhisperConfig, WhisperLiveTranscription
from integration import IntegrationService


class ClockService(IntegrationService):
    """테스트에서 세션 시계를 직접 정한다."""

    now = 60_000

    def elapsed_ms(self, session_id):
        with self._lock:
            state = self._get(session_id)
            return self.now if state["running"] else state["session"]["duration_ms"]


class ChunkClockTests(unittest.TestCase):
    def test_chunk_clock_contiguous_and_resyncs(self):
        service = ClockService()
        sid = service.start_session("clock")
        clock = ChunkClock(service, sid, 16000, resync_ms=200)
        service.now = 1000
        self.assertEqual(clock.stamp(1600), (900, 1000))
        service.now = 1130  # 도착 지터: 샘플 기준으로 이어 붙인다
        self.assertEqual(clock.stamp(1600), (1000, 1100))
        service.now = 2000  # 크게 밀리면 세션 시계로 다시 맞춘다
        self.assertEqual(clock.stamp(1600), (1900, 2000))
        self.assertEqual(clock.resync_count, 1)
        service.now = 2050  # 샘플 위치가 앞서면 끝이 현재 시각이 되도록 끌어내린다
        self.assertEqual(clock.stamp(1600), (1950, 2050))


class AudioChunkTests(unittest.TestCase):
    def test_int16_stereo_to_mono_float32(self):
        samples = np.array([[32767, -32768], [0, 0]], dtype=np.int16)
        mono = to_mono_float32(AudioChunk("s", 0, 0, 1, 16000, 2, "int16", samples))
        self.assertEqual(mono.dtype, np.float32)
        self.assertEqual(mono.shape, (2,))
        self.assertAlmostEqual(float(mono[1]), 0.0)

    def test_channel_mismatch(self):
        with self.assertRaises(ValueError):
            to_mono_float32(AudioChunk("s", 0, 0, 1, 16000, 2, "float32", np.zeros(10, dtype=np.float32)))

    def test_invalid_times(self):
        with self.assertRaises(ValueError):
            AudioChunk("s", 0, 10, 5, 16000, 1, "float32", np.zeros(1))


class MicrophoneCaptureTests(unittest.TestCase):
    def test_full_queue_drops_chunk_and_leaves_id_gap(self):
        service = ClockService()
        sid = service.start_session("mic")
        capture = MicrophoneCapture(service, sid, max_queue=1)
        block = np.full((1600, 1), 0.1, dtype=np.float32)
        capture._callback(block, 1600, None, None)
        capture._callback(block, 1600, None, None)
        self.assertEqual(capture.dropped_chunks, 1)
        self.assertEqual(capture.chunks.get_nowait().chunk_id, 0)
        capture._callback(block, 1600, None, None)
        # 버린 청크 번호(1)는 건너뛰므로 실행기가 누락을 알 수 있다.
        self.assertEqual(capture.chunks.get_nowait().chunk_id, 2)


class RunnerFailureTests(unittest.TestCase):
    def test_processing_failure_stops_capture_without_closing_session(self):
        stopped = threading.Event()

        class Capture:
            def __init__(self):
                self.chunks = queue.Queue()
                self.chunks.put(object())  # AudioChunk가 아니어서 처리 중 오류가 난다
                self.stop_calls = 0
            def start(self): pass
            def stop(self):
                self.stop_calls += 1
                stopped.set()

        service = IntegrationService()
        sid = service.start_session("test")
        capture = Capture()
        runner = WhisperLiveTranscription(service, sid, WhisperConfig(), source="microphone",
                                          engine=object(), capture=capture)
        self.assertTrue(runner.start())
        self.assertTrue(stopped.wait(2))
        self.assertEqual(service.get_dashboard_snapshot(sid)["session"]["status"], "running")
        runner.stop()
        self.assertEqual(capture.stop_calls, 1)
        self.assertIsNotNone(runner.last_error)
        self.assertFalse(runner.is_alive())
        service.stop_session(sid)

    def test_start_failure_reports_unavailable(self):
        class BrokenCapture:
            chunks = queue.Queue()
            def start(self): raise RuntimeError("no device")
            def stop(self): pass

        service = IntegrationService()
        sid = service.start_session("test")
        runner = WhisperLiveTranscription(service, sid, WhisperConfig(), source="loopback",
                                          engine=object(), capture=BrokenCapture())
        self.assertFalse(runner.start())
        self.assertEqual(service.get_dashboard_snapshot(sid)["status"]["stt"], "unavailable")
        with self.assertRaises(ValueError):
            WhisperLiveTranscription(service, sid, WhisperConfig(), source="speaker")


if __name__ == "__main__":
    unittest.main()
