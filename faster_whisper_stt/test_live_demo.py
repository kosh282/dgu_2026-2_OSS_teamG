import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from faster_whisper_stt.live_demo import RollingWhisper
from sherpa_stt.audio import AudioChunk


class FakeModel:
    def __init__(self):
        self.calls = 0
        self.called = threading.Event()

    def transcribe(self, audio, **kwargs):
        self.calls += 1
        self.called.set()
        word = SimpleNamespace(start=0.1, end=0.3, word=' 안녕')
        return [SimpleNamespace(text=' 안녕', words=[word])], None


class RollingWhisperTests(unittest.TestCase):
    def test_does_not_confirm_by_reprocessing_identical_audio(self):
        model = FakeModel()
        service = SimpleNamespace(elapsed_ms=lambda _: 3000)
        worker = RollingWhisper(service, 's', model, sample_rate=10, window_s=8, update_s=0.05)
        worker.accept(AudioChunk('s', 0, 0, 1000, 10, 1, 'float32', np.zeros(10, dtype=np.float32)))
        worker.start()
        try:
            self.assertTrue(model.called.wait(2))
            time.sleep(0.2)
            self.assertEqual(model.calls, 1)
            self.assertEqual(worker.merger.final_text(), '')
            self.assertEqual(worker.merger.pending_text(), '안녕')
        finally:
            worker.stop()


if __name__ == '__main__':
    unittest.main()
