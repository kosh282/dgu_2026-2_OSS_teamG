import unittest
from types import SimpleNamespace

import numpy as np

from faster_whisper_stt.direct_demo import DirectTranscriber
from faster_whisper_stt.loopback import LoopbackCapture as DemoLoopbackCapture
from integration import IntegrationService
from faster_whisper_stt.audio import AudioChunk


class FakeModel:
    def __init__(self):
        self.samples = []

    def transcribe(self, audio, **kwargs):
        self.samples.append(audio.copy())
        return [SimpleNamespace(text=' 테스트')], None


class DirectTranscriberTests(unittest.TestCase):
    def test_demo_loopback_stops_recorder_without_stop_method(self):
        service = IntegrationService()
        sid = service.start_session('test')
        capture = DemoLoopbackCapture(service, sid)
        capture._recorder = object()
        capture.stop()
        self.assertTrue(capture._stop.is_set())

    def test_audio_is_processed_once_and_tail_is_not_lost(self):
        model = FakeModel()
        worker = DirectTranscriber(model, sample_rate=10, chunk_seconds=2)
        worker.start()
        samples = np.linspace(-0.5, 0.5, 55, dtype=np.float32)
        for idx, start in enumerate(range(0, len(samples), 10)):
            values = samples[start:start + 10]
            worker.accept(AudioChunk('s', idx, start * 100, (start + len(values)) * 100,
                                     10, 1, 'float32', values))
        worker.stop()
        self.assertEqual([len(part) for part in model.samples], [20, 20, 15])
        np.testing.assert_array_equal(np.concatenate(model.samples), samples)

    def test_detects_missing_input(self):
        worker = DirectTranscriber(FakeModel(), sample_rate=10)
        worker.accept(AudioChunk('s', 0, 0, 1000, 10, 1, 'float32', np.zeros(10, np.float32)))
        with self.assertRaisesRegex(RuntimeError, '누락'):
            worker.accept(AudioChunk('s', 2, 2000, 3000, 10, 1, 'float32', np.zeros(10, np.float32)))


if __name__ == '__main__':
    unittest.main()
