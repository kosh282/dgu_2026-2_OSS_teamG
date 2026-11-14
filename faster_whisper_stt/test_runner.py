import queue
import unittest
from types import SimpleNamespace

import numpy as np

from faster_whisper_stt.runner import WhisperConfig, WhisperLiveTranscription
from integration import IntegrationService
from sherpa_stt.audio import AudioChunk


class FakeCapture:
    def __init__(self, chunks):
        self.chunks = queue.Queue()
        for chunk in chunks:
            self.chunks.put(chunk)
        self.error = None
        self.dropped_chunks = self.overflow_count = self.discontinuity_count = 0

    def start(self): pass
    def stop(self): pass


class FakeWhisper:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(audio.copy())
        return [SimpleNamespace(text=' 테스트')], None


class WhisperRunnerTests(unittest.TestCase):
    def test_invalid_window_is_rejected_before_processing_can_spin(self):
        for value in (0, -1, True, 0.5, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                WhisperConfig(window_seconds=value)

    def test_invalid_capture_or_model_settings_fail_before_loading(self):
        for settings in ({'sample_rate_hz': 8000}, {'chunk_ms': 0},
                         {'num_threads': 0}, {'model': ''}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                WhisperConfig(**settings)

    def test_ten_second_events_and_shutdown_tail(self):
        service = IntegrationService()
        sid = service.start_session('test', {'config_version': 'test', 'participation_model': None, 'stt_model': None})
        service.elapsed_ms = lambda _: 20000
        data = np.linspace(-0.5, 0.5, 168000, dtype=np.float32)
        chunks = [AudioChunk(sid, i, i * 100, (i + 1) * 100, 16000, 1, 'float32', data[i*1600:(i+1)*1600])
                  for i in range(105)]
        model = FakeWhisper()
        runner = WhisperLiveTranscription(service, sid, WhisperConfig(), source='microphone',
                                          capture=FakeCapture(chunks), engine=model)
        self.assertTrue(runner.start())
        runner.stop()
        self.assertIsNone(runner.last_error)
        self.assertEqual([len(x) for x in model.calls], [160000, 8000])
        np.testing.assert_array_equal(np.concatenate(model.calls), data)
        events = service.get_dashboard_snapshot(sid)['transcripts']
        self.assertEqual([(e['start_ms'], e['end_ms']) for e in events], [(0, 10000), (10000, 10500)])
        self.assertEqual([e['payload']['text'] for e in events], ['테스트', '테스트'])
        self.assertTrue(all(e['provenance']['model_id'] == 'faster-whisper-base' for e in events))
        self.assertEqual(service.get_dashboard_snapshot(sid)['partial_transcripts'], [])
        stats = runner.stats()
        self.assertEqual((stats['processed_windows'], stats['empty_windows']), (2, 0))
        self.assertIsNotNone(stats['capture_level_dbfs'])

    def test_silent_input_and_empty_output_are_counted(self):
        service = IntegrationService()
        sid = service.start_session('test')
        service.elapsed_ms = lambda _: 20000
        chunk = AudioChunk(sid, 0, 0, 10000, 16000, 1, 'float32', np.zeros(160000, np.float32))
        class EmptyModel:
            def transcribe(self, audio, **kwargs):
                return [], None
        runner = WhisperLiveTranscription(service, sid, WhisperConfig(), source='microphone',
                                          capture=FakeCapture([chunk]), engine=EmptyModel())
        self.assertTrue(runner.start())
        runner.stop()
        stats = runner.stats()
        self.assertEqual((stats['processed_windows'], stats['empty_windows']), (1, 1))
        self.assertEqual(stats['capture_level_dbfs'], -200.0)
        self.assertEqual(service.get_dashboard_snapshot(sid)['transcripts'], [])


if __name__ == '__main__':
    unittest.main()
