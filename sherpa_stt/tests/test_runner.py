import queue
import threading
import unittest
from unittest.mock import patch

from integration import IntegrationService
from sherpa_stt.config import SttConfig
from sherpa_stt.runner import LiveTranscription


class RunnerTests(unittest.TestCase):
    def test_processing_failure_stops_capture_without_closing_session(self):
        stopped = threading.Event()

        class Capture:
            def __init__(self):
                self.chunks = queue.Queue()
                self.chunks.put(object())
                self.stop_calls = 0
            def start(self): pass
            def stop(self):
                self.stop_calls += 1
                stopped.set()

        class Transcriber:
            def __init__(self, *args, **kwargs): pass
            def report_status(self, *args): pass
            def accept_chunk(self, chunk): raise RuntimeError('transcription failed')

        service = IntegrationService()
        sid = service.start_session('test')
        capture = Capture()
        with patch('sherpa_stt.runner.SherpaTranscriber', Transcriber):
            runner = LiveTranscription(service, sid, SttConfig(), source='microphone', engine=object(), capture=capture)
            self.assertTrue(runner.start())
            self.assertTrue(stopped.wait(2))
            self.assertEqual(service.get_dashboard_snapshot(sid)['session']['status'], 'running')
            runner.stop()
            self.assertEqual(capture.stop_calls, 1)
            self.assertIsNotNone(runner.last_error)
        service.stop_session(sid)


if __name__ == '__main__':
    unittest.main()
