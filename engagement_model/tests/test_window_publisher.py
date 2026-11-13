import tempfile
import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

from engagement_model.config import ModelConfig
from engagement_model.window_publisher import WindowPublisher
from integration import IntegrationService


class FakeRuntime:
    model_id = "gaussian-head-direction"
    model_version = "angle-time-0.4"
    def __init__(self):
        self.config = ModelConfig(window_seconds=1, output_interval_seconds=0.5)

    def __call__(self, feature):
        return {'angles': {} if feature['observable'] else None, 'score': feature['score'],
                'gaussian_observation': (None, None, feature['score']) if feature['score'] is not None else None}


class WindowPublisherTests(unittest.TestCase):
    def setUp(self):
        self.clock = patch('integration.service.time.monotonic_ns', return_value=0)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.service = IntegrationService()
        self.sid = self.service.start_session('test', {'config_version': 'test', 'participation_model': {}})
        self.service.elapsed_ms = lambda sid: 10000
        self.publisher = WindowPublisher(self.service, self.sid, FakeRuntime())

    def test_default_windows_publish_every_second_after_first_ten_seconds(self):
        runtime = FakeRuntime()
        runtime.config = ModelConfig()
        publisher = WindowPublisher(self.service, self.sid, runtime)
        self.service.elapsed_ms = lambda sid: 12000
        publisher.advance(12000)
        events = self.service.get_dashboard_snapshot(self.sid)['engagement_windows']
        self.assertEqual([(e['start_ms'], e['end_ms']) for e in events],
                         [(0, 10000), (1000, 11000), (2000, 12000)])

    def test_time_weighted_window_and_contract(self):
        for ms in range(0, 1001, 100):
            self.publisher({'start_ms': ms, 'observable': True, 'score': 80 if ms < 500 else 60})
        event = self.service.get_dashboard_snapshot(self.sid)['engagement_windows'][0]
        self.assertEqual((event['start_ms'], event['end_ms']), (0, 1000))
        self.assertEqual(event['payload']['model_output']['value'], 65)
        self.assertIsNone(event['payload']['observation']['eligible_count'])
        self.assertEqual(event['quality']['status'], 'valid')

    def test_long_gap_is_not_filled_with_stale_score(self):
        self.publisher({'start_ms': 0, 'observable': True, 'score': 90})
        self.publisher.advance(1500)
        events = self.service.get_dashboard_snapshot(self.sid)['engagement_windows']
        self.assertEqual(len(events), 2)
        self.assertIsNone(events[0]['payload']['model_output'])
        self.assertEqual(events[0]['quality']['status'], 'not_observable')
        self.assertEqual(events[0]['payload']['model_extensions']['gaussian-head-direction']['scoreable_duration_ms'], 300)
        self.assertIsNone(events[1]['payload']['model_output'])

    def test_finish_partial_window_once(self):
        for ms in range(0, 601, 100):
            self.publisher({'start_ms': ms, 'observable': True, 'score': 80})
        self.publisher.finish(600)
        self.publisher.finish(600)
        events = self.service.get_dashboard_snapshot(self.sid)['engagement_windows']
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['quality']['status'], 'partial')
        self.assertTrue(events[0]['payload']['model_extensions']['gaussian-head-direction']['partial_window'])

    def test_transcript_and_model_survive_shared_session_round_trip(self):
        for ms in range(0, 1001, 100):
            self.publisher({'start_ms': ms, 'observable': True, 'score': 75})
        utterance = str(uuid4())
        for kind, payload in [
            ('transcript_partial', {'utterance_id': utterance, 'revision': 1, 'text': '안녕'}),
            ('transcript_final', {'utterance_id': utterance, 'revision': 1, 'text': '안녕하세요', 'is_edited': False}),
        ]:
            self.service.receive_event(self.service.make_event(self.sid, kind, payload, 200, 800, 'speech'))
        self.service.stop_session(self.sid)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session.json'
            self.service.save_session(self.sid, path)
            restored = IntegrationService()
            sid = restored.load_session(path)
            snapshot = restored.get_dashboard_snapshot(sid)
        self.assertEqual(snapshot['transcripts'][0]['payload']['text'], '안녕하세요')
        self.assertEqual(snapshot['engagement_windows'][0]['payload']['model_output']['value'], 75)
        self.assertEqual(snapshot['partial_transcripts'], [])

    def test_restart_window_does_not_cover_previous_gap(self):
        publisher = WindowPublisher(self.service, self.sid, FakeRuntime(), start_ms=5000)
        for ms in range(5000, 6001, 100):
            publisher({'start_ms': ms, 'observable': True, 'score': 80})
        event = self.service.get_dashboard_snapshot(self.sid)['engagement_windows'][0]
        self.assertEqual((event['start_ms'], event['end_ms']), (5000, 6000))


if __name__ == '__main__':
    unittest.main()
