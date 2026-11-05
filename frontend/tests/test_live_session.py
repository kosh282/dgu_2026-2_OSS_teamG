import threading
import time
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

from frontend.live_session import LiveSession


class FakeTranscription:
    def __init__(self, service, sid, *args, **kwargs):
        self.service, self.sid = service, sid
        self.stopped = False
        self.last_error = None

    def start(self):
        return True

    def is_alive(self):
        return not self.stopped

    def emit_final(self):
        event = self.service.make_event(
            self.sid, 'transcript_final',
            {'utterance_id': str(uuid4()), 'revision': 1, 'text': '수업 전사', 'is_edited': False},
            0, 0, 'speech',
        )
        self.service.receive_event(event)

    def stop(self):
        self.emit_final()
        self.stopped = True


class LiveSessionIsolationTests(unittest.TestCase):
    def test_video_failure_does_not_stop_stt_or_session(self):
        failure = threading.Event()

        def fail_video(*args, **kwargs):
            failure.set()
            raise RuntimeError('camera failed')

        with patch('sherpa_stt.engine.SherpaEngine', return_value=object()), \
             patch('sherpa_stt.runner.LiveTranscription', FakeTranscription), \
             patch('frontend.live_session.RETRY_DELAYS_SECONDS', (0.01, 0.01, 0.01)), \
             patch('frontend.live_session.FaceAnalyzer', side_effect=fail_video):
            live = LiveSession(123, audio_source='microphone')
            live.start('test')
            self.addCleanup(lambda: live.stop() if live.is_running() else None)
            self.assertTrue(failure.wait(2))
            live.vision_thread.join(2)
            self.assertTrue(live.is_running())
            self.assertFalse(live.transcription.stopped)
            self.assertEqual(live.service.get_dashboard_snapshot(live.sid)['status']['vision'], 'unavailable')
            live.transcription.emit_final()
            live.stop()
            snapshot = live.service.get_dashboard_snapshot(live.sid)
            self.assertEqual(len(snapshot['transcripts']), 2)
            self.assertEqual(snapshot['session']['status'], 'stopped')

    def test_stt_failure_keeps_video_and_session_running(self):
        started = threading.Event()

        class FakeResource:
            def __init__(self, *args, **kwargs): pass
            def close(self): pass

        class FakePipeline(FakeResource):
            def process_one(self):
                started.set()
                live.stop_event.wait(0.01)
                return None

        with patch('sherpa_stt.runner.LiveTranscription', side_effect=RuntimeError('STT failed')), \
             patch('frontend.live_session.FaceAnalyzer', FakeResource), \
             patch('frontend.live_session.PoseAnalyzer', FakeResource), \
             patch('frontend.live_session.WindowCapture', FakeResource), \
             patch('frontend.live_session.PreviewCapture', FakeResource), \
             patch('frontend.live_session.VisionPipeline', FakePipeline), \
             patch.object(LiveSession, '_close_developer_windows'):
            live = LiveSession(123, audio_source='microphone')
            live.start('test')
            self.addCleanup(lambda: live.stop() if live.is_running() else None)
            self.assertTrue(started.wait(2))
            self.assertTrue(live.is_running())
            deadline = time.monotonic() + 2
            while 'stt' not in live.service.get_dashboard_snapshot(live.sid)['status'] and time.monotonic() < deadline:
                threading.Event().wait(0.01)
            self.assertEqual(live.service.get_dashboard_snapshot(live.sid)['status']['stt'], 'unavailable')
            live.stop()

    def test_model_error_does_not_propagate_to_video(self):
        live = LiveSession(123)
        live.sid = live.service.start_session('test', {'config_version': 'test', 'participation_model': {}})
        class FailingPublisher:
            def __call__(self, feature):
                raise RuntimeError('bad model')
        live.window_publisher = FailingPublisher()
        live._consume_model({'start_ms': 0})
        live._consume_model({'start_ms': 1})
        self.assertTrue(live._model_failed)
        self.assertIsNone(live.error)
        self.assertEqual(live.service.get_dashboard_snapshot(live.sid)['status']['model'], 'unavailable')

    def test_video_processing_continues_after_model_failure(self):
        processed_again = threading.Event()

        class FakeResource:
            def __init__(self, *args, **kwargs): pass
            def close(self): pass

        class BrokenPublisher:
            def __init__(self, *args): pass
            def __call__(self, feature): raise RuntimeError('model failed')

        class FakePipeline(FakeResource):
            def __init__(self, *args, **kwargs):
                self.consumer = kwargs['model_consumer']
                self.count = 0
            def process_one(self):
                self.count += 1
                self.consumer({'start_ms': 0})
                if self.count > 1:
                    processed_again.set()
                live.stop_event.wait(0.01)
                return None

        with patch('frontend.live_session.FaceAnalyzer', FakeResource), \
             patch('frontend.live_session.PoseAnalyzer', FakeResource), \
             patch('frontend.live_session.WindowCapture', FakeResource), \
             patch('frontend.live_session.PreviewCapture', FakeResource), \
             patch('frontend.live_session.WindowPublisher', BrokenPublisher), \
             patch('frontend.live_session.VisionPipeline', FakePipeline), \
             patch.object(LiveSession, '_close_developer_windows'):
            live = LiveSession(123)
            live.start('test')
            self.addCleanup(lambda: live.stop() if live.is_running() else None)
            self.assertTrue(processed_again.wait(2))
            self.assertTrue(live.is_running())
            self.assertIsNone(live.error)
            self.assertEqual(live.service.get_dashboard_snapshot(live.sid)['status']['model'], 'unavailable')
            live.stop()

    def test_model_initialization_failure_keeps_video_running(self):
        started = threading.Event()

        class FakeResource:
            def __init__(self, *args, **kwargs): pass
            def close(self): pass

        class FakePipeline(FakeResource):
            def __init__(self, *args, **kwargs):
                self.consumer = kwargs['model_consumer']
            def process_one(self):
                started.set()
                live.stop_event.wait(0.01)
                return None

        with patch('frontend.live_session.FaceAnalyzer', FakeResource), \
             patch('frontend.live_session.PoseAnalyzer', FakeResource), \
             patch('frontend.live_session.WindowCapture', FakeResource), \
             patch('frontend.live_session.PreviewCapture', FakeResource), \
             patch('frontend.live_session.EngagementRuntime', side_effect=RuntimeError('model init')), \
             patch('frontend.live_session.VisionPipeline', FakePipeline), \
             patch.object(LiveSession, '_close_developer_windows'):
            live = LiveSession(123)
            live.start('test')
            self.addCleanup(lambda: live.stop() if live.is_running() else None)
            self.assertTrue(started.wait(2))
            self.assertTrue(live.is_running())
            self.assertEqual(live.service.get_dashboard_snapshot(live.sid)['status']['model'], 'unavailable')
            live.stop()

    def test_audio_only_does_not_open_window(self):
        with patch('sherpa_stt.engine.SherpaEngine', return_value=object()), \
             patch('sherpa_stt.runner.LiveTranscription', FakeTranscription), \
             patch('frontend.live_session.WindowCapture') as capture:
            live = LiveSession(None, audio_source='microphone', video_enabled=False)
            live.start('test')
            self.addCleanup(lambda: live.stop() if live.is_running() else None)
            self.assertTrue(live.is_running())
            capture.assert_not_called()
            self.assertEqual(live.service.get_dashboard_snapshot(live.sid)['status']['vision'], 'stopped')
            live.stop()
            self.assertEqual(len(live.service.get_dashboard_snapshot(live.sid)['transcripts']), 1)

    def test_video_restarts_then_session_can_stop(self):
        recovered = threading.Event()
        attempts = []

        def attempt():
            attempts.append(1)
            if len(attempts) == 1:
                live.error = 'temporary capture error'
                return
            recovered.set()
            live.stop_event.wait(2)

        with patch('frontend.live_session.RETRY_DELAYS_SECONDS', (0.01, 0.01, 0.01)), \
             patch.object(LiveSession, '_run_vision', side_effect=attempt):
            live = LiveSession(123, model_enabled=False)
            live.start('test')
            self.addCleanup(lambda: live.stop() if live.is_running() else None)
            self.assertTrue(recovered.wait(2))
            self.assertEqual(len(attempts), 2)
            live.stop()

    def test_retry_limit_and_stop_cancel_wait(self):
        attempted = threading.Event()
        def fail():
            attempted.set()
            raise RuntimeError('unexpected')
        with patch('frontend.live_session.RETRY_DELAYS_SECONDS', (0.01, 0.01, 0.01)), \
             patch.object(LiveSession, '_run_vision', side_effect=fail):
            # 예상 밖 예외도 감독자가 처리하는지 확인한다.
            live = LiveSession(123, model_enabled=False)
            live.start('test')
            self.addCleanup(lambda: live.stop() if live.is_running() else None)
            self.assertTrue(attempted.wait(2))
            live.vision_thread.join(2)
            snapshot = live.service.get_dashboard_snapshot(live.sid)
            self.assertEqual(snapshot['status']['vision'], 'unavailable')
            self.assertIn('횟수 초과', snapshot['status_messages']['vision'])
            live.stop()

    def test_stt_start_failure_then_recovers(self):
        started = threading.Event()
        attempts = []

        class SometimesFails(FakeTranscription):
            def start(self):
                attempts.append(1)
                if len(attempts) == 1:
                    partial = self.service.make_event(
                        self.sid, 'transcript_partial',
                        {'utterance_id': str(uuid4()), 'revision': 1, 'text': '미완성'},
                        0, 0, 'speech',
                    )
                    self.service.receive_event(partial)
                    raise RuntimeError('temporary audio error')
                started.set()
                return True
            def stop(self):
                if self.last_error is None:
                    self.emit_final()
                self.stopped = True

        with patch('frontend.live_session.RETRY_DELAYS_SECONDS', (0.01, 0.01, 0.01)), \
             patch('sherpa_stt.runner.LiveTranscription', SometimesFails):
            live = LiveSession(None, audio_source='microphone', video_enabled=False)
            live.start('test')
            self.addCleanup(lambda: live.stop() if live.is_running() else None)
            self.assertTrue(started.wait(2))
            self.assertEqual(len(attempts), 2)
            self.assertTrue(live.is_running())
            self.assertEqual(live.service.get_dashboard_snapshot(live.sid)['partial_transcripts'], [])
            live.stop()

    def test_model_can_restart_after_failure(self):
        live = LiveSession(123)
        live.sid = live.service.start_session('test', {'config_version': 'test', 'participation_model': {}})
        with patch('frontend.live_session.RETRY_DELAYS_SECONDS', (0.01,)), \
             patch('frontend.live_session.EngagementRuntime', side_effect=[RuntimeError('model failed'), object()]), \
             patch('frontend.live_session.WindowPublisher', return_value=object()):
            live._start_model()
            self.assertTrue(live._model_failed)
            live._model_retry_at = 0
            live._maybe_restart_model()
            self.assertFalse(live._model_failed)
            self.assertEqual(live.service.get_dashboard_snapshot(live.sid)['status']['model'], 'running')
        live.service.stop_session(live.sid)

    def test_stop_interrupts_retry_delay(self):
        attempted = threading.Event()
        def fail():
            attempted.set()
            raise RuntimeError('capture failed')
        with patch('frontend.live_session.RETRY_DELAYS_SECONDS', (30.0,)), \
             patch.object(LiveSession, '_run_vision', side_effect=fail):
            live = LiveSession(123, model_enabled=False)
            live.start('test')
            self.assertTrue(attempted.wait(2))
            started = time.monotonic()
            live.stop()
            self.assertLess(time.monotonic() - started, 2)

    def test_running_checkpoint_can_be_loaded_after_interruption(self):
        from integration import IntegrationService
        with tempfile.TemporaryDirectory() as directory, \
             patch('frontend.live_session.CHECKPOINT_INTERVAL_SECONDS', 0.01):
            live = LiveSession(None, video_enabled=False, recovery_root=directory)
            live.start('복구 수업')
            try:
                event = live.service.make_event(live.sid, 'note_created', {'text': '중간 메모', 'linked_event_ids': []}, 0, 0, 'ui')
                live.service.receive_event(event)
                path = Path(directory) / f'{live.sid}.json'
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    try:
                        reader = IntegrationService()
                        sid = reader.load_session(path)
                        if reader.get_dashboard_snapshot(sid)['notes']:
                            break
                    except (OSError, ValueError):
                        pass
                    threading.Event().wait(0.01)
                else:
                    self.fail('메모가 복구 파일에 기록되지 않았습니다')
                self.assertEqual(reader.get_dashboard_snapshot(sid)['notes'][0]['payload']['text'], '중간 메모')
            finally:
                live.stop()


if __name__ == '__main__':
    unittest.main()
