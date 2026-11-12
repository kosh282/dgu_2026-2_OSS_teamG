import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from frontend.live_session import LiveSession
from frontend.performance_monitor import PerformanceMonitor


class PerformanceMonitorTests(unittest.TestCase):
    def test_snapshot_excludes_transcript_video_and_model_values(self):
        live = LiveSession(None, video_enabled=False, audio_source='microphone')
        live.sid = live.service.start_session('private lesson title')
        event = live.service.make_event(live.sid, 'transcript_final', {
            'utterance_id': str(uuid4()), 'revision': 1, 'text': 'PRIVATE TRANSCRIPT', 'is_edited': False,
        }, 0, 0, 'speech')
        live.service.receive_event(event)
        live.latest = {'start_ms': 0, 'features': {'faces': [{'secret': 'PRIVATE FACE'}],
                       'sampling': {'actual_face_hz': 9.5, 'source_size': {'width': 1920, 'height': 1080}}}}
        live.transcription = Mock()
        live.transcription.stats.return_value = {'rtf': 0.2, 'queue_size': 0, 'private': 'PRIVATE AUDIO'}
        serialized = json.dumps(live.performance_snapshot())
        self.assertNotIn('PRIVATE', serialized)
        self.assertNotIn('private lesson title', serialized)
        self.assertEqual(live.performance_snapshot()['transcript']['final_count'], 1)
        self.assertEqual(live.performance_snapshot()['video']['face_hz'], 9.5)
        live.service.stop_session(live.sid)

    def test_monitor_records_and_stops_without_writing_private_content(self):
        live = Mock()
        live.performance_snapshot.return_value = {'session_elapsed_ms': 100, 'stt': {'rtf': 0.1}}
        with tempfile.TemporaryDirectory() as folder:
            monitor = PerformanceMonitor(live, folder, interval_s=0.02)
            with patch('frontend.performance_monitor.process_memory_bytes', return_value=1024):
                monitor.start()
                deadline = time.monotonic() + 2
                while not monitor.path.read_text(encoding='utf-8') and time.monotonic() < deadline:
                    time.sleep(0.01)
                monitor.stop()
            self.assertFalse(monitor.is_running())
            rows = [json.loads(line) for line in monitor.path.read_text(encoding='utf-8').splitlines()]
            self.assertTrue(rows)
            self.assertEqual(rows[0]['process_working_set_bytes'], 1024)
            self.assertGreaterEqual(rows[0]['process_cpu_percent_one_core'], 0)
            self.assertEqual(rows[0]['stt']['rtf'], 0.1)


if __name__ == '__main__':
    unittest.main()
