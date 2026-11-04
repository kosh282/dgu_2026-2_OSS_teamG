import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import numpy as np
from stt_core import Session, STTProcessor, audio_chunks_from_f32

class FakeRecognizer:
    def __init__(self):
        self.n = 0
    def create_stream(self):
        return FakeStream()
    def is_ready(self, stream):
        return False
    def decode_stream(self, stream):
        pass
    def get_result(self, stream):
        return SimpleNamespace(text='안녕하세요' if self.n < 2 else '다음 문장')
    def is_endpoint(self, stream):
        self.n += 1
        return self.n == 2
    def reset(self, stream):
        pass
class FakeStream:
    def accept_waveform(self, rate, samples):
        assert rate == 16000
    def input_finished(self):
        pass

class ContractTests(unittest.TestCase):
    def test_events_timing_and_storage(self):
        sess = Session('test')
        observed = []
        stt = STTProcessor(sess, FakeRecognizer(), observed.append)
        stt.accept(audio_chunks_from_f32(np.zeros(3200), sess.session_id, 0, 0))
        stt.accept(audio_chunks_from_f32(np.zeros(3200), sess.session_id, 1, 3200))
        stt.finish()
        partials = [e for e in observed if e['event_type'] == 'transcript_partial']
        finals = [e for e in observed if e['event_type'] == 'transcript_final']
        self.assertGreaterEqual(len(partials), 1)
        self.assertEqual(len(finals), 2)
        self.assertEqual(finals[0]['end_ms'], 400)
        self.assertEqual(finals[0]['start_ms'], 0)
        self.assertEqual(finals[0]['payload']['is_edited'], False)
        self.assertTrue(all(e['event_type'] == 'transcript_final' for e in sess.events))
        self.assertTrue(all(e['schema_version']=='0.1.0' for e in observed))
        self.assertEqual(partials[0]['payload']['utterance_id'], finals[0]['payload']['utterance_id'])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'session.json'
            sess.save(path)
            result = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(len(result['events']), 2)
            self.assertEqual(result['session']['duration_ms'], 400)

    def test_contract_validation(self):
        sess = Session('test')
        stt = STTProcessor(sess, FakeRecognizer())
        bad = audio_chunks_from_f32(np.zeros(3200), sess.session_id, 0, 0)
        bad.sample_rate_hz = 44100
        with self.assertRaises(ValueError):
            stt.accept(bad)

if __name__ == '__main__':
    unittest.main()
