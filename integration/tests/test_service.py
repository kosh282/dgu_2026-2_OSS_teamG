"""저장 경계와 스냅샷 격리 회귀 검사. 실제 수업 데이터는 사용하지 않는다."""
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from integration import IntegrationService
from integration.contracts import EventValidationError
from integration.service import IntegrationError


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = IntegrationService()
        self.sid = self.service.start_session('합성 수업')
        self.service.stop_session(self.sid)

    def saved_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session.json'
            self.service.save_session(self.sid, path)
            return json.loads(path.read_text(encoding='utf-8'))

    def load_data(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            reader = IntegrationService()
            reader.load_session(path)
            return reader.get_dashboard_snapshot(self.sid)

    def test_start_rejects_raw_data_in_nested_config_including_tuple(self):
        for nested in ([{'audio': [0.1]}], ({'frame': '원본'},)):
            with self.subTest(nested=nested), self.assertRaises(EventValidationError):
                IntegrationService().start_session('수업', {'config_version': 'test', 'extra': nested})

    def test_load_rejects_raw_data_outside_events(self):
        for section in ('config', 'session'):
            data = self.saved_data()
            data[section]['extra'] = {'samples': [0.1]}
            with self.subTest(section=section), self.assertRaises(EventValidationError):
                self.load_data(data)

    def test_load_rejects_non_finite_config_before_registering_session(self):
        for value in (float('nan'), float('inf')):
            data = self.saved_data()
            data['config']['extra'] = value
            with self.subTest(value=value), self.assertRaises(IntegrationError):
                self.load_data(data)

    def test_valid_legacy_session_and_unknown_config_fields_round_trip(self):
        data = self.saved_data()
        data['schema_version'] = '0.1.0'
        data['config']['extra'] = {'enabled': True, 'value': None}
        snapshot = self.load_data(data)
        self.assertEqual(snapshot['session']['session_id'], self.sid)
        self.assertEqual(snapshot['session']['status'], 'stopped')

    def test_snapshot_is_detached_including_evidence(self):
        utterance = str(uuid4())
        original = self.service.make_event(self.sid, 'note_created',
            {'text': '메모', 'linked_event_ids': []}, 0, 0, 'ui')
        self.service.receive_event(original)
        feedback = self.service.make_event(self.sid, 'post_session_feedback',
            {'text': '초안', 'status': 'draft', 'evidence_event_ids': [original['event_id']]}, 0, 0, 'assistant')
        self.service.receive_event(feedback)
        before = self.service.get_dashboard_snapshot(self.sid)
        changed = self.service.get_dashboard_snapshot(self.sid)
        changed['session']['title'] = '변경'
        changed['notes'][0]['payload']['linked_event_ids'].append(utterance)
        changed['feedback'][0]['payload']['evidence_event_ids'].clear()
        changed['feedback'][0]['evidence_needs_refresh'] = True
        self.assertEqual(before, self.service.get_dashboard_snapshot(self.sid))

    def test_tuple_cannot_bypass_event_raw_data_guard(self):
        event = self.service.make_event(self.sid, 'note_created',
            {'text': '', 'linked_event_ids': [], 'extra': ({'video': '원본'},)}, 0, 0, 'ui')
        with self.assertRaises(EventValidationError):
            self.service.receive_event(event)


if __name__ == '__main__':
    unittest.main()
