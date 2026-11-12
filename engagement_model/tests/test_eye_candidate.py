import unittest
import json
from unittest.mock import patch
from engagement_model.eye_candidate import EyeCandidate
from engagement_model.gaussian import GaussianRuntime
from engagement_model.multi_face import MultiFaceGaussianRuntime
from engagement_model.tests.test_runtime import feature_window
from engagement_model.window_publisher import WindowPublisher
from integration import IntegrationService


def frame(t, ear=0.3, track=1):
    item = feature_window(t, track_key=track)
    face = item['features']['faces'][0]
    face['eye_aspect_ratio_left'] = ear
    face['eye_aspect_ratio_right'] = ear
    return item


class EyeCandidateTests(unittest.TestCase):
    def test_short_blink_not_penalized_and_long_closure_reduces_candidate(self):
        model = GaussianRuntime()
        for t in range(0, 10001, 100):
            r = model(frame(t, ear=0.1 if 3000 <= t < 3300 else 0.3))
        self.assertEqual(r['eye_candidate']['eye_score'], 100.0)
        self.assertEqual(r['eye_candidate']['candidate_total'], 100.0)
        for t in range(10100, 15101, 100): r = model(frame(t, ear=0.1))
        self.assertLess(r['eye_candidate']['eye_score'], 100)
        self.assertEqual(r['score'], 100.0)
        self.assertLess(r['eye_candidate']['candidate_total'], r['score'])

    def test_missing_or_track_change_does_not_become_closed(self):
        candidate = EyeCandidate()
        head = {'angles': {'pitch': 0}, 'score': 100, 'downward_status': None}
        for t in range(0, 8000, 100): candidate.update(frame(t, 0.1), head)
        missing = frame(8000, None)
        self.assertIsNone(candidate.update(missing, head)['candidate_total'])
        after = candidate.update(frame(8100, 0.3, track=2), head)
        self.assertIsNone(after['candidate_total'])
        self.assertEqual(after['closed_seconds'], 0)

    def test_observation_gaps_and_looking_down_defer_candidate(self):
        candidate = EyeCandidate()
        head = {'angles': {'pitch': 0}, 'score': 100, 'downward_status': None}
        for t in range(0, 12000, 1000): output = candidate.update(frame(t), head)
        self.assertIsNone(output['eye_score'])
        self.assertLess(output['coverage'], .7)
        head['downward_status'] = 'looking_down_confirmed'
        self.assertEqual(candidate.update(frame(12100), head)['status'], 'eye_unavailable')

    def test_group_candidate_requires_every_face_to_be_matched_and_observable(self):
        model = MultiFaceGaussianRuntime()
        for t in range(0, 15001, 100):
            item = feature_window(t, faces=2)
            for i, face in enumerate(item['features']['faces'], 1):
                face['transient_track_key'] = i
                face['head_angles_degrees'] = {'pitch': 0, 'yaw': 0, 'roll': 0}
                face['eye_aspect_ratio_left'] = face['eye_aspect_ratio_right'] = 0.3
            model.update_identities({1: 'a', 2: 'b'})
            result = model(item)
        self.assertEqual(result['eye_candidate']['candidate_total'], 100)
        model.update_identities({1: 'a'})
        item['start_ms'] = item['end_ms'] = 15100
        self.assertIsNone(model(item)['eye_candidate']['candidate_total'])

    def test_comparison_candidate_is_not_in_published_events(self):
        service = IntegrationService()
        sid = service.start_session('test')
        model = GaussianRuntime()
        publisher = WindowPublisher(service, sid, model, start_ms=0)
        for t in range(0, 12001, 100):
            sample = frame(t)
            sample['session_id'] = sid
            with patch.object(service, 'elapsed_ms', return_value=t):
                publisher(sample)
        self.assertIsNotNone(model.latest['eye_candidate']['candidate_total'])
        events = service.get_events_in_range(sid, 0, 12000)
        self.assertNotIn('eye_candidate', json.dumps(events))
        self.assertNotIn('eye_score', json.dumps(events))

if __name__ == '__main__': unittest.main()
