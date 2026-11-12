import unittest
from unittest.mock import patch
from engagement_model.gaussian import GaussianRuntime, gaussian_window_score
from engagement_model.multi_face import MultiFaceGaussianRuntime
from engagement_model.tests.test_runtime import feature_window
from engagement_model.window_publisher import WindowPublisher
from integration import IntegrationService

class DownwardPolicyTests(unittest.TestCase):
    def test_five_second_grace_then_ten_second_decay_and_recovery(self):
        model = GaussianRuntime()
        for t in range(0, 10100, 100): model(feature_window(t))
        scores = {}
        for t in range(10100, 25200, 100):
            result = model(feature_window(t, pitch=20))
            if t in (10100, 15100, 20100, 25100): scores[t] = result['current_direction_score']
        self.assertEqual(scores, {10100:100.0, 15100:100.0, 20100:90.0, 25100:80.0})
        self.assertIn('looking_down_score_decay', result['reason_codes'])
        self.assertEqual(model(feature_window(25200))['current_direction_score'], 100)

    def test_no_previous_valid_score_does_not_invent_one(self):
        model = GaussianRuntime()
        for t in range(0, 2000, 100): result = model(feature_window(t, pitch=20))
        self.assertIsNone(result['current_direction_score'])
        self.assertIsNone(result['gaussian_observation'])

    def test_downward_observations_are_included_in_single_and_group_windows(self):
        single = GaussianRuntime()
        segments = [(0, 10000, True, None, (None, None, 80.0))]
        self.assertEqual(gaussian_window_score(segments, 0, 10000, single.config)['score'], 80.0)
        multi = MultiFaceGaussianRuntime()
        group = [(0, 10000, True, None, [(None, None, 80.0), (0.0, 0.0)])]
        self.assertEqual(multi.aggregate_window(group, 0, 10000)['score'], 90.0)

    def test_publisher_emits_downward_decay_as_actual_window_score(self):
        service = IntegrationService()
        sid = service.start_session('test')
        runtime = GaussianRuntime()
        publisher = WindowPublisher(service, sid, runtime, start_ms=0)
        for t in range(0, 30001, 100):
            feature = feature_window(t, pitch=20 if t >= 10100 else 0)
            feature['session_id'] = sid
            with patch.object(service, 'elapsed_ms', return_value=t):
                publisher(feature)
        events = [e for e in service.get_events_in_range(sid, 0, 30000)
                  if e['event_type'] == 'engagement_window']
        self.assertTrue(events)
        self.assertLess(events[-1]['payload']['model_output']['value'], 90)
        self.assertEqual(events[-1]['provenance']['model_version'], 'angle-time-0.4')

if __name__ == '__main__': unittest.main()
