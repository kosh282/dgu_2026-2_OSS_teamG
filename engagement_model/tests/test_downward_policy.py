import unittest
from unittest.mock import patch
from engagement_model.gaussian import GaussianRuntime, gaussian_window_score
from engagement_model.multi_face import MultiFaceGaussianRuntime
from engagement_model.tests.test_runtime import feature_window
from engagement_model.window_publisher import WindowPublisher
from integration import IntegrationService

class DownwardPolicyTests(unittest.TestCase):
    def test_down_threshold_and_five_second_grace(self):
        model = GaussianRuntime()
        for t in range(0, 10100, 100): model(feature_window(t))
        self.assertEqual(model(feature_window(10100, pitch=9))['orientation_state'], 'front')
        first = model(feature_window(10200, pitch=10))
        self.assertEqual(first['orientation_state'], 'looking_down')
        for t in range(10300, 15201, 100): result = model(feature_window(t, pitch=10))
        self.assertEqual(result['current_direction_score'], 100)
        self.assertLess(model(feature_window(15300, pitch=10))['current_direction_score'], 100)
    def test_short_missing_frames_preserve_only_confirmed_down_time(self):
        model = GaussianRuntime()
        for t in range(0, 10100, 100): model(feature_window(t))
        for t in range(10100, 21100, 100):
            item = feature_window(t, pitch=20)
            if t in (12000, 14500, 17000):
                item['features']['faces'][0]['head_angles_relative_degrees'] = None
            result = model(item)
        self.assertGreater(result['looking_down_seconds'], 9)
        self.assertLess(result['current_direction_score'], 80)
        self.assertEqual(model(feature_window(21100))['looking_down_seconds'], 0)

    def test_long_gap_does_not_continue_downward_penalty(self):
        model = GaussianRuntime()
        for t in range(0, 10100, 100): model(feature_window(t))
        for t in range(10100, 18100, 100): result = model(feature_window(t, pitch=20))
        self.assertGreater(result['looking_down_seconds'], 5)
        result = model(feature_window(19000, pitch=20))
        self.assertEqual(result['looking_down_seconds'], 0)
        self.assertIsNone(result['current_direction_score'])

    def test_group_score_reflects_downward_decay_after_brief_missing_face(self):
        from engagement_model.position_tracks import PositionTrackGaussianRuntime
        from engagement_model.tests.test_position_tracks import sample
        model = PositionTrackGaussianRuntime(eye_fusion=True)
        for t in range(0, 24001, 100):
            item = sample(t, (.5,))
            if t >= 10000:
                item['features']['faces'][0]['head_angles_degrees']['pitch'] = 20
            if t in (12000, 15000, 18000):
                item['features']['faces'] = []
                item['quality'] = {'status': 'not_observable', 'reason_codes': ['face_not_found']}
            result = model(item)
        row = model.participants()[0]
        self.assertGreater(row['looking_down_seconds'], 10)
        self.assertLess(result['current_direction_score'], 80)
    def test_five_second_grace_then_seven_points_per_second_and_recovery(self):
        model = GaussianRuntime()
        for t in range(0, 10100, 100): model(feature_window(t))
        scores = {}
        for t in range(10100, 25200, 100):
            result = model(feature_window(t, pitch=20))
            if t in (10100, 15100, 20100, 25100): scores[t] = result['current_direction_score']
        self.assertEqual(scores, {10100:100.0, 15100:100.0, 20100:65.0, 25100:30.0})
        self.assertIn('looking_down_score_decay', result['reason_codes'])
        self.assertEqual(model(feature_window(25200))['current_direction_score'], 100)

    def test_penalty_reaches_zero_without_crossing_zero(self):
        model = GaussianRuntime()
        for t in range(0, 10100, 100):
            model(feature_window(t))
        for t in range(10100, 70200, 100):
            result = model(feature_window(t, pitch=20))
        self.assertEqual(result['current_direction_score'], 0.0)

    def test_no_previous_valid_score_does_not_invent_one(self):
        model = GaussianRuntime()
        for t in range(0, 2000, 100): result = model(feature_window(t, pitch=20))
        self.assertIsNone(result['current_direction_score'])
        self.assertIsNone(result['gaussian_observation'])

    def test_downward_observations_are_included_in_single_and_group_windows(self):
        from engagement_model.config import ModelConfig
        single = GaussianRuntime(ModelConfig.gaussian(window_seconds=10))
        segments = [(0, 10000, True, None, (None, None, 80.0))]
        self.assertEqual(gaussian_window_score(segments, 0, 10000, single.config)['score'], 80.0)
        multi = MultiFaceGaussianRuntime(ModelConfig.gaussian(window_seconds=10))
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
