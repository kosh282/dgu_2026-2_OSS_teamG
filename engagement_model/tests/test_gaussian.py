"""각도 Gaussian·논문 기록 가중식·실시간 가중 변형과 기존 계약 회귀 검증."""

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from engagement_model.config import ModelConfig
from engagement_model.gaussian import (
    GaussianRuntime, gaussian_angle_score, gaussian_window_score, li_liu_rank_average,
)
from engagement_model.input_adapter import HeadAngles
from engagement_model.tests.test_runtime import feature_window
from engagement_model.window_publisher import WindowPublisher
from integration import IntegrationService


class GaussianFormulaTests(unittest.TestCase):
    def test_center_symmetry_and_monotonicity(self):
        config = ModelConfig.gaussian()
        self.assertEqual(gaussian_angle_score(HeadAngles(0, 0, 0), config), 100)
        scores = [gaussian_angle_score(HeadAngles(0, angle, 0), config) for angle in (0, 10, 20, 60, 90)]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertAlmostEqual(scores[2], 100 * math.exp(-.5))
        self.assertEqual(gaussian_angle_score(HeadAngles(2, -10, 0), config),
                         gaussian_angle_score(HeadAngles(-2, 10, 0), config))
        self.assertGreater(gaussian_angle_score(HeadAngles(2, 0, 0), config), 99)

    def test_sigma_is_configurable_and_validated(self):
        narrow = ModelConfig.gaussian(gaussian_sigma_yaw_degrees=10)
        broad = ModelConfig.gaussian(gaussian_sigma_yaw_degrees=30)
        self.assertLess(gaussian_angle_score(HeadAngles(0, 20, 0), narrow),
                        gaussian_angle_score(HeadAngles(0, 20, 0), broad))
        for bad in (0, -1, True, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                ModelConfig.gaussian(gaussian_sigma_pitch_degrees=bad)
        tiny = ModelConfig.gaussian(gaussian_sigma_pitch_degrees=1e-300)
        self.assertEqual(gaussian_angle_score(HeadAngles(180, 0, 0), tiny), 0)

    def test_original_rank_formula(self):
        self.assertEqual(li_liu_rank_average([100, 100, 0, 0]), 30)
        self.assertEqual(li_liu_rank_average([0, 0, 100, 100]), 70)
        self.assertIsNone(li_liu_rank_average([]))
        for bad in (None, True, float('nan'), -1, 101):
            with self.assertRaises(ValueError):
                li_liu_rank_average([bad])

    def test_time_kernel_matches_hand_integral_and_ignores_segmentation(self):
        config = ModelConfig.gaussian()
        # 직전 10초 중 오래된 5초는 100, 최근 5초는 Gaussian(0,60)점.
        rows = [(0, 5000, True, 0, (0, 0)), (5000, 10000, True, 100, (0, 60))]
        result = gaussian_window_score(rows, 0, 10000, config)
        expected = .25 * 100 + .75 * 100 * math.exp(-4.5)
        self.assertEqual(result['score'], round(expected, 1))
        split = [(ms, ms + 100, True, None, (0, 0 if ms < 5000 else 60)) for ms in range(0, 10000, 100)]
        self.assertEqual(result, gaussian_window_score(split, 0, 10000, config))

    def test_none_is_excluded_and_observation_gate_is_preserved(self):
        config = ModelConfig.gaussian()
        result = gaussian_window_score([(0, 8000, True, None, (0, 0)),
                                        (8000, 10000, True, None, None)], 0, 10000, config)
        self.assertEqual(result['score'], 100)
        self.assertEqual(result['scoreable_duration_ms'], 8000)
        insufficient = gaussian_window_score([(0, 1000, True, None, (0, 0))], 0, 10000, config)
        self.assertIsNone(insufficient['score'])


class GaussianRuntimeTests(unittest.TestCase):

    def test_scores_and_observation_states_match_before_model_cleanup(self):
        """삭제 작업 전 두 Gaussian 경로의 602개 출력과 대조한다."""
        models = [GaussianRuntime(), GaussianRuntime(downward_policy=lambda f, r: (False, 'not_down'))]
        keys = ['score', 'current_direction_score', 'orientation_state', 'status', 'reason_codes', 'gaussian_observation', 'front_seconds', 'away_seconds', 'looking_down_seconds', 'eye_candidate']
        rows = []
        for ms in range(0, 30001, 100):
            options = ({} if ms < 3000 or ms >= 24000 else
                       {'yaw': 45} if ms < 10000 else
                       {'pitch': 20} if ms < 20000 else {'faces': 0})
            rows.append([{key: result.get(key) for key in keys}
                         for result in (model(feature_window(ms, **options)) for model in models)])
        digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, 'f922bb95ca4bd929b7f6edc1fa440326daac49956fbb22cdf2bb2236e0458c37')

    def test_softening_improves_small_motion_without_changing_state_or_down_policy(self):
        for pitch, yaw in [(0, 10), (-5, 0), (5, 0), (0, 60), (12, 0), (20, 0)]:
            old = GaussianRuntime(ModelConfig.gaussian(gaussian_angle_mode='relative'))
            improved = GaussianRuntime()
            a = old(feature_window(0, pitch=pitch, yaw=yaw))
            b = improved(feature_window(0, pitch=pitch, yaw=yaw))
            self.assertEqual(a['orientation_state'], b['orientation_state'])
            self.assertEqual(a['downward_status'], b['downward_status'])
            if a['current_direction_score'] is None:
                self.assertIsNone(b['current_direction_score'])
            else:
                self.assertGreater(b['current_direction_score'], a['current_direction_score'])
            returned = improved(feature_window(100))
            self.assertEqual(returned['current_direction_score'], 100)
        neutral = GaussianRuntime()(feature_window(0, pitch=-2, yaw=2))
        self.assertEqual(neutral['current_direction_score'], 100)

    def test_shared_softening_is_continuous_symmetric_and_preserves_large_angle_slope(self):
        from engagement_model.input_adapter import soften_angle
        for boundary in (2, 15):
            a = soften_angle(boundary-1e-6, 2, .75, 15)
            b = soften_angle(boundary+1e-6, 2, .75, 15)
            self.assertLess(abs(a-b), 3e-6)
        self.assertEqual(soften_angle(60, 2, .75, 15), 54.75)
        self.assertEqual(soften_angle(-60, 2, .75, 15), -54.75)

    def test_left_right_symmetry_through_runtime_and_formula(self):
        for pitch in (-10, 0, 10):
            for yaw in (5, 15, 30, 60, 70):
                self.assertEqual(gaussian_angle_score(HeadAngles(pitch, yaw, 0), ModelConfig.gaussian()),
                                 gaussian_angle_score(HeadAngles(pitch, -yaw, 0), ModelConfig.gaussian()))
                left, right = GaussianRuntime(), GaussianRuntime()
                for ms in range(0, 1100, 100):
                    a = left(feature_window(ms, pitch=pitch, yaw=yaw))
                    b = right(feature_window(ms, pitch=pitch, yaw=-yaw))
                    self.assertEqual(a['current_direction_score'], b['current_direction_score'])
                    self.assertEqual(a['score'], b['score'])

    def test_downward_transition_excluded_immediately_then_confirmed_and_return(self):
        model = GaussianRuntime()
        self.run_frames(model, 0, 10000)
        small = model(feature_window(10100, pitch=5))
        self.assertIsNotNone(small['current_direction_score'])
        for ms, pitch in [(10200, 11), (10300, 14)]:
            r = model(feature_window(ms, pitch=pitch))
            self.assertIsNone(r['current_direction_score'])
        for ms in (10400, 10500, 10600):
            r = model(feature_window(ms, pitch=16))
            self.assertIsNotNone(r['current_direction_score'])
            self.assertIn('looking_down_grace', r['reason_codes'])
        self.assertEqual(model.latest['downward_status'], 'looking_down_uncertain')
        confirmed = model(feature_window(10700, pitch=16))
        self.assertEqual(confirmed['downward_status'], 'looking_down_confirmed')
        resumed = model(feature_window(10800))
        self.assertEqual(resumed['current_direction_score'], 100)
        self.assertIsNone(resumed['downward_status'])
        self.assertTrue(any(row[4] and len(row[4]) == 3 for row in model._gaussian_segments))

    def test_transition_that_does_not_become_down_stays_unscored(self):
        model = GaussianRuntime()
        model(feature_window(0))
        transition = model(feature_window(100, pitch=12))
        self.assertEqual(transition['downward_status'], 'transition_uncertain')
        self.assertIn('downward_transition_uncertain', transition['reason_codes'])
        resumed = model(feature_window(200))
        self.assertEqual(resumed['current_direction_score'], 100)
        self.assertIsNone(model._gaussian_segments[-1][4])

    def test_confirmation_resets_after_missing_track_change_and_recalibration(self):
        model = GaussianRuntime()
        for ms in (0, 100, 200):
            model(feature_window(ms, pitch=20))
        model(feature_window(300, faces=0))
        self.assertEqual(model(feature_window(400, pitch=20))['downward_status'], 'looking_down_uncertain')
        self.assertEqual(model(feature_window(500, pitch=20, track_key=2))['downward_status'], 'looking_down_uncertain')
        model.reset()
        self.assertIsNone(model.latest['downward_status'])
        for bad in (0, 15, 20):
            with self.assertRaises(ValueError):
                ModelConfig.gaussian(gaussian_down_transition_pitch_degrees=bad)
        custom = GaussianRuntime(ModelConfig.gaussian(gaussian_down_transition_pitch_degrees=13))
        self.assertIsNotNone(custom(feature_window(0, pitch=12))['current_direction_score'])

    def test_publisher_matches_exclusion_without_changing_json_contract(self):
        service = IntegrationService()
        model = GaussianRuntime()
        sid = service.start_session('transition')
        service.elapsed_ms = lambda sid: 20000
        publisher = WindowPublisher(service, sid, model)
        for ms in range(0, 10001, 100):
            feature = feature_window(ms, pitch=12 if 2000 <= ms < 3000 else 0)
            feature['session_id'] = sid
            publisher(feature)
        event = service.get_dashboard_snapshot(sid)['engagement_windows'][0]
        self.assertEqual(event['payload']['model_output']['value'], 100)
        self.assertEqual(model.latest['score'], 100)
        ext = event['payload']['model_extensions'][model.model_id]
        self.assertEqual(ext['scoreable_duration_ms'], 9000)
        self.assertEqual(ext['observable_duration_ms'], 10000)
        self.assertNotIn('downward_status', event['payload'])
        service.stop_session(sid)

    def run_frames(self, runtime, start, end, **kwargs):
        for ms in range(start, end + 1, 100):
            result = runtime(feature_window(ms, **kwargs))
        return result

    def test_front_stable_across_pitch_zero_and_small_jitter(self):
        runtime = GaussianRuntime()
        for ms in range(0, 12001, 100):
            result = runtime(feature_window(ms, pitch=(ms % 300 / 100 - 1) * 2))
            if ms:
                self.assertGreater(result['score'], 99)
        self.assertTrue(runtime.sampling_diagnostics()['regular'])

    def test_current_direction_updates_on_first_valid_sample_without_time_gate(self):
        runtime = GaussianRuntime()
        first = runtime(feature_window(0))
        self.assertEqual(first['current_direction_score'], 100)
        self.assertIsNone(first['score'])
        self.run_frames(runtime, 100, 10000)
        away = runtime(feature_window(10100, yaw=60))
        self.assertEqual(away['current_direction_score'], 2.4)
        self.assertEqual(away['orientation_state'], 'away')
        self.assertGreater(away['score'], away['current_direction_score'])
        front = runtime(feature_window(10200))
        self.assertEqual(front['current_direction_score'], 100)
        self.assertLess(front['score'], 100)

    def test_current_recovers_immediately_after_long_down_while_aggregate_waits(self):
        runtime = GaussianRuntime()
        self.run_frames(runtime, 0, 10000)
        down = self.run_frames(runtime, 10100, 30100, pitch=20)
        self.assertEqual(down['current_direction_score'], 80.0)
        boundary = runtime(feature_window(30200, pitch=12))
        self.assertEqual(boundary['current_direction_score'], 80.0)
        front = runtime(feature_window(30300))
        self.assertEqual(front['current_direction_score'], 100)
        self.assertIsNotNone(front['score'])
        missing = runtime(feature_window(30400, faces=0))
        self.assertIsNone(missing['current_direction_score'])
        self.assertIsNone(missing['display_score'])
        invalid = runtime({})
        self.assertIsNone(invalid['current_direction_score'])
        runtime.reset()
        self.assertIsNone(runtime.latest['current_direction_score'])

    def test_current_direction_is_independent_of_aggregate_gate_configuration(self):
        runtime = GaussianRuntime(ModelConfig.gaussian(min_scoreable_ratio=1))
        self.run_frames(runtime, 0, 3000, pitch=20)
        front = runtime(feature_window(3100))
        self.assertEqual(front['current_direction_score'], 100)
        self.assertIsNone(front['score'])

    def test_short_and_sustained_away_then_return(self):
        runtime = GaussianRuntime()
        self.run_frames(runtime, 0, 10000)
        short = self.run_frames(runtime, 10100, 11000, yaw=60)
        sustained = self.run_frames(runtime, 11100, 21100, yaw=60)
        self.assertGreater(short['score'], sustained['score'])
        self.assertEqual(sustained['score'], 2.4)
        recovered = self.run_frames(runtime, 21200, 31200)
        self.assertEqual(recovered['score'], 100)

    def test_irregular_timestamps_do_not_change_time_weighting(self):
        regular, irregular = GaussianRuntime(), GaussianRuntime()
        times = [0]
        while times[-1] < 10000:
            times.append(times[-1] + (50 if len(times) % 2 else 150))
        for model, samples in ((regular, range(0, 10001, 100)), (irregular, times)):
            for ms in samples:
                model(feature_window(ms, yaw=0 if ms < 5000 else 60))
        self.assertEqual(regular.latest['score'], irregular.latest['score'])
        self.assertFalse(irregular.sampling_diagnostics()['regular'])

    def test_down_and_missing_never_expose_held_score_and_can_resume(self):
        runtime = GaussianRuntime()
        self.run_frames(runtime, 0, 10000)
        down = runtime(feature_window(10100, pitch=20))
        self.assertEqual(down['score'], 100.0)
        self.assertIn('looking_down_grace', down['reason_codes'])
        missing = runtime(feature_window(10200, faces=0))
        self.assertIsNone(missing['score'])
        self.assertIsNone(missing['display_score'])
        self.run_frames(runtime, 10300, 20300, pitch=20)
        recovered = self.run_frames(runtime, 20400, 30400)
        self.assertEqual(recovered['score'], 100)




class GaussianPublisherTests(unittest.TestCase):
    def test_contract_uses_raw_gaussian_once_and_round_trips(self):
        with patch('integration.service.time.monotonic_ns', return_value=0):
            service = IntegrationService()
            runtime = GaussianRuntime()
            sid = service.start_session('Gaussian', {'config_version': runtime.config.config_version,
                'participation_model': {'model_id': runtime.model_id, 'model_version': runtime.model_version}})
            service.elapsed_ms = lambda sid: 20000
            publisher = WindowPublisher(service, sid, runtime)
            for ms in range(0, 10001, 100):
                feature = feature_window(ms, yaw=0 if ms < 5000 else 60)
                feature['session_id'] = sid
                publisher(feature)
            event = service.get_dashboard_snapshot(sid)['engagement_windows'][0]
            self.assertEqual(event['payload']['model_output']['value'], 26.8)
            self.assertEqual(event['payload']['model_output']['value'], runtime.latest['score'])
            self.assertEqual(event['provenance']['model_id'], 'gaussian-head-direction')
            extension = event['payload']['model_extensions']['gaussian-head-direction']
            self.assertEqual(extension['sigma_pitch_degrees'], 15)
            self.assertEqual(extension['sigma_yaw_degrees'], 20)
            self.assertEqual(extension['temporal_weighting'], 'linear-time-integral')
            self.assertIsNone(event['payload']['observation']['eligible_count'])
            service.stop_session(sid)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'session.json'
                service.save_session(sid, path)
                raw = path.read_text(encoding='utf-8')
                self.assertNotIn('gaussian_observation', raw)
                self.assertNotIn('current_direction_score', raw)
                restored = IntegrationService()
                restored.load_session(path)
                self.assertEqual(event, restored.get_dashboard_snapshot(sid)['engagement_windows'][0])


if __name__ == '__main__':
    unittest.main()
