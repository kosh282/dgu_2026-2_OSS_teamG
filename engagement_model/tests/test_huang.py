"""논문 손계산, 온라인 변형, 하향 보류와 기존 계약 호환성을 검증한다."""

from copy import deepcopy
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from engagement_model.config import ModelConfig
from engagement_model.runtime import EngagementRuntime, ComparisonRuntime
from engagement_model.scoring import (
    huang_head_up, huang_concentration, huang_front_facing, huang_window_score,
)
from engagement_model.window_publisher import WindowPublisher
from engagement_model.tests.test_runtime import feature_window
from integration import IntegrationService


class HuangFormulaTests(unittest.TestCase):
    def test_dynamic_threshold_and_amplitude_against_hand_calculation(self):
        # max-min=20 >=5, q=0, a=5/10=.5, P=sigmoid(2.5).
        self.assertAlmostEqual(huang_head_up(5, [-10, 0, 10], 40), 1 / (1 + math.exp(-2.5)))
        self.assertAlmostEqual(huang_head_up(-5, [-10, 0, 10], 40), 1 / (1 + math.exp(2.5)))

    def test_global_reference_branch_and_threshold_boundary(self):
        # 큰 구간 진폭 2<5: 명시한 전체/보정 기준 0을 사용한다.
        self.assertAlmostEqual(huang_head_up(2, [1, 2, 3], 0), 1 / (1 + math.exp(-4 / 3)))
        self.assertAlmostEqual(huang_head_up(4, [0, 5], 50), 1 / (1 + math.exp(-0.9)))

    def test_degenerate_range_is_deferred_and_extremes_do_not_overflow(self):
        self.assertIsNone(huang_head_up(0, [0, 0, 0], 0))
        self.assertIsNone(huang_head_up(0, [], 0))
        self.assertEqual(huang_head_up(-1000, [-1000, 1000], 0), 0)
        self.assertEqual(huang_head_up(1000, [-1000, 1000], 0), 1)

    def test_online_neutral_extension_is_explicit_and_continuous_at_zero(self):
        # 원식 기본 경로는 보류를 유지하고 온라인 경로만 중립 sigmoid(0)를 쓴다.
        self.assertIsNone(huang_head_up(0, [0, 0], 0))
        self.assertEqual(huang_head_up(0, [0, 0], 0, neutral_on_degenerate=True), .5)
        for pitch in (-1e-12, -1e-100, 1e-100, 1e-12):
            result = huang_head_up(pitch, [-abs(pitch), 0, abs(pitch)], 0,
                                   neutral_on_degenerate=True)
            self.assertAlmostEqual(result, .5, places=10)
        # 분모만 0이고 분자가 0이 아닌 계산을 중립값으로 덮지 않는다.
        self.assertIsNone(huang_head_up(1, [0], 0, neutral_on_degenerate=True))

    def test_neutral_zero_pitch_window_still_uses_yaw_and_quality(self):
        config = ModelConfig.huang(window_seconds=6)
        front = huang_window_score([(0, 6000, True, None, (0, 0))], 0, 6000, config)
        side = huang_window_score([(0, 6000, True, None, (0, 90))], 0, 6000, config)
        self.assertEqual(front['score'], 64.6)
        self.assertLess(side['score'], front['score'])
        self.assertEqual(front['scoreable_duration_ms'], 6000)
        missing = huang_window_score([(0, 6000, False, None, None)], 0, 6000, config)
        self.assertIsNone(missing['score'])
        sparse = huang_window_score([(0, 300, True, None, (0, 0))], 0, 6000, config)
        self.assertIsNone(sparse['score'])

    def test_window_near_zero_jitter_produces_finite_bounded_score(self):
        rows = [(ms, ms + 100, True, None, ((1 if ms % 200 else -1) * 1e-12, 0))
                for ms in range(0, 6000, 100)]
        result = huang_window_score(rows, 0, 6000, ModelConfig.huang(window_seconds=6))
        self.assertTrue(math.isfinite(result['score']))
        self.assertEqual(result['score'], 64.6)

    def test_nonfinite_inputs_and_unrepresentable_intermediates_are_not_scores(self):
        for pitch in (float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                huang_head_up(pitch, [0, 1], 0, neutral_on_degenerate=True)
        self.assertIsNone(huang_head_up(1e308, [-1e308], -1e308,
                                       neutral_on_degenerate=True))

    def test_euclidean_normalization(self):
        self.assertEqual(huang_concentration(0, 0), 0)
        self.assertEqual(huang_concentration(1, 1), 1)
        self.assertEqual(huang_concentration(.5, .5), .5)
        self.assertAlmostEqual(huang_concentration(.5, 1), 1 - .5 / math.sqrt(2))
        for value in (True, float('nan'), -1, 100):
            with self.assertRaises(ValueError):
                huang_concentration(value, 1)

    def test_original_yaw_geometry_and_range_are_not_silently_changed(self):
        self.assertEqual(huang_front_facing(45, 50, 10, 100), .5)
        self.assertEqual(huang_front_facing(45, 60, 10, 100), 1)
        self.assertEqual(huang_front_facing(45, 40, 10, 100), 0)
        self.assertGreater(huang_front_facing(0, 60, 10, 100), 1)
        with self.assertRaises(ValueError):
            huang_front_facing(0, 50, 0, 100)

    def test_settings_reject_invalid_method_and_interval(self):
        for kwargs in ({'scoring_method': 'unknown'}, {'huang_small_seconds': 0},
                       {'huang_max_yaw_degrees': 0}, {'huang_small_seconds': True}):
            with self.assertRaises(ValueError):
                ModelConfig(**kwargs)
        with self.assertRaises(ValueError):
            ModelConfig.huang(window_seconds=2)

    def test_window_combines_rates_before_distance_and_weights_time(self):
        config = ModelConfig.huang(window_seconds=6)
        # P는 모든 작은 구간에서 sigmoid(2), Y는 1과 0의 시간 가중 평균 .5.
        rows = [(0, 3000, True, 100, (2, 0)), (3000, 6000, True, 0, (2, 90))]
        result = huang_window_score(rows, 0, 6000, config)
        p = 1 / (1 + math.exp(-2))
        self.assertEqual(result['score'], round(100 * huang_concentration(p, .5), 1))
        self.assertEqual(result['scoreable_duration_ms'], 6000)
        self.assertNotEqual(result['score'], round(50 * (huang_concentration(p, 1) + huang_concentration(p, 0)), 1))
        # 표시용 기존 score를 바꾸어도 Huang 구간 결과는 바뀌지 않는다.
        self.assertEqual(result, huang_window_score([(a, b, o, None, v) for a, b, o, _, v in rows], 0, 6000, config))


class HuangRuntimeTests(unittest.TestCase):
    def run_frames(self, runtime, start, end, **kwargs):
        result = None
        for timestamp in range(start, end + 1, 100):
            result = runtime(feature_window(timestamp, **kwargs))
        return result

    def test_front_and_side_use_uncompressed_source_angles(self):
        runtime = EngagementRuntime(ModelConfig.huang())
        front = self.run_frames(runtime, 0, 6000, pitch=-2)
        self.assertEqual(front['orientation_state'], 'front')
        self.assertEqual(front['status'], 'scored')
        self.assertEqual(front['model_version'], 'huang-online-0.2')
        side = self.run_frames(runtime, 6100, 36100, pitch=-2, yaw=45)
        self.assertEqual(side['orientation_state'], 'away')
        self.assertLess(side['score'], front['score'])
        expected = round(100 * huang_concentration(1 / (1 + math.exp(-2)), .5), 1)
        self.assertEqual(side['score'], expected)

    def test_stationary_zero_pitch_scores_after_real_warmup(self):
        runtime = EngagementRuntime(ModelConfig.huang())
        waiting = self.run_frames(runtime, 0, 2900, pitch=0)
        self.assertIsNone(waiting['score'])
        scored = runtime(feature_window(3000, pitch=0))
        self.assertEqual(scored['score'], 64.6)
        self.assertEqual(scored['status'], 'scored')
        self.assertNotIn('huang_degenerate_pitch_range', scored['reason_codes'])

    def test_return_to_zero_pitch_restarts_only_live_window(self):
        runtime = EngagementRuntime(ModelConfig.huang())
        self.run_frames(runtime, 0, 6000, pitch=-2)
        self.run_frames(runtime, 6100, 36100, pitch=20)
        first = runtime(feature_window(36200, pitch=0))
        self.assertIsNone(first['score'])
        self.assertIn('huang_recovery_warmup', first['reason_codes'])
        waiting = self.run_frames(runtime, 36300, 39100, pitch=0)
        self.assertIsNone(waiting['score'])
        recovered = runtime(feature_window(39200, pitch=0))
        self.assertEqual(recovered['score'], 64.6)
        self.assertEqual(recovered['orientation_state'], 'front')
        self.assertNotIn('huang_recovery_warmup', recovered['reason_codes'])
        self.assertEqual(recovered['front_seconds'], 3)
        for ms in range(39300, 45301, 100):
            result = runtime(feature_window(ms, pitch=0))
            self.assertEqual(result['score'], 64.6, f'복귀 후 재보류: {ms}')

    def test_brief_front_glance_and_missing_face_do_not_skip_recovery(self):
        runtime = EngagementRuntime(ModelConfig.huang())
        self.run_frames(runtime, 0, 3000, pitch=20)
        self.run_frames(runtime, 3100, 4100, pitch=0)
        down = runtime(feature_window(4200, pitch=16))
        self.assertIsNone(down['score'])
        boundary = runtime(feature_window(4300, pitch=12))
        self.assertEqual(boundary['orientation_state'], 'looking_down')
        self.run_frames(runtime, 4400, 7300, pitch=0)
        missing = runtime(feature_window(7400, faces=0))
        self.assertIsNone(missing['score'])
        self.assertIn('face_not_found', missing['reason_codes'])
        runtime.reset()
        self.assertIsNone(runtime(feature_window(7500, pitch=0))['score'])

    def test_recovery_uses_configured_small_interval(self):
        runtime = EngagementRuntime(ModelConfig.huang(huang_small_seconds=1))
        self.run_frames(runtime, 0, 1000, pitch=20)
        self.run_frames(runtime, 1100, 2000, pitch=0)
        self.assertIsNone(runtime.latest['score'])
        self.assertEqual(runtime(feature_window(2100, pitch=0))['score'], 64.6)
        for ms in range(2200, 4001, 100):
            self.assertEqual(runtime(feature_window(ms, pitch=0))['score'], 64.6)

    def test_down_is_deferred_even_after_long_duration_and_recovers(self):
        runtime = EngagementRuntime(ModelConfig.huang())
        self.run_frames(runtime, 0, 6000, pitch=-2)
        down = self.run_frames(runtime, 6100, 22100, pitch=20)
        self.assertIsNone(down['score'])
        self.assertEqual(down['orientation_state'], 'looking_down')
        self.assertEqual(down['looking_down_seconds'], 16)
        self.assertIn('downward_activity_uncertain', down['reason_codes'])
        self.assertIsNone(down['display_score'])
        # 12도에서도 기존 하향 히스테리시스를 유지한다.
        boundary = runtime(feature_window(22200, pitch=12))
        self.assertEqual(boundary['orientation_state'], 'looking_down')
        recovered = self.run_frames(runtime, 22300, 52300, pitch=-2)
        self.assertEqual(recovered['orientation_state'], 'front')
        self.assertIsNotNone(recovered['score'])

    def test_missing_multiple_faces_and_gap_reset(self):
        runtime = EngagementRuntime(ModelConfig.huang())
        self.run_frames(runtime, 0, 6000, pitch=-2)
        missing = runtime(feature_window(6100, faces=0))
        self.assertIsNone(missing['score'])
        self.assertIn('face_not_found', missing['reason_codes'])
        multiple = runtime(feature_window(6200, faces=2))
        self.assertIsNone(multiple['score'])
        resumed = runtime(feature_window(7000, pitch=-2))
        self.assertIn('observation_gap', resumed['reason_codes'])
        self.assertIsNone(resumed['score'])
        self.assertIn('huang_warmup', resumed['reason_codes'])

    def test_track_change_time_reversal_reset_and_input_immutability(self):
        runtime = EngagementRuntime(ModelConfig.huang())
        self.run_frames(runtime, 0, 6000, pitch=-2)
        feature = feature_window(6100, pitch=-2, track_key=2)
        original = deepcopy(feature)
        changed = runtime(feature)
        self.assertEqual(feature, original)
        self.assertIn('tracking_key_changed', changed['reason_codes'])
        self.assertIsNone(changed['score'])
        reversed_time = runtime(feature_window(6000, pitch=-2, track_key=2))
        self.assertIn('non_monotonic_timestamp', reversed_time['reason_codes'])
        runtime.reset()
        self.assertIsNone(runtime.latest['score'])
        invalid = runtime({})
        self.assertEqual(invalid['status'], 'input_error')

    def test_comparison_does_not_change_legacy_and_reports_deferral(self):
        comparison = ComparisonRuntime('huang')
        legacy = EngagementRuntime()
        for ms in range(0, 20001, 100):
            feature = feature_window(ms, pitch=-2 if ms < 5000 else 20)
            comparison(feature)
            self.assertEqual(comparison.runtimes['legacy'].latest, legacy(feature))
        summary = comparison.summary()
        self.assertGreater(summary['huang']['deferred_ratio'], summary['legacy']['deferred_ratio'])
        self.assertEqual(summary['huang']['frames'], 201)

    def test_missing_upper_body_does_not_invent_writing_evidence(self):
        with_body = EngagementRuntime(ModelConfig.huang())
        without_body = EngagementRuntime(ModelConfig.huang())
        for ms in range(0, 4001, 100):
            feature = feature_window(ms, pitch=20)
            expected = without_body(feature)
            feature['features']['upper_bodies'] = [
                {'shoulder_center': {'x': .5, 'y': .5}, 'shoulder_visibility': 1,
                 'quality': {'status': 'valid', 'reason_codes': []}}]
            self.assertEqual(with_body(feature), expected)

    def test_future_observations_do_not_modify_past_results(self):
        runtime = EngagementRuntime(ModelConfig.huang())
        previous = self.run_frames(runtime, 0, 6000, pitch=-2)
        snapshot = deepcopy(previous)
        self.run_frames(runtime, 6100, 12000, pitch=-10, yaw=90)
        self.assertEqual(previous, snapshot)


class HuangPublisherTests(unittest.TestCase):
    def test_live_recovery_does_not_erase_downward_time_from_saved_window(self):
        with patch('integration.service.time.monotonic_ns', return_value=0):
            service = IntegrationService()
            runtime = EngagementRuntime(ModelConfig.huang())
            sid = service.start_session('recovery', {
                'config_version': runtime.config.config_version,
                'participation_model': {'model_id': runtime.model_id, 'model_version': runtime.model_version}})
            service.elapsed_ms = lambda sid: 100000
            publisher = WindowPublisher(service, sid, runtime)
            for ms in range(0, 63201, 100):
                feature = feature_window(ms, pitch=20 if 30000 < ms < 60200 else 0)
                feature['session_id'] = sid
                result = publisher(feature)
            self.assertEqual(result['orientation_state'], 'front')
            self.assertEqual(result['score'], 64.6)
            publisher.finish(63300)
            event = service.get_dashboard_snapshot(sid)['engagement_windows'][-1]
            self.assertEqual((event['start_ms'], event['end_ms']), (33300, 63300))
            self.assertIsNone(event['payload']['model_output'])
            extension = event['payload']['model_extensions'][runtime.model_id]
            self.assertEqual(extension['window_duration_ms'], 30000)
            self.assertEqual(extension['observable_duration_ms'], 30000)
            self.assertEqual(extension['scoreable_duration_ms'], 3000)
            self.assertIn('insufficient_scoreable_time', event['quality']['reason_codes'])

    def test_event_identity_null_gates_and_round_trip(self):
        with patch('integration.service.time.monotonic_ns', return_value=0):
            service = IntegrationService()
            runtime = EngagementRuntime(ModelConfig.huang(window_seconds=6))
            sid = service.start_session('synthetic', {
                'config_version': runtime.config.config_version,
                'participation_model': {'model_id': runtime.model_id, 'model_version': runtime.model_version}})
            service.elapsed_ms = lambda sid: 20000
            publisher = WindowPublisher(service, sid, runtime)
            for ms in range(0, 6001, 100):
                feature = feature_window(ms, pitch=-2)
                feature['session_id'] = sid
                publisher(feature)
            first = service.get_dashboard_snapshot(sid)['engagement_windows'][0]
            self.assertEqual(first['provenance']['model_version'], 'huang-online-0.2')
            self.assertEqual(first['payload']['model_output']['value'], runtime.latest['score'])
            self.assertIsNone(first['payload']['observation']['eligible_count'])
            publisher.advance(14000)
            events = service.get_dashboard_snapshot(sid)['engagement_windows']
            self.assertIsNone(events[-1]['payload']['model_output'])
            self.assertEqual(events[-1]['quality']['status'], 'not_observable')
            service.stop_session(sid)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'session.json'
                service.save_session(sid, path)
                saved = path.read_text(encoding='utf-8')
                self.assertNotIn('huang_observation', saved)
                self.assertNotIn('transient_track_key', saved)
                restored = IntegrationService()
                restored.load_session(path)
                self.assertEqual(events, restored.get_dashboard_snapshot(sid)['engagement_windows'])

    def test_downward_only_is_observable_but_not_scoreable(self):
        rows = [(0, 6000, True, None, None)]
        result = huang_window_score(rows, 0, 6000, ModelConfig.huang())
        self.assertEqual(result['observable_duration_ms'], 6000)
        self.assertEqual(result['scoreable_duration_ms'], 0)
        self.assertIsNone(result['score'])


@unittest.skipUnless(importlib.util.find_spec('cv2'), '영상 의존성이 설치된 환경에서 렌더러 검증')
class HuangDemoCompatibilityTests(unittest.TestCase):
    def test_unmodified_panel_accepts_both_model_output_formats(self):
        from video_recognition.developer_view import render_feature_panel
        for config in (ModelConfig(), ModelConfig.huang()):
            feature = feature_window(0, pitch=20)
            result = EngagementRuntime(config)(feature)
            with patch('video_recognition.developer_view.cv2.putText') as draw:
                panel = render_feature_panel(feature, model_result=result)
            self.assertEqual(panel.shape[1:], (1200, 3))
            text = '\n'.join(call.args[1] for call in draw.call_args_list)
            self.assertIn(result['orientation_state'], text)
            self.assertIn(result['status'], text)
            for reason in result['reason_codes']:
                self.assertIn(reason, text)

    def test_unmodified_demo_retains_legacy_metadata(self):
        from video_recognition.webcam_test import open_demo_log
        import json
        for runtime in (EngagementRuntime(),):
            with tempfile.TemporaryDirectory() as directory:
                handle, path = open_demo_log(Path(directory), 'synthetic', runtime)
                handle.close()
                metadata = json.loads(path.read_text(encoding='utf-8'))
                self.assertEqual(metadata['model_version'], runtime.model_version)
                self.assertEqual(metadata['config']['scoring_method'], runtime.config.scoring_method)


@unittest.skipUnless(importlib.util.find_spec('cv2'), '영상 의존성이 설치된 환경에서 실행기 검증')
class HuangWebcamRunnerTests(unittest.TestCase):
    def test_normal_screen_hides_debug_data_and_debug_shows_both_scores(self):
        from engagement_model.webcam_demo import panel_lines, render_panel
        from video_recognition.preview import LandmarkOverlaySmoother
        import numpy as np
        from engagement_model.gaussian import GaussianRuntime
        model = GaussianRuntime()
        feature = feature_window(0)
        model(feature)
        models = {'gaussian': model}
        normal = '\n'.join(panel_lines(feature, models))
        self.assertIn('Current direction: 100.0/100', normal)
        self.assertNotIn('Reasons:', normal)
        self.assertNotIn('aggregate', normal)
        self.assertNotIn('Relative pitch/yaw', normal)
        detailed = '\n'.join(panel_lines(feature, models, debug=True))
        self.assertIn('10s aggregate:', detailed)
        self.assertIn('Reasons:', detailed)
        self.assertIn('Relative pitch/yaw', detailed)
        with patch('engagement_model.webcam_demo.render_video_frame') as overlay:
            panel = render_panel(np.zeros((480, 640, 3), dtype=np.uint8), feature, models,
                overlay_smoother=LandmarkOverlaySmoother(), face=Mock(last_landmarks=[]),
                pose=Mock(), show_overlay=True)
            overlay.assert_not_called()
        self.assertEqual(panel.shape, (860, 1400, 3))

    def test_cli_defaults_to_simple_gaussian_and_debug_is_explicit(self):
        from engagement_model.webcam_demo import main
        for arguments, debug in (([], False), (['--debug'], True)):
            with patch('sys.argv', ['webcam_demo'] + arguments), \
                 patch('engagement_model.webcam_demo.run') as run:
                main()
            self.assertEqual(run.call_args.kwargs['primary'], 'gaussian')
            self.assertEqual(run.call_args.kwargs['debug'], debug)

    def test_summary_fields_stay_fixed_and_debug_pages_preserve_lines(self):
        from engagement_model.webcam_demo import render_panel, _debug_pages
        from engagement_model.gaussian import GaussianRuntime
        import cv2
        import numpy as np
        model = GaussianRuntime()
        feature = feature_window(0)
        model(feature)
        face = Mock(last_landmarks=[])
        pose = Mock(last_landmarks=[])
        diagnostics = Mock(lines=['diagnostic-' + str(i) for i in range(90)])
        pages = _debug_pages(feature, {'gaussian': model}, 'gaussian', diagnostics, face)
        self.assertGreater(len(pages), 1)
        self.assertIn('diagnostic-89', [line for page in pages for line in page])
        with patch('engagement_model.webcam_demo.cv2.putText', wraps=cv2.putText) as draw:
            first = render_panel(np.zeros((480,640,3), dtype=np.uint8), feature, {'gaussian':model},
                overlay_smoother=None, face=face, pose=pose, show_overlay=False, debug=True,
                diagnostics=diagnostics, debug_page=0)
        positions = {call.args[1]: call.args[2] for call in draw.call_args_list}
        self.assertEqual(positions['CURRENT DIRECTION'], (660, 65))
        self.assertEqual(positions['10s AGGREGATE'], (1030, 65))
        self.assertEqual(positions['RELATIVE HEAD PITCH / YAW'], (660, 165))
        with patch('engagement_model.webcam_demo.cv2.putText', wraps=cv2.putText) as draw:
            second = render_panel(np.zeros((480,640,3), dtype=np.uint8), feature, {'gaussian':model},
                overlay_smoother=None, face=face, pose=pose, show_overlay=False, debug=True,
                diagnostics=diagnostics, debug_page=1)
        self.assertEqual(first.shape, second.shape)
        self.assertIn('diagnostic-89', [line for page in pages for line in page])
        self.assertEqual(next(call.args[2] for call in draw.call_args_list if call.args[1]=='CURRENT DIRECTION'), (660, 65))

    def test_gaussian_cli_passes_sigma_settings_without_opening_camera(self):
        from engagement_model.webcam_demo import main
        with patch('sys.argv', ['webcam_demo', '--model', 'gaussian', '--sigma-pitch', '12',
                                '--sigma-yaw', '25', '--single-model']), \
             patch('engagement_model.webcam_demo.run') as run:
            main()
        self.assertEqual(run.call_args.kwargs['primary'], 'gaussian')
        self.assertFalse(run.call_args.args[2])
        config = run.call_args.kwargs['gaussian_config']
        self.assertEqual(config.gaussian_sigma_pitch_degrees, 12)
        self.assertEqual(config.gaussian_sigma_yaw_degrees, 25)

    def test_deferred_panel_never_uses_stale_display_score(self):
        from engagement_model.webcam_demo import model_lines
        result = EngagementRuntime(ModelConfig.huang())(feature_window(0, pitch=20))
        result['display_score'] = 100
        lines = '\n'.join(model_lines('huang', result))
        self.assertIn('DEFERRED', lines)
        self.assertIn('looking_down', lines)
        self.assertIn('downward_activity_uncertain', lines)
        self.assertNotIn('100.0/100', lines)

    def test_unavailable_camera_is_released(self):
        from engagement_model.webcam_demo import run
        camera = Mock()
        camera.isOpened.return_value = False
        with patch('engagement_model.webcam_demo.cv2.VideoCapture', return_value=camera):
            with self.assertRaisesRegex(RuntimeError, '웹캠 0'):
                run()
        camera.release.assert_called_once()

    def test_failed_camera_read_closes_models_and_camera(self):
        from engagement_model.webcam_demo import run
        camera, face, pose = Mock(), Mock(), Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (False, None)
        with patch('engagement_model.webcam_demo.cv2.VideoCapture', return_value=camera), \
             patch('engagement_model.webcam_demo.FaceAnalyzer', return_value=face), \
             patch('engagement_model.webcam_demo.PoseAnalyzer', return_value=pose), \
             patch('engagement_model.webcam_demo.cv2.namedWindow'), \
             patch('engagement_model.webcam_demo.cv2.resizeWindow'), \
             patch('engagement_model.webcam_demo.cv2.setMouseCallback'), \
             patch('engagement_model.webcam_demo.cv2.destroyWindow'):
            with self.assertRaisesRegex(RuntimeError, '웹캠 화면'):
                run()
        camera.release.assert_called_once()
        face.close.assert_called_once()
        pose.close.assert_called_once()


def comparison_report(include_gaussian=False):
    """사람 주석 없는 합성 입력 비교. 평가 전 준비 구간은 통계에서 제외한다."""
    scenarios = {
        'front': {'pitch': -2},
        'side': {'pitch': -2, 'yaw': 60},
        'downward_activity_unknown': {'pitch': 20},
        'face_missing': {'faces': 0},
        'return_from_down': {'pitch': -2},
        'stationary_zero_pitch': {'pitch': 0},
        'short_away': {'pitch': 0},
    }
    report = {}
    for name, options in scenarios.items():
        runtime = ComparisonRuntime(include_gaussian=include_gaussian)
        preparation_pitch = 0 if name == 'stationary_zero_pitch' else -2
        for timestamp in range(0, 30001, 100):
            runtime(feature_window(timestamp, pitch=preparation_pitch))
        start = 30100
        if name == 'return_from_down':
            for timestamp in range(start, 60101, 100):
                runtime(feature_window(timestamp, pitch=20))
            start = 60200
        before = deepcopy(runtime.counts)
        for timestamp in range(start, start + 30000, 100):
            current_options = ({'pitch': -2, 'yaw': 60}
                               if name == 'short_away' and timestamp - start < 1000 else
                               {'pitch': -2} if name == 'short_away' else options)
            runtime(feature_window(timestamp, **current_options))
        report[name] = {}
        for method, current in runtime.counts.items():
            delta = {key: value - before[method][key] for key, value in current.items()}
            report[name][method] = {
                'evaluation_frames': delta['frames'],
                'mean_score': round(delta['score_sum'] / delta['scored'], 2) if delta['scored'] else None,
                'deferred_ratio': round(delta['deferred'] / delta['frames'], 4),
            }
    return report


if __name__ == '__main__':
    import sys
    if sys.argv[1:] in (['--compare'], ['--compare-all']):
        import json
        print(json.dumps(comparison_report(include_gaussian=sys.argv[1] == '--compare-all'),
                         indent=2, allow_nan=False))
    else:
        unittest.main()
