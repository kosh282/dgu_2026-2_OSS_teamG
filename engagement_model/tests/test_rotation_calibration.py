"""행렬 보정의 표현 불변성·좌표축·Gaussian 연결과 기존 입력 보존을 검증한다."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np

from engagement_model.rotation_calibration import (
    RotationCalibration, RotationGaussianRuntime, euler_rotation, rotation_euler, calibration_frame)
from engagement_model.runtime import EngagementRuntime
from engagement_model.config import ModelConfig
from engagement_model.tests.test_runtime import feature_window
from engagement_model.tests.test_yaw_calibration import calibrated, observe, face
from engagement_model.tests import test_yaw_calibration as yaw_fixtures
from engagement_model.webcam_demo import DualWebcamCalibration, ScoreTrace
from integration import IntegrationService
from video_recognition.integrated_analysis import VisionPipeline


def ready(angles):
    calibration = RotationCalibration(duration_ms=100, min_samples=2)
    calibration.start()
    for timestamp in (0, 100):
        calibration.update([face(angles)], timestamp)
    return calibration


def feature(calibration, angles, timestamp=200):
    frame = feature_window(timestamp)
    frame['features']['faces'] = [face(angles)]
    calibration.update(frame['features']['faces'], timestamp)
    return frame


class ScoreTraceTests(unittest.TestCase):
    def test_trace_keeps_same_frame_raw_and_both_baselines(self):
        trace = ScoreTrace()
        frame = feature_window(100)
        raw = dict(pitch=-172, yaw=-20, roll=4)
        frame['features']['faces'] = [face(raw)]
        model = SimpleNamespace(latest={'source_angles': dict(pitch=18.6, yaw=-73, roll=161),
                                       'orientation_state': 'looking_down'})
        old = SimpleNamespace(latest={'source_angles': dict(pitch=6, yaw=-22, roll=171),
                                     'orientation_state': 'ambiguous'})
        baselines = {'matrix': raw.copy(), 'euler': raw.copy()}
        trace.update(frame, {'gaussian': model, 'gaussian_axiswise': old}, baselines=baselines)
        raw['yaw'] = 90
        baselines['matrix']['yaw'] = 90
        row = trace.disagreements()['examples'][0]
        self.assertEqual(row['raw']['yaw'], -20)
        self.assertEqual(row['baselines']['matrix']['yaw'], -20)
        self.assertEqual(row['time_ms'], 100)
        self.assertEqual(trace.disagreements()['count'], 1)

    def test_webcam_defaults_to_euler_and_comparison_requires_explicit_matrix(self):
        import inspect
        from engagement_model.webcam_demo import run
        self.assertEqual(inspect.signature(run).parameters['calibration_method'].default, 'euler')
        with self.assertRaisesRegex(ValueError, 'matrix'):
            run(compare_calibration=True)

    def test_current_null_is_distinct_from_high_history_and_zero(self):
        trace = ScoreTrace()
        latest = {'current_direction_score': None, 'score': 98.,
                  'downward_status': 'transition_uncertain',
                  'source_angles': {'pitch': 12., 'yaw': 0.}}
        model = SimpleNamespace(latest=latest)
        trace.update({'start_ms': 0}, {'gaussian': model})
        latest['source_angles']['pitch'] = 0.
        self.assertEqual(trace.records[0]['models']['gaussian']['source_angles']['pitch'], 12.)
        latest.update(current_direction_score=0., score=None, downward_status='none')
        trace.update({'start_ms': 100}, {'gaussian': model, 'gaussian_axiswise': model})
        summary = trace.summary()['gaussian']
        self.assertEqual(summary['current_direction_score']['min'], 0.)
        self.assertEqual(summary['current_direction_score']['null_ratio'], .5)
        self.assertEqual(summary['current_null_history_available'], 1)
        self.assertEqual(trace.records.maxlen, 6000)


class RotationCalibrationTests(unittest.TestCase):
    def test_axiswise_pitch_contract_at_inverted_roll_reference(self):
        # 실제 기록처럼 기준 Roll이 뒤집힌 경우 카메라 축 차감은 Pitch 부호를 반전한다.
        for yaw in (-20, 0, 20):
            for roll in (0, 179):
                origin = dict(pitch=-8, yaw=yaw, roll=roll)
                c = ready(origin)
                model = RotationGaussianRuntime(c)
                for timestamp, delta in ((200, 20), (300, -10), (400, 0)):
                    raw = {**origin, 'pitch': origin['pitch'] + delta}
                    result = model(feature(c, raw, timestamp))
                    self.assertAlmostEqual(result['source_angles']['pitch'], delta)
                    self.assertAlmostEqual(result['source_angles']['yaw'], 0)
                    self.assertAlmostEqual(result['source_angles']['roll'], 0)
                    if delta == 20:
                        self.assertEqual(result['orientation_state'], 'looking_down')
                        self.assertIsNone(result['current_direction_score'])
                    elif delta == 0:
                        self.assertEqual(result['current_direction_score'], 100)

    def test_equivalent_euler_branches_during_normal_motion(self):
        for yaw in (-20, 0, 20):
            c = ready(dict(pitch=-172, yaw=yaw, roll=179))
            for pitch, delta in ((20, -30), (-10, 30), (0, 0)):
                frame = calibration_frame(c.reference)
                rotation = frame @ euler_rotation(dict(pitch=pitch, yaw=delta, roll=0)) @ frame.T @ c.reference
                canonical = rotation_euler(rotation)
                alternate = dict(pitch=(canonical['pitch']+360)%360-180,
                                 yaw=180-canonical['yaw'] if canonical['yaw'] >= 0 else -180-canonical['yaw'],
                                 roll=(canonical['roll']+360)%360-180)
                np.testing.assert_allclose(euler_rotation(alternate), rotation, atol=1e-10)
                c.update([face(alternate)], 200)
                self.assertAlmostEqual(c.relative['pitch'], pitch)
                self.assertAlmostEqual(c.relative['yaw'], delta)
                self.assertAlmostEqual(c.relative['roll'], 0)

    def test_real_head_angles_projection_connects_to_matrix_gaussian(self):
        for baseline in (-20, 0, 20):
            c = ready(yaw_fixtures.IdealPnPFlowTests.estimate(baseline))
            model = RotationGaussianRuntime(c)
            pair = []
            for timestamp, delta in ((200, -30), (300, 30)):
                raw = yaw_fixtures.IdealPnPFlowTests.estimate(baseline+delta)
                result = model(feature(c, raw, timestamp))
                self.assertAlmostEqual(result['source_angles']['yaw'], delta, places=5)
                pair.append(result['current_direction_score'])
            self.assertEqual(pair[0], pair[1])

    def test_real_pipeline_routes_matrix_input_without_changing_legacy_calibration(self):
        c = RotationCalibration(duration_ms=100, min_samples=2)
        model = RotationGaussianRuntime(c)
        service = IntegrationService()
        sid = service.start_session('matrix calibration test')
        capture = SimpleNamespace(read=lambda: np.zeros((100, 100, 3), dtype=np.uint8))
        analyzer = SimpleNamespace(analyze=Mock(), last_face_centers=[(.5, .5)])
        pipeline = VisionPipeline(service, sid, capture, analyzer, model_consumer=model)
        pipeline.head_calibration = DualWebcamCalibration(c)
        pipeline.head_calibration.start()
        origin = dict(pitch=170, yaw=-20, roll=0)
        with patch.object(service, 'elapsed_ms') as elapsed:
            for timestamp, present in ((0, True), (100, True), (200, False), (600, False), (700, True)):
                elapsed.return_value = timestamp
                analyzer.analyze.return_value = [face(origin)] if present else []
                frame = pipeline.process_one()
                self.assertIsNotNone(frame)
                if timestamp == 100 or timestamp == 700:
                    self.assertEqual(model.latest['current_direction_score'], 100)
                    self.assertEqual(model.input_feature['features']['head_calibration_status'], 'ready')
                if not present:
                    self.assertIsNone(model.latest['current_direction_score'])
                # 원본 파이프라인의 기존 보정은 3초 미만이므로 미완료 상태를 유지한다.
                self.assertEqual(frame['features']['head_calibration_status'], 'collecting')
        service.stop_session(sid)

    def test_stale_rotation_is_not_reused_as_current_observation(self):
        c = ready(dict(pitch=170, yaw=20, roll=0))
        result = RotationGaussianRuntime(c)(feature_window(200))
        self.assertIsNone(result['current_direction_score'])
        self.assertIsNone(result['relative_rotation_matrix'])

    def test_downward_policy_and_immediate_front_return_are_preserved(self):
        origin = dict(pitch=170, yaw=20, roll=0)
        c = ready(origin)
        model = RotationGaussianRuntime(c)
        down = rotation_euler(calibration_frame(c.reference) @ euler_rotation(dict(pitch=20, yaw=0, roll=0)) @ calibration_frame(c.reference).T @ c.reference)
        for timestamp in range(200, 700, 100):
            result = model(feature(c, down, timestamp))
            self.assertIsNone(result['current_direction_score'])
            self.assertEqual(result['orientation_state'], 'looking_down')
        self.assertEqual(model(feature(c, origin, 700))['current_direction_score'], 100)

    def test_equivalent_euler_pose_recovers_zero_instead_of_old_penalty(self):
        origin = dict(pitch=180, yaw=30, roll=0)
        equivalent = dict(pitch=0, yaw=150, roll=180)
        old = calibrated(origin)
        _, previous = observe(old, equivalent)
        self.assertEqual(previous['current_direction_score'], 0)
        calibration = ready(origin)
        frame = feature(calibration, equivalent)
        result = RotationGaussianRuntime(calibration)(frame)
        self.assertEqual(result['current_direction_score'], 100)
        for value in result['source_angles'].values():
            self.assertAlmostEqual(value, 0)

    def test_calibrated_frame_relative_rotation_is_invariant_to_reference(self):
        for pitch in (-5, 0, 5):
            for delta_yaw in (-30, 30):
                scores = []
                for baseline_yaw in (-20, 0, 20):
                    origin = dict(pitch=170, yaw=baseline_yaw, roll=4)
                    calibration = ready(origin)
                    current = rotation_euler(calibration_frame(calibration.reference) @ euler_rotation(dict(pitch=pitch, yaw=delta_yaw, roll=0)) @ calibration_frame(calibration.reference).T @ euler_rotation(origin))
                    result = RotationGaussianRuntime(calibration)(feature(calibration, current))
                    self.assertAlmostEqual(result['source_angles']['pitch'], pitch)
                    self.assertAlmostEqual(result['source_angles']['yaw'], delta_yaw)
                    scores.append(result['current_direction_score'])
                self.assertEqual(len(set(scores)), 1)

    def test_same_relative_pitch_opposite_yaw_scores_are_equal(self):
        for baseline in (-20, 0, 20):
            origin = dict(pitch=170, yaw=baseline, roll=0)
            scores = []
            for delta in (-30, 30):
                c = ready(origin)
                current = rotation_euler(calibration_frame(c.reference) @ euler_rotation(dict(pitch=-5, yaw=delta, roll=0)) @ calibration_frame(c.reference).T @ euler_rotation(origin))
                scores.append(RotationGaussianRuntime(c)(feature(c, current))['current_direction_score'])
            self.assertEqual(scores[0], scores[1])

    def test_calibration_accepts_equivalent_encodings_and_camera_signs(self):
        c = RotationCalibration(duration_ms=100, min_samples=2)
        c.start()
        a, b = dict(pitch=180, yaw=30, roll=0), dict(pitch=0, yaw=150, roll=180)
        c.update([face(a)], 0)
        c.update([face(b)], 100)
        self.assertEqual(c.status, 'ready')
        np.testing.assert_allclose(c.relative_matrix, np.eye(3), atol=1e-12)
        for axis in ('pitch', 'yaw', 'roll'):
            for sign in (-1, 1):
                delta = dict(pitch=0, yaw=0, roll=0)
                delta[axis] = sign*5
                current = rotation_euler(calibration_frame(c.reference) @ euler_rotation(delta) @ calibration_frame(c.reference).T @ c.reference)
                c.update([face(current)], 200)
                self.assertAlmostEqual(c.relative[axis], sign*5)

    def test_face_gap_invalid_angles_multi_faces_and_return(self):
        origin = dict(pitch=170, yaw=-20, roll=0)
        c = ready(origin)
        model = RotationGaussianRuntime(c)
        missing = feature_window(200, faces=0)
        c.update([], 200)
        self.assertEqual(c.status, 'ready')
        self.assertIsNone(model(missing)['current_direction_score'])
        self.assertEqual(model(feature(c, origin, 300))['current_direction_score'], 100)
        c.update([face(dict(pitch=0, yaw=float('nan'), roll=0))], 400)
        self.assertEqual(c.status, 'ready')
        self.assertIsNone(c.relative)
        c.update([face(origin), face(origin)], 500)
        self.assertEqual(c.status, 'waiting')
        self.assertIsNone(c.reference)
        c.start()
        self.assertEqual(c.status, 'collecting')

    def test_input_contract_and_legacy_huang_outputs_remain_unchanged(self):
        origin = dict(pitch=170, yaw=20, roll=0)
        c = ready(origin)
        matrix = RotationGaussianRuntime(c)
        for config in (ModelConfig(), ModelConfig.huang()):
            control, observed = EngagementRuntime(config), EngagementRuntime(config)
            for timestamp in range(200, 4200, 100):
                frame = feature(c, origin, timestamp)
                frame['features']['faces'][0]['head_angles_relative_degrees'] = dict(pitch=0, yaw=10, roll=0)
                before = deepcopy(frame)
                expected = control(frame)
                matrix(frame)
                self.assertEqual(frame, before)
                self.assertEqual(expected, observed(frame))

    def test_singular_relative_rotation_round_trip_and_invalid_rotation(self):
        for yaw in (-90, 90):
            rotation = euler_rotation(dict(pitch=12, yaw=yaw, roll=7))
            np.testing.assert_allclose(euler_rotation(rotation_euler(rotation)), rotation, atol=1e-10)
        with self.assertRaises(ValueError):
            rotation_euler(np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            euler_rotation(dict(pitch=0, yaw=True, roll=0))


if __name__ == '__main__':
    unittest.main()
