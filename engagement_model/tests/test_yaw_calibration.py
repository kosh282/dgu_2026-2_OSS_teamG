"""보정 기준의 이동, 좌우 대칭, Euler 표현의 한계를 구분하는 합성 진단.

동일 회전의 다른 Euler 표현에 관한 검사는 현재 보정 방식의 한계를 재현한다.
테스트 통과가 실제 웹캠의 각도 정확성이나 문제 해결을 의미하지 않는다.
"""

from copy import deepcopy
import math
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from video_recognition.head_calibration import HeadCalibration
from engagement_model.config import ModelConfig
from engagement_model.gaussian import GaussianRuntime
from engagement_model.tests.test_runtime import feature_window
from engagement_model.webcam_demo import WebcamHeadCalibration


def face(angles, smoothed=None):
    result = {'quality': {'status': 'valid', 'reason_codes': []},
              'head_angles_degrees': dict(angles), 'transient_track_key': 1}
    if smoothed is not None:
        result['head_angles_smoothed_degrees'] = dict(smoothed)
    return result


def calibrated(angles, calibration_type=HeadCalibration, smoothed=None):
    calibration = calibration_type(duration_ms=100, min_samples=2)
    calibration.start()
    for timestamp in (0, 100):
        calibration.update([face(angles, smoothed)], timestamp)
    if calibration.status != 'ready':
        raise AssertionError(calibration.diagnostics())
    return calibration


def observe(calibration, angles, smoothed=None, timestamp=200):
    observed = face(angles, smoothed)
    calibration.update([observed], timestamp)
    feature = feature_window(timestamp)
    feature['features'].update(faces=[observed], head_calibration_status=calibration.status)
    before = deepcopy(feature)
    result = GaussianRuntime()(feature)
    if feature != before:
        raise AssertionError('모델이 FeatureWindow를 수정했습니다')
    return observed, result


def rotation_matrix(pitch, yaw, roll):
    """고정된 합성 좌표계의 Rz*Ry*Rx. 실제 얼굴의 회전 정답으로 사용하지 않는다."""
    p, y, r = map(math.radians, (pitch, yaw, roll))
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


class YawCalibrationTests(unittest.TestCase):
    def test_fixed_raw_endpoints_reverse_imbalance_when_baseline_changes(self):
        scores = {}
        for baseline in (-20, 20):
            calibration = calibrated(dict(pitch=0, yaw=baseline, roll=0))
            scores[baseline] = []
            for timestamp, yaw in ((200, -40), (300, 40)):
                observed, result = observe(calibration, dict(pitch=0, yaw=yaw, roll=0), timestamp=timestamp)
                self.assertAlmostEqual(observed['head_angles_relative_degrees']['yaw'], yaw-baseline)
                scores[baseline].append(result['current_direction_score'])
        self.assertGreater(scores[-20][0], scores[-20][1])
        self.assertGreater(scores[20][1], scores[20][0])
        self.assertEqual(scores[-20][0], scores[20][1])
        self.assertEqual(scores[-20][1], scores[20][0])

    def test_shifted_baseline_and_observations_preserve_relative_angle_and_score(self):
        for calibration_type in (HeadCalibration, WebcamHeadCalibration):
            for delta in (5, 15, 30):
                reference = None
                for baseline in (-20, 0, 20, -179, 179):
                    with self.subTest(calibration=calibration_type.__name__, delta=delta, baseline=baseline):
                        calibration = calibrated(dict(pitch=5, yaw=baseline, roll=0), calibration_type)
                        pair = []
                        for timestamp, sign in ((200, -1), (300, 1)):
                            yaw = (baseline+sign*delta+180) % 360-180
                            observed, result = observe(calibration, dict(pitch=5, yaw=yaw, roll=0), timestamp=timestamp)
                            self.assertAlmostEqual(observed['head_angles_relative_degrees']['yaw'], sign*delta)
                            self.assertAlmostEqual(result['source_angles']['pitch'], 0)
                            pair.append(result['current_direction_score'])
                        self.assertEqual(pair[0], pair[1])
                        if reference is None:
                            reference = pair
                        self.assertEqual(pair, reference)

    def test_smoothed_angles_define_calibration_and_model_input_when_available(self):
        for calibration_type in (HeadCalibration, WebcamHeadCalibration):
            with self.subTest(calibration=calibration_type.__name__):
                calibration = calibrated(dict(pitch=0, yaw=-20, roll=0), calibration_type,
                                         smoothed=dict(pitch=0, yaw=20, roll=0))
                self.assertAlmostEqual(calibration.baseline['yaw'], 20)
                observed, result = observe(calibration, dict(pitch=0, yaw=-35, roll=0),
                                          smoothed=dict(pitch=0, yaw=10, roll=0))
                self.assertAlmostEqual(observed['head_angles_relative_degrees']['yaw'], -10)
                self.assertAlmostEqual(observed['head_angles_relative_raw_degrees']['yaw'], -55)
                self.assertEqual(observed['head_angle_model_input_source'], 'smoothed')
                self.assertEqual(result['source_angles']['yaw'], -10)
                self.assertEqual(result['gaussian_observation'], (0, -6))
                self.assertEqual(result['current_direction_score'], 95.6)

    def test_same_pitch_and_opposite_relative_yaw_are_symmetric_in_both_modes(self):
        for mode in ('relative', 'softened'):
            config = ModelConfig.gaussian(gaussian_angle_mode=mode)
            for pitch in (-10, 0, 5, 10):
                for yaw in (0, 2, 10, 30, 80):
                    with self.subTest(mode=mode, pitch=pitch, yaw=yaw):
                        a = GaussianRuntime(config)(feature_window(0, pitch=pitch, yaw=-yaw))
                        b = GaussianRuntime(config)(feature_window(0, pitch=pitch, yaw=yaw))
                        self.assertIsNotNone(a['current_direction_score'])
                        self.assertEqual(a['current_direction_score'], b['current_direction_score'])

    def test_equivalent_euler_encodings_expose_axiswise_calibration_limitation(self):
        # 같은 행렬인데 Euler 축별 차이는 0이 아니다. 현재 한계를 고정해 진단하는 검사다.
        first = dict(pitch=180, yaw=30, roll=0)
        equivalent = dict(pitch=0, yaw=150, roll=180)
        np.testing.assert_allclose(rotation_matrix(**first), rotation_matrix(**equivalent), atol=1e-12)
        for calibration_type in (HeadCalibration, WebcamHeadCalibration):
            with self.subTest(calibration=calibration_type.__name__):
                calibration = calibrated(first, calibration_type)
                observed, result = observe(calibration, equivalent)
                self.assertAlmostEqual(observed['head_angles_relative_degrees']['yaw'], 120)
                self.assertAlmostEqual(observed['head_angles_relative_degrees']['pitch'], -180)
                self.assertEqual(result['current_direction_score'], 0)
        # 행렬 기준의 상대 회전은 항등행렬이다. 운영 보정 로직은 이 테스트에서 변경하지 않는다.
        relative_rotation = rotation_matrix(**first).T @ rotation_matrix(**equivalent)
        np.testing.assert_allclose(relative_rotation, np.eye(3), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
