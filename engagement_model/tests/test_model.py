"""실제 카메라 없이 입력 계약과 예외 상황, 점수 계산을 검증합니다."""

from copy import deepcopy
import unittest

from engagement_model.config import ModelConfig
from engagement_model.input_adapter import HeadAngles, parse_feature_window
from engagement_model.scoring import classify_orientation


def feature_window():
    """현재 VisionPipeline 형식에 맞는 최소 단일 얼굴 입력을 만듭니다."""
    return {
        "session_id": "c4b1f6c0-511a-46be-9a39-29e714654001",
        "start_ms": 100, "end_ms": 100, "feature_version": "experimental-0.9",
        "quality": {"status": "partial", "reason_codes": ["insufficient_observation"]},
        "features": {
            "head_calibration_status": "ready",
            "faces": [{"quality": {"status": "valid", "reason_codes": []},
                       "head_angles_relative_degrees": {"pitch": 0, "yaw": 0, "roll": 0},
                       "transient_track_key": 1}],
        },
    }


class ConfigTests(unittest.TestCase):

    def test_gaussian_softened_pitch_validation_rejects_invalid_settings(self):
        """실제 완화 모드의 경계·계수·유한값 검증을 유지한다."""
        for kwargs in (
            {'front_pitch_degrees': 1},
            {'gaussian_pitch_deadband_degrees': 10},
            {'gaussian_pitch_deadband_degrees': 11},
            {'gaussian_pitch_small_angle_scale': 0},
            {'gaussian_pitch_small_angle_scale': 1.1},
            {'gaussian_pitch_deadband_degrees': -1},
            {'gaussian_pitch_small_angle_scale': True},
            {'gaussian_pitch_deadband_degrees': float('nan')},
            {'gaussian_pitch_small_angle_scale': float('inf')},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ModelConfig.gaussian(**kwargs)

    def test_gaussian_softened_accepts_valid_pitch_settings(self):
        """기본값과 유효한 사용자 정의 완화 설정을 허용한다."""
        for kwargs in ({}, {'gaussian_pitch_deadband_degrees': 0, 'gaussian_pitch_small_angle_scale': 1},
                       {'front_pitch_degrees': 1, 'gaussian_pitch_deadband_degrees': 0.25,
                        'gaussian_pitch_small_angle_scale': 0.5}):
            with self.subTest(kwargs=kwargs):
                config = ModelConfig.gaussian(**kwargs)
                self.assertEqual(config.scoring_method, 'gaussian')
                self.assertEqual(config.gaussian_angle_mode, 'softened')
                self.assertEqual(config.gaussian_sigma_pitch_degrees, 15)
                self.assertEqual(config.gaussian_sigma_yaw_degrees, 20)

    def test_configurable_window(self):
        """집계와 출력 간격을 바꾸고 기본 설정은 그대로 유지할 수 있습니다."""
        self.assertEqual(ModelConfig().window_seconds, 10)
        self.assertEqual(ModelConfig().output_interval_seconds, 1)
        self.assertEqual(ModelConfig(window_seconds=20, output_interval_seconds=4).window_seconds, 20)

    def test_invalid_config(self):
        """분모가 0이 되거나 유한하지 않은 설정을 거부합니다."""
        for kwargs in ({"away_yaw_degrees": 15}, {"gaussian_sigma_yaw_degrees": 0},
                       {"min_observable_ratio": 1.1}, {"window_seconds": True},
                       {"output_interval_seconds": 11}, {"scoring_method": "unsupported"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ModelConfig(**kwargs)


class AdapterTests(unittest.TestCase):
    def test_face_valid_with_missing_body_and_no_mutation(self):
        """얼굴 전용 입력을 관측하고 원본은 변경하지 않습니다."""
        data = feature_window()
        original = deepcopy(data)
        result = parse_feature_window(data)
        self.assertTrue(result.observable)
        self.assertEqual(result.angles, HeadAngles(0, 0, 0))
        self.assertEqual(result.source_angles, HeadAngles(0, 0, 0))
        self.assertEqual(data, original)

    def test_yaw_neutral_deadband_and_soft_sensitivity(self):
        """작은 yaw는 완만히 줄이고 큰 좌우 회전은 대부분 보존합니다."""
        data = feature_window()
        data["features"]["faces"][0]["head_angles_relative_degrees"]["yaw"] = 1.5
        deadbanded = parse_feature_window(data)
        self.assertEqual(deadbanded.source_angles.yaw, 1.5)
        self.assertEqual(deadbanded.angles.yaw, 0.0)

        data["features"]["faces"][0]["head_angles_relative_degrees"]["yaw"] = 10.0
        softened = parse_feature_window(data)
        self.assertEqual(softened.angles.yaw, 6.0)

        data["features"]["faces"][0]["head_angles_relative_degrees"]["yaw"] = -45.0
        large_turn = parse_feature_window(data)
        self.assertEqual(large_turn.source_angles.yaw, -45.0)
        self.assertEqual(large_turn.angles.yaw, -39.75)

    def test_missing_and_multiple_faces(self):
        """미검출과 다중 얼굴은 0도 대신 관측 불가로 반환합니다."""
        for count, reason in ((0, "face_not_found"), (2, "multiple_faces_unsupported")):
            data = feature_window()
            data["features"]["faces"] *= count
            result = parse_feature_window(data)
            self.assertFalse(result.observable)
            self.assertIn(reason, result.reason_codes)
            self.assertIsNone(result.angles)

    def test_failed_calibration_and_low_quality(self):
        """정면 기준과 얼굴 품질이 부족하면 점수를 만들지 않습니다."""
        for field in ("calibration", "face_quality", "frame_quality"):
            data = feature_window()
            if field == "calibration":
                data["features"]["head_calibration_status"] = "waiting"
            elif field == "face_quality":
                data["features"]["faces"][0]["quality"]["status"] = "not_observable"
            else:
                data["quality"]["status"] = "not_observable"
            with self.subTest(field=field):
                self.assertFalse(parse_feature_window(data).observable)

    def test_invalid_angles_are_unobservable(self):
        """결측·잘못된 숫자는 참여도 0점으로 바꾸지 않습니다."""
        for value in (None, True, float("nan"), float("inf"), 181, "0"):
            data = feature_window()
            data["features"]["faces"][0]["head_angles_relative_degrees"]["pitch"] = value
            with self.subTest(value=value):
                self.assertFalse(parse_feature_window(data).observable)

    def test_malformed_envelope(self):
        """잘못된 버전·시각·자료 구조는 입력 계약 오류로 구분합니다."""
        for key, value in (("session_id", "bad"), ("start_ms", True),
                           ("end_ms", 101), ("feature_version", "old"),
                           ("features", None), ("quality", None)):
            data = feature_window()
            data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                parse_feature_window(data)




if __name__ == "__main__":
    unittest.main()
