"""실제 카메라 없이 입력 계약과 예외 상황, 점수 계산을 검증합니다."""

from copy import deepcopy
import unittest

from engagement_model.config import ModelConfig
from engagement_model.input_adapter import HeadAngles, Observation, parse_feature_window
from engagement_model.scoring import behavior_score, classify_orientation, orientation_severity


def feature_window():
    """현재 VisionPipeline 형식에 맞는 최소 단일 얼굴 입력을 만듭니다."""
    return {
        "session_id": "c4b1f6c0-511a-46be-9a39-29e714654001",
        "start_ms": 100, "end_ms": 100, "feature_version": "experimental-0.8",
        "quality": {"status": "partial", "reason_codes": ["insufficient_observation"]},
        "features": {
            "head_calibration_status": "ready", "upper_bodies": [],
            "faces": [{"quality": {"status": "valid", "reason_codes": []},
                       "head_angles_relative_degrees": {"pitch": 0, "yaw": 0, "roll": 0},
                       "transient_track_key": 1}],
        },
    }


def observation(pitch=0, yaw=0, roll=0):
    """점수 함수 시험용으로 상대 각도를 지정한 관측을 만듭니다."""
    return Observation("test", 100, "experimental-0.8", HeadAngles(pitch, yaw, roll))


class ConfigTests(unittest.TestCase):
    def test_configurable_window(self):
        """집계와 출력 간격을 바꾸고 기본 설정은 그대로 유지할 수 있습니다."""
        self.assertEqual(ModelConfig().window_seconds, 10)
        self.assertEqual(ModelConfig().output_interval_seconds, 2)
        self.assertEqual(ModelConfig(window_seconds=20, output_interval_seconds=4).window_seconds, 20)

    def test_invalid_config(self):
        """분모가 0이 되거나 유한하지 않은 설정을 거부합니다."""
        for kwargs in ({"away_yaw_degrees": 60}, {"full_away_seconds": 3},
                       {"front_full_seconds": 0}, {"angle_penalty": float("nan")},
                       {"min_observable_ratio": 1.1}, {"window_seconds": True},
                       {"front_bonus": 20}, {"output_interval_seconds": 11}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ModelConfig(**kwargs)


class AdapterTests(unittest.TestCase):
    def test_face_valid_with_missing_body_and_no_mutation(self):
        """상체 결측은 얼굴 관측을 막지 않으며 원본 입력은 변하지 않습니다."""
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
            self.assertIsNone(behavior_score(result))

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
                self.assertIsNone(behavior_score(parse_feature_window(data)))

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


class ScoringTests(unittest.TestCase):
    def test_front_hold_increases_score(self):
        """정면 유지 점수는 기본값에서 상한까지 증가합니다."""
        self.assertEqual(behavior_score(observation()), 90)
        self.assertEqual(behavior_score(observation(), front_seconds=5), 100)
        self.assertEqual(behavior_score(observation(), front_seconds=100), 100)

    def test_short_away_preserves_previous_score(self):
        """유예 시간 이내에는 새 감점 없이 직전 점수를 유지합니다."""
        self.assertEqual(behavior_score(observation(yaw=60), away_seconds=3,
                                        previous_stable_score=95), 95)
        self.assertIsNone(behavior_score(observation(yaw=60), away_seconds=2))

    def test_severity_boundaries_and_symmetry(self):
        """좌우 부호와 상관없이 이탈 강도가 커지며 상한을 넘지 않습니다."""
        self.assertEqual(orientation_severity(HeadAngles(0, 30, 0)), 0)
        self.assertEqual(orientation_severity(HeadAngles(0, -45, 0)), 0.5)
        self.assertEqual(orientation_severity(HeadAngles(-30, 0, 0)), 0.5)
        self.assertEqual(orientation_severity(HeadAngles(0, 180, 0)), 1)

    def test_persistent_away_monotonic_and_bounded(self):
        """지속 시간·이탈 강도가 늘면 점수가 증가하지 않습니다."""
        short = behavior_score(observation(yaw=45), away_seconds=4)
        long = behavior_score(observation(yaw=45), away_seconds=8)
        severe = behavior_score(observation(yaw=60), away_seconds=8)
        self.assertGreater(short, long)
        self.assertGreater(long, severe)
        self.assertEqual(behavior_score(observation(yaw=60), away_seconds=100), 0)
        self.assertGreaterEqual(severe, 0)

    def test_down_is_deferred_even_with_side_turn_or_long_duration(self):
        """초기 아래 보기는 이전 안정 점수가 없으면 기준 점수를 만들지 않습니다."""
        for yaw in (0, 80):
            sample = observation(pitch=20, yaw=yaw)
            self.assertEqual(classify_orientation(sample.angles), "looking_down")
            self.assertIsNone(orientation_severity(sample.angles))
            self.assertIsNone(behavior_score(sample, away_seconds=100,
                                             previous_stable_score=90))

    def test_ambiguous_and_missing(self):
        """경계 구간과 관측 불가는 점수를 확정하지 않습니다."""
        self.assertEqual(classify_orientation(observation(yaw=20).angles), "ambiguous")
        self.assertIsNone(behavior_score(observation(yaw=20)))
        self.assertIsNone(orientation_severity(None))

    def test_bad_numeric_arguments(self):
        """시간과 이전 점수에 잘못된 숫자가 들어오면 알립니다."""
        for kwargs in ({"front_seconds": -1}, {"away_seconds": float("nan")},
                       {"previous_stable_score": 101}, {"away_seconds": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                behavior_score(observation(), **kwargs)

    def test_custom_weights(self):
        """가중치를 설정에서 바꾸면 새 설정대로 계산됩니다."""
        config = ModelConfig(front_base_score=80, front_bonus=5)
        self.assertEqual(behavior_score(observation(), front_seconds=5, config=config), 85)


if __name__ == "__main__":
    unittest.main()
