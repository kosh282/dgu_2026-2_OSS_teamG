"""실제 카메라 없이 데모 시간 상태와 보류 동작을 확인한다."""

import unittest

from engagement_model.gaussian import GaussianRuntime


SESSION_ID = "c4b1f6c0-511a-46be-9a39-29e714654001"


def feature_window(timestamp_ms, *, pitch=0.0, yaw=0.0, faces=1, track_key=1):
    face_list = [{
        "quality": {"status": "valid", "reason_codes": []},
        "head_angles_relative_degrees": {
            "pitch": pitch, "yaw": yaw, "roll": 0.0,
        },
        "transient_track_key": track_key,
    } for _ in range(faces)]
    return {
        "session_id": SESSION_ID,
        "start_ms": timestamp_ms,
        "end_ms": timestamp_ms,
        "feature_version": "experimental-0.9",
        "quality": {
            "status": "valid" if faces else "not_observable",
            "reason_codes": [] if faces else ["face_not_found"],
        },
        "features": {
            "head_calibration_status": "ready",
            "faces": face_list,
        },
    }


class RuntimeTests(unittest.TestCase):
    def test_front_score_increases_with_continuous_observations(self):
        runtime = GaussianRuntime()
        result = None
        for timestamp in range(1000, 6100, 100):
            result = runtime(feature_window(timestamp))
        self.assertEqual(result["orientation_state"], "front")
        self.assertEqual(result["score"], 100.0)
        self.assertEqual(result["front_seconds"], 5.0)


    def test_multiple_faces_are_deferred_and_do_not_get_a_score(self):
        result = GaussianRuntime()(feature_window(1000, faces=2))
        self.assertIsNone(result["score"])
        self.assertIn("multiple_faces_unsupported", result["reason_codes"])

    def test_missing_face_is_not_converted_to_zero(self):
        result = GaussianRuntime()(feature_window(1000, faces=0))
        self.assertIsNone(result["score"])
        self.assertIn("face_not_found", result["reason_codes"])

    def test_long_observation_gap_resets_score_history(self):
        runtime = GaussianRuntime()
        for timestamp in range(1000, 5100, 100):
            runtime(feature_window(timestamp))
        missing = runtime(feature_window(6000, faces=0))
        self.assertIsNone(missing["score"])
        resumed = runtime(feature_window(6100))
        self.assertIsNone(resumed["score"])
        self.assertEqual(resumed["current_direction_score"], 100)
        self.assertEqual(resumed["front_seconds"], 0.0)

    def test_tracking_key_change_starts_new_timing_state(self):
        runtime = GaussianRuntime()
        first = runtime(feature_window(1000))
        switched = runtime(feature_window(1100, track_key=2))
        self.assertIsNone(first["score"])
        self.assertIsNone(switched["score"])
        self.assertEqual(switched["front_seconds"], 0.0)
        self.assertIn("tracking_key_changed", switched["reason_codes"])

    def test_downward_look_holds_then_gradually_reduces_score(self):
        runtime = GaussianRuntime()
        for timestamp in range(1000, 6100, 100):
            result = runtime(feature_window(timestamp))
        self.assertEqual(result["score"], 100.0)

        for timestamp in range(6200, 11200, 100):
            result = runtime(feature_window(timestamp, pitch=20.0))
        self.assertEqual(result["orientation_state"], "looking_down")
        self.assertEqual(result["score"], 100.0)  # 5초 유예 구간
        self.assertEqual(result["looking_down_seconds"], 4.9)

        for timestamp in range(11200, 16200, 100):
            result = runtime(feature_window(timestamp, pitch=20.0))
        self.assertLess(result["score"], 100.0)
        self.assertGreater(result["score"], 80.0)
        self.assertIn("looking_down_score_decay", result["reason_codes"])

        for timestamp in range(16200, 21201, 100):
            result = runtime(feature_window(timestamp, pitch=20.0))
        self.assertEqual(result["current_direction_score"], 80.0)  # 현재 점수의 감쇠 상한
        self.assertGreater(result["score"], 80)  # 이동 평균에는 이전 점수가 남는다

    def test_down_state_uses_hysteresis_near_threshold(self):
        runtime = GaussianRuntime()
        runtime(feature_window(1000))
        down = runtime(feature_window(1100, pitch=16.0))
        self.assertEqual(down["orientation_state"], "looking_down")
        still_down = runtime(feature_window(1200, pitch=12.0))
        self.assertEqual(still_down["orientation_state"], "looking_down")
        recovered = runtime(feature_window(1300, pitch=10.0))
        self.assertEqual(recovered["orientation_state"], "front")


if __name__ == "__main__":
    unittest.main()
