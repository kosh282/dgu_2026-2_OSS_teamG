"""개인 상태는 메모리에만 두고 익명 사람·시간 집계를 검사한다."""
import copy
import math
import unittest

from engagement_model.multi_face import MultiFaceGaussianRuntime
from engagement_model.tests.test_runtime import feature_window


def frame(ms, angles=(0, 20)):
    value = feature_window(ms, faces=len(angles))
    for i, (face, yaw) in enumerate(zip(value['features']['faces'], angles), 1):
        face['transient_track_key'] = i
        face['head_angles_degrees'] = dict(pitch=0, yaw=yaw, roll=0)
    return value


class MultiFaceTests(unittest.TestCase):
    def ready(self):
        runtime = MultiFaceGaussianRuntime()
        runtime.update_identities({1: 'A', 2: 'B'})
        for ms in range(0, 3301, 100):
            runtime(frame(ms))
        return runtime

    def test_independent_neutral_and_immutable_input(self):
        runtime = self.ready()
        value = frame(3400, (0, 60))
        original = copy.deepcopy(value)
        result = runtime(value)
        self.assertEqual(value, original)
        rows = runtime.participants()
        self.assertEqual(rows[0]['current_direction_score'], 100)
        self.assertLess(rows[1]['current_direction_score'], 30)
        self.assertEqual(len(result['gaussian_observation']), 2)
        self.assertNotIn('A', repr(result))

    def test_missing_reappearance_retains_baseline_but_resets_time(self):
        runtime = self.ready()
        runtime(frame(3400, (0,)))
        self.assertIsNone(runtime.participants()[1]['current_direction_score'])
        self.assertIsNone(runtime.participants()[1]['score'])
        runtime(frame(3500))
        row = runtime.participants()[1]
        self.assertEqual(row['calibration_status'], 'ready')
        self.assertEqual(row['current_direction_score'], 100)
        self.assertIsNone(row['score'])

    def test_uncertain_duplicate_identity_and_recalibration(self):
        runtime = self.ready()
        runtime.update_identities({1: 'A', 2: 'A'})
        self.assertIsNone(runtime(frame(3400))['score'])
        runtime.update_identities({1: 'A', 2: 'B'})
        runtime(frame(3500))
        runtime.recalibrate()
        self.assertTrue(all(r['calibration_status'] == 'collecting' for r in runtime.participants()))
        self.assertTrue(all(r['score'] is None for r in runtime.participants()))
        runtime.reset()
        self.assertEqual(runtime.participants(), [])

    def test_track_change_preserves_neutral_and_invalid_quality_never_scores(self):
        runtime = self.ready()
        runtime.update_identities({1: 'B', 2: 'A'})
        runtime(frame(3400, (20, 0)))
        self.assertTrue(all(r['current_direction_score'] == 100 for r in runtime.participants()))
        self.assertTrue(all(r['score'] is None for r in runtime.participants()))
        bad = frame(3500, (20, 0))
        bad['features']['faces'][0]['quality']['status'] = 'not_observable'
        runtime(bad)
        row = next(r for r in runtime.participants() if r['id'] == 'B')
        self.assertIsNone(row['current_direction_score'])
        self.assertEqual(row['calibration_status'], 'ready')
        runtime(frame(3600, (20, 0)))
        self.assertEqual(next(r for r in runtime.participants() if r['id'] == 'B')['current_direction_score'], 100)

    def test_unconfirmed_identity_does_not_create_calibration_or_mutate_feature(self):
        runtime = MultiFaceGaussianRuntime()
        value = frame(0)
        original = copy.deepcopy(value)
        result = runtime(value)
        self.assertEqual(runtime.participants(), [])
        self.assertIsNone(result['score'])
        self.assertIn('identity_unconfirmed', result['reason_codes'])
        self.assertEqual(value, original)
        runtime.update_identities({1: 'A'})
        for ms in range(100, 3301, 100):
            runtime(frame(ms, (0,)))
        rows = runtime.participants()
        rows[0]['id'] = 'changed'
        self.assertEqual(runtime.participants()[0]['id'], 'A')
        runtime({})
        self.assertIsNone(runtime.participants()[0]['score'])
        self.assertIsNone(runtime.latest['angles'])

    def test_ttl_and_capacity(self):
        runtime = self.ready()
        runtime(frame(124000, ()))
        self.assertEqual(runtime.participants(), [])
        for i in range(40):
            runtime.update_identities({1: str(i)})
            runtime(frame(125000 + i * 100, (0,)))
        self.assertLessEqual(len(runtime.participants()), 32)

    def test_anonymous_person_time_weighting_and_null(self):
        runtime = MultiFaceGaussianRuntime()
        segments = [(0, 5000, True, 99, [(0, 0), (0, 60)]),
                    (5000, 10000, True, 0, [(0, 60)])]
        summary = runtime.aggregate_window(segments, 0, 10000)
        low = 100 * math.exp(-4.5)
        self.assertEqual(summary['score'], round((.25 * (100 + low) + .75 * low) / 1.25, 1))
        self.assertEqual(summary['observable_duration_ms'], 10000)
        self.assertEqual(summary['scoreable_duration_ms'], 10000)
        self.assertIsNone(runtime.aggregate_window([(0, 10000, True, None, [])], 0, 10000)['score'])
        self.assertIsNone(runtime.aggregate_window([(0, 1000, True, None, [(0, 0)])], 0, 10000)['score'])
