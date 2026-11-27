"""하위 20%의 인원 선정·시간 기준·익명 이벤트 호환성을 검증한다."""
from collections import deque
import json
import math
from pathlib import Path
import tempfile
import unittest

from integration import IntegrationService
from engagement_model.multi_face import MultiFaceGaussianRuntime
from engagement_model.position_tracks import PositionTrackGaussianRuntime
from engagement_model.tests.test_multi_face import frame
from engagement_model.window_publisher import WindowPublisher


class Lower20Tests(unittest.TestCase):
    def pipeline(self, count=4, *, positional=False, fusion=True):
        service = IntegrationService()
        sid = service.start_session('synthetic lower-20 test')
        service.elapsed_ms = lambda _: 15000
        cls = PositionTrackGaussianRuntime if positional else MultiFaceGaussianRuntime
        model = cls(eye_fusion=fusion)
        model.update_identities({i: f'S{i:03}' for i in range(1, count + 1)})
        return service, sid, model, WindowPublisher(service, sid, model)

    def feed(self, sid, publisher, timestamp, yaws):
        value = frame(timestamp, yaws)
        value['session_id'] = sid
        for i, face in enumerate(value['features']['faces']):
            face['head_position'] = {'x': .15 + .2 * (i % 4), 'y': .2 + .2 * (i // 4)}
        return publisher(value)

    def prepare(self, count=4, *, positional=False, fusion=True):
        items = self.pipeline(count, positional=positional, fusion=fusion)
        _, sid, _, publisher = items
        for t in range(0, 9001, 100):
            self.feed(sid, publisher, t, [0 if t < 4000 else i * 8 for i in range(count)])
        return items

    def extension(self, service, sid, model):
        event = service.get_dashboard_snapshot(sid)['engagement_windows'][-1]
        return event, event['payload']['model_extensions'][model.model_id]

    def test_ceil_counts_and_null_zero_validation(self):
        for n, k in ((1, 1), (2, 1), (4, 1), (5, 1), (6, 2), (10, 2)):
            with self.subTest(n=n):
                fields = MultiFaceGaussianRuntime._lower_20_fields(range(n))
                self.assertEqual(fields['lower_20_valid_count'], n)
                self.assertEqual(fields['lower_20_count'], k)
                self.assertEqual(fields['lower_20_mean_score'], sum(range(k)) / k)
        fields = MultiFaceGaussianRuntime._lower_20_fields([None, True, float('nan'), float('inf'), -1, 101, 0, 80])
        self.assertEqual(fields, dict(lower_20_mean_score=0, lower_20_count=1, lower_20_valid_count=2))
        self.assertEqual(MultiFaceGaussianRuntime._lower_20_fields([None]),
                         dict(lower_20_mean_score=None, lower_20_count=0, lower_20_valid_count=0))

    def test_real_pipeline_counts_both_identity_modes_and_head_eye_paths(self):
        for positional in (False, True):
            for fusion in (False, True):
                for n in (1, 2, 4, 10):
                    with self.subTest(positional=positional, fusion=fusion, n=n):
                        service, sid, model, _ = self.prepare(n, positional=positional, fusion=fusion)
                        event, fields = self.extension(service, sid, model)
                        k = max(1, math.ceil(n * .2))
                        scores = sorted(row['current_direction_score'] for row in model.participants())
                        self.assertEqual(fields['lower_20_valid_count'], n)
                        self.assertEqual(fields['lower_20_count'], k)
                        self.assertAlmostEqual(fields['lower_20_mean_score'], round(sum(scores[:k]) / k, 1), delta=.1)
                        self.assertEqual((event['start_ms'], event['end_ms']), (4000, 9000))
                        self.assertEqual(fields['window_seconds'], 5)
                        self.assertIsNotNone(event['payload']['model_output'])

    def test_missing_people_are_excluded_without_zero_and_all_missing_is_null(self):
        for positional in (False, True):
            service, sid, model, publisher = self.prepare(positional=positional)
            for t in range(9100, 10001, 100):
                self.feed(sid, publisher, t, [0, 8, 16])
            _, fields = self.extension(service, sid, model)
            self.assertEqual(fields['lower_20_valid_count'], 3)
            self.assertGreater(fields['lower_20_mean_score'], 0)
            for t in range(10100, 11001, 100):
                self.feed(sid, publisher, t, [])
            _, fields = self.extension(service, sid, model)
            self.assertIsNone(fields['lower_20_mean_score'])
            self.assertEqual(fields['lower_20_count'], 0)
            self.assertEqual(fields['lower_20_valid_count'], 0)

    def test_return_without_sufficient_window_coverage_is_excluded(self):
        for positional in (False, True):
            service, sid, model, publisher = self.pipeline(2, positional=positional)
            for t in range(0, 9001, 100):
                yaws = [0, 40] if t < 4000 or t >= 8600 else [0]
                self.feed(sid, publisher, t, yaws)
            _, fields = self.extension(service, sid, model)
            self.assertEqual(fields['lower_20_valid_count'], 1)
            self.assertEqual(fields['lower_20_mean_score'], 100)

    def test_same_time_kernel_before_selecting_people_not_instantaneous_tail(self):
        model = MultiFaceGaussianRuntime()
        # 한 사람은 앞 3초 100점, 뒤 2초 0점: (0.18*100 + 0.32*0)/0.5 = 36.
        for step in (100, 50):
            samples = deque((t, True, (None, None, 100 if t < 7000 else 0))
                            for t in range(4000, 9000, step))
            model._people = {'synthetic': {'window_samples': samples}}
            self.assertEqual(model._lower_20_window(4000, 9000)['lower_20_mean_score'], 36)
        timestamps = [4000]
        while timestamps[-1] < 8900:
            timestamps.append(timestamps[-1] + (50 if len(timestamps) % 2 else 150))
        # 행동 경계 시각을 포함하는 불규칙 표본도 같은 시간 적분을 사용한다.
        timestamps = sorted(set(timestamps + [7000, 8900]))
        model._people = {'synthetic': {'window_samples': deque(
            (t, True, (None, None, 100 if t < 7000 else 0)) for t in timestamps)}}
        self.assertEqual(model._lower_20_window(4000, 9000)['lower_20_mean_score'], 36)
        model._people['other'] = {'window_samples': deque(
            (t, True, (None, None, 10 if t < 7000 else 90)) for t in timestamps)}
        # 두 사람의 구간 점수는 36과 61.2. 순간 최저(0)나 시점별 최저 평균(3.6)이 아니다.
        self.assertEqual(model._lower_20_window(4000, 9000),
                         dict(lower_20_mean_score=36, lower_20_count=1, lower_20_valid_count=2))

    def test_partial_window_and_stale_observation_use_existing_hold_limit(self):
        model = MultiFaceGaussianRuntime()
        model._people = {'synthetic': {'window_samples': deque(
            (t, True, (None, None, 50)) for t in range(0, 1000, 100))}}
        self.assertEqual(model._lower_20_window(0, 1000)['lower_20_mean_score'], 50)
        self.assertIsNone(model._lower_20_window(0, 1500)['lower_20_mean_score'])

    def test_unconfirmed_identity_expired_input_and_recalibration(self):
        service, sid, model, publisher = self.prepare(2)
        model.recalibrate('S001')
        self.assertEqual(model._lower_20_window(4000, 9000)['lower_20_valid_count'], 1)
        model.update_identities({})
        for t in range(9100, 10001, 100):
            self.feed(sid, publisher, t, [0, 8])
        _, fields = self.extension(service, sid, model)
        self.assertIsNone(fields['lower_20_mean_score'])
        self.assertEqual(model._lower_20_window(6000, 11000)['lower_20_valid_count'], 0)

    def test_sample_storage_is_bounded_and_invalid_input_clears_history(self):
        _, sid, model, publisher = self.prepare()
        for person in model._people.values():
            self.assertLessEqual(len(person['window_samples']), 52)
        invalid = frame(9100)
        invalid['feature_version'] = 'unsupported'
        invalid['session_id'] = sid
        publisher(invalid)
        self.assertTrue(all(not person['window_samples'] for person in model._people.values()))

    def test_event_save_reload_and_frontend_snapshot_preserve_optional_fields(self):
        service, sid, model, _ = self.prepare(10)
        event, fields = self.extension(service, sid, model)
        self.assertEqual(fields['lower_20_policy'], 'individual-window-ceil-v1')
        self.assertEqual(fields['lower_20_count'], 2)
        self.assertEqual(event['payload']['observation']['eligible_count'], None)
        encoded = json.dumps(event)
        for key in ('window_samples', 'transient_track_key', 'S001', 'S010'):
            self.assertNotIn(key, encoded)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.json'
            service.save_session(sid, path)
            reopened = IntegrationService()
            restored = reopened.load_session(path)
            stored = reopened.get_dashboard_snapshot(restored)['engagement_windows'][-1]
            self.assertEqual(stored['payload'], event['payload'])


if __name__ == '__main__':
    unittest.main()
