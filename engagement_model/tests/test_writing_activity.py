"""규칙 기반 필기 간주와 기존 점수·개인별 상태·저장 경계를 검증한다."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from engagement_model.gaussian import GaussianRuntime
from engagement_model.multi_face import MultiFaceGaussianRuntime
from engagement_model.position_tracks import PositionTrackGaussianRuntime
from engagement_model.tests.test_position_tracks import sample
from engagement_model.tests.test_runtime import feature_window
from engagement_model.window_publisher import WindowPublisher
from frontend.live_session import LiveSession
from integration import IntegrationService


def single(t, pitch=0, ear=.3, **kwargs):
    item = feature_window(t, pitch=pitch, **kwargs)
    for face in item['features']['faces']:
        face['eye_aspect_ratio_left'] = face['eye_aspect_ratio_right'] = ear
    return item


def group(t, pitches=(0, 0), ears=(.3, .3)):
    item = sample(t, (.25, .75))
    for face, pitch, ear in zip(item['features']['faces'], pitches, ears):
        face['head_angles_degrees']['pitch'] = pitch
        face['eye_aspect_ratio_left'] = face['eye_aspect_ratio_right'] = ear
    return item


class WritingActivityTests(unittest.TestCase):
    def ready(self):
        model = GaussianRuntime()
        for t in range(0, 10000, 100):
            model(single(t))
        return model

    def test_short_returns_include_exact_five_seconds_and_wait_for_stability(self):
        for elapsed in (1000, 3000, 4900, 5000):
            with self.subTest(return_ms=elapsed):
                model = self.ready()
                for t in range(10000, 10000 + elapsed, 100):
                    result = model(single(t, 20))
                    self.assertEqual(result['activity_state'], 'downward_pending')
                    self.assertEqual(result['current_direction_score'], 100)
                returned = 10000 + elapsed
                result = model(single(returned))
                self.assertEqual(result['activity_state'], 'downward_pending')
                self.assertEqual(result['current_direction_score'], 100)
                for offset in (100, 200, 300):
                    result = model(single(returned + offset))
                self.assertEqual(result['activity_state'], 'writing_assumed')
                self.assertEqual(result['orientation_state'], 'front')
                self.assertEqual(result['activity_down_start_ms'], 10000)
                self.assertEqual(result['activity_down_duration_seconds'], elapsed / 1000)
                self.assertEqual(result['current_direction_score'], 100)

    def test_prolonged_down_uses_original_grace_and_decay_without_rewriting_history(self):
        model = self.ready()
        history = []
        for t in range(10000, 18001, 100):
            result = model(single(t, 20))
            if t == 15000:
                self.assertEqual(result['current_direction_score'], 100)
                self.assertEqual(result['activity_state'], 'downward_pending')
            if t == 15100:
                self.assertEqual(result['current_direction_score'], 99.3)
                self.assertEqual(result['activity_state'], 'downward_prolonged')
            history.append((result, deepcopy(result)))
        self.assertEqual(result['current_direction_score'], 79)
        for t in (18100, 18200, 18300, 18400):
            result = model(single(t))
            self.assertEqual(result['current_direction_score'], 100)
            self.assertNotEqual(result['activity_state'], 'writing_assumed')
        self.assertLess(result['score'], 100)
        self.assertTrue(all(old == snapshot for old, snapshot in history))

    def test_writing_does_not_promote_nonperfect_head_score_to_one_hundred(self):
        model = GaussianRuntime()
        for t in range(0, 10000, 100):
            result = model(single(t, yaw=10))
        reference = result['score']
        self.assertLess(reference, 100)
        for t in range(10000, 13000, 100):
            result = model(single(t, 20, yaw=10))
            self.assertEqual(result['current_direction_score'], reference)
        for t in range(13000, 13401, 100):
            result = model(single(t, yaw=10))
        self.assertEqual(result['activity_state'], 'writing_assumed')
        self.assertEqual(result['current_direction_score'], reference)

    def test_initial_down_without_prior_score_does_not_fabricate_grace_score(self):
        model = GaussianRuntime()
        for t in range(0, 2000, 100):
            result = model(single(t, 20))
            self.assertIsNone(result['current_direction_score'])
            self.assertEqual(result['activity_state'], 'downward_pending')
        for t in range(2000, 2401, 100):
            result = model(single(t))
        self.assertEqual(result['activity_state'], 'writing_assumed')
        self.assertEqual(result['current_direction_score'], 100)

    def test_one_frame_front_pulse_does_not_complete_or_restart_activity_episode(self):
        model = self.ready()
        for t in range(10000, 15000, 100):
            model(single(t, 20))
        self.assertEqual(model(single(15000))['activity_state'], 'downward_pending')
        resumed = model(single(15100, 20))
        self.assertEqual(resumed['activity_state'], 'downward_prolonged')
        self.assertEqual(resumed['activity_down_start_ms'], 10000)
        for t in (15200, 15300, 15400, 15500):
            result = model(single(t))
            self.assertNotEqual(result['activity_state'], 'writing_assumed')

    def test_missing_invalid_quality_gap_and_track_change_cancel_assumption(self):
        for interruption in ('missing', 'quality', 'gap', 'track', 'invalid'):
            with self.subTest(interruption=interruption):
                model = self.ready()
                for t in range(10000, 12000, 100):
                    model(single(t, 20))
                f = single(12000)
                end = 12400
                if interruption == 'missing':
                    f = single(12000, faces=0)
                elif interruption == 'quality':
                    f['features']['faces'][0]['quality']['status'] = 'not_observable'
                elif interruption == 'gap':
                    f = single(13000)
                    end = 13400
                elif interruption == 'track':
                    f = single(12000, track_key=2)
                else:
                    f['feature_version'] = 'unsupported'
                interrupted = model(f)
                if interruption in ('missing', 'quality', 'invalid'):
                    self.assertEqual(interrupted['activity_state'], 'activity_unavailable')
                    self.assertIsNone(interrupted['activity_down_start_ms'])
                for t in range(end, end + 501, 100):
                    result = model(single(t, track_key=2 if interruption == 'track' else 1))
                    self.assertNotEqual(result['activity_state'], 'writing_assumed')

    def test_sideways_return_does_not_count_as_front_return(self):
        model = self.ready()
        for t in range(10000, 11000, 100):
            model(single(t, 20))
        result = model(single(11000, yaw=60))
        self.assertEqual(result['orientation_state'], 'away')
        for t in range(11100, 11600, 100):
            self.assertNotEqual(model(single(t))['activity_state'], 'writing_assumed')

    def test_custom_gaze_policy_does_not_claim_head_pattern_writing(self):
        model = GaussianRuntime(downward_policy=lambda feature, result: (False, 'gaze_custom'))
        for t in range(0, 16000, 100):
            result = model(single(t, 20 if 10000 <= t < 12000 else 0))
            self.assertEqual(result['activity_state'], 'activity_unavailable')


class GroupWritingActivityTests(unittest.TestCase):
    def ready(self, reidentify):
        model = MultiFaceGaussianRuntime(eye_fusion=True) if reidentify else PositionTrackGaussianRuntime(eye_fusion=True)
        if reidentify:
            model.update_identities({1: 'A', 2: 'B'})
        for t in range(0, 10000, 100):
            model(group(t))
        return model

    def test_both_group_paths_all_boundaries_and_return(self):
        for reidentify in (False, True):
            for elapsed in (1000, 3000, 4900, 5000, 5100, 8000):
                with self.subTest(reidentify=reidentify, elapsed=elapsed):
                    model = self.ready(reidentify)
                    # 5.1/8초는 그 시각까지 아래 보기를 관측하고 다음 입력에서 복귀한다.
                    end = 10000 + elapsed
                    stop = end + 100 if elapsed > 5000 else end
                    for t in range(10000, stop, 100):
                        model(group(t, (20, 0)))
                    rows = model.participants()
                    self.assertEqual(rows[1]['current_head_score'], 100)
                    if elapsed > 5000:
                        self.assertEqual(rows[0]['activity_state'], 'downward_prolonged')
                        self.assertEqual(rows[0]['current_head_score'], round(100 - 7 * (elapsed / 1000 - 5), 1))
                    for t in range(stop, stop + 401, 100):
                        model(group(t))
                    rows = model.participants()
                    self.assertEqual(rows[0]['activity_state'], 'writing_assumed' if elapsed <= 5000 else 'downward_prolonged')
                    self.assertEqual(rows[0]['current_head_score'], 100)
                    self.assertIsNone(rows[1]['activity_state'])

    def test_individual_episodes_are_independent(self):
        for reidentify in (False, True):
            with self.subTest(reidentify=reidentify):
                model = self.ready(reidentify)
                for t in range(10000, 19001, 100):
                    model(group(t, (20 if t < 14000 else 0, 20 if t >= 11000 else 0)))
                a, b = model.participants()
                self.assertEqual(a['activity_state'], 'writing_assumed')
                self.assertEqual(b['activity_state'], 'downward_prolonged')
                self.assertEqual((a['activity_down_start_ms'], b['activity_down_start_ms']), (10000, 11000))
                self.assertEqual((a['current_head_score'], b['current_head_score']), (100, 79))

    def test_eye_closure_stays_independent_and_missing_eye_stays_null(self):
        for reidentify in (False, True):
            for ear in (.1, None):
                with self.subTest(reidentify=reidentify, ear=ear):
                    model = self.ready(reidentify)
                    for t in range(10000, 14401, 100):
                        model(group(t, (20 if t < 14000 else 0, 0), (ear, .3)))
                    a, b = model.participants()
                    self.assertEqual(a['activity_state'], 'writing_assumed')
                    self.assertEqual(a['current_head_score'], 100)
                    if ear is None:
                        self.assertIsNone(a['current_eye_score'])
                        self.assertIsNone(a['current_direction_score'])
                    else:
                        self.assertLess(a['current_eye_score'], 100)
                        self.assertEqual(a['current_direction_score'], round(.7 * 100 + .3 * a['current_eye_score'], 1))
                    self.assertEqual(b['current_direction_score'], 100)

    def test_missing_and_changed_identity_do_not_inherit_episode(self):
        for reidentify in (False, True):
            for changed in (False, True):
                with self.subTest(reidentify=reidentify, changed=changed):
                    model = self.ready(reidentify)
                    for t in range(10000, 12000, 100):
                        model(group(t, (20, 0)))
                    f = group(12000)
                    if changed:
                        if reidentify:
                            model.update_identities({1: 'new-A', 2: 'B'})
                        else:
                            f['features']['faces'][0]['transient_track_key'] = 11
                    else:
                        f['features']['faces'] = f['features']['faces'][1:]
                    model(f)
                    for row in model.participants():
                        self.assertNotEqual(row['activity_state'], 'writing_assumed')
                    for t in range(12100, 12601, 100):
                        f = group(t)
                        if changed and not reidentify:
                            f['features']['faces'][0]['transient_track_key'] = 11
                        model(f)
                    self.assertTrue(all(r['activity_state'] != 'writing_assumed' for r in model.participants()))

    def test_recalibration_clears_activity(self):
        for reidentify in (False, True):
            model = self.ready(reidentify)
            for t in range(10000, 12401, 100):
                model(group(t, (20 if t < 12000 else 0, 0)))
            self.assertEqual(model.participants()[0]['activity_state'], 'writing_assumed')
            model.recalibrate()
            self.assertTrue(all(r['activity_state'] == 'activity_unavailable' for r in model.participants()))

    def test_snapshot_carries_optional_fields_but_json_keeps_existing_contract(self):
        for reidentify in (False, True):
            with self.subTest(reidentify=reidentify):
                service = IntegrationService()
                sid = service.start_session('synthetic writing policy')
                model = self.ready(reidentify)
                publisher = WindowPublisher(service, sid, model, start_ms=10000)
                for t in range(10000, 22001, 100):
                    f = group(t, (20 if 11000 <= t < 14000 else 0, 0))
                    f['session_id'] = sid
                    with patch.object(service, 'elapsed_ms', return_value=t):
                        publisher(f)
                live = LiveSession(123, reidentification_enabled=reidentify)
                live.service, live.sid, live.model_runtime = service, sid, model
                live.latest = {'start_ms': 22000}
                live._participant_rows = model.participants()
                with patch.object(service, 'elapsed_ms', return_value=22000):
                    snapshot = live.participant_snapshot()
                self.assertEqual(snapshot[0]['activity_state'], 'writing_assumed')
                snapshot[0]['activity_state'] = 'changed-copy'
                self.assertEqual(model.participants()[0]['activity_state'], 'writing_assumed')
                events = service.get_dashboard_snapshot(sid)['engagement_windows']
                self.assertTrue(events)
                encoded = json.dumps(events)
                for forbidden in ('activity_state', 'activity_down_start_ms', 'writing_assumed'):
                    self.assertNotIn(forbidden, encoded)


if __name__ == '__main__':
    unittest.main()
