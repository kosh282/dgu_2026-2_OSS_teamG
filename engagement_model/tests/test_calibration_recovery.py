import unittest
from copy import deepcopy
from video_recognition.recovering_calibration import RecoveringHeadCalibration
from engagement_model.multi_face import MultiFaceGaussianRuntime
from engagement_model.tests.test_multi_face import frame


def face(track=1, yaw=0):
    return {'quality': {'status':'valid', 'reason_codes':[]},
            'head_angles_degrees': {'pitch':0., 'yaw':float(yaw), 'roll':0.},
            'transient_track_key':track}


class RecoveryTests(unittest.TestCase):
    def ready(self):
        cal = RecoveringHeadCalibration(duration_ms=200, min_samples=3)
        cal.start()
        for t in (0,100,200): cal.update([face()], t)
        self.assertEqual(cal.status, 'ready')
        return cal

    def test_short_loss_recovers_only_same_track_without_stale_angles(self):
        cal = self.ready()
        self.assertEqual(cal.update([], 300), 'recovering')
        self.assertIsNotNone(cal.baseline)
        item = face(yaw=10)
        self.assertEqual(cal.update([item], 400), 'ready')
        self.assertAlmostEqual(item['head_angles_relative_degrees']['yaw'], 10)
        cal.update([], 500)
        changed = face(track=2, yaw=30)
        self.assertEqual(cal.update([changed], 600), 'collecting')
        self.assertIsNone(changed['head_angles_relative_degrees'])
        self.assertIsNone(cal.baseline)

    def test_long_gap_and_multiple_faces_restart_automatically(self):
        for faces, t in (([face()], 3301), ([face(), face(2)], 300)):
            cal = self.ready()
            self.assertEqual(cal.update(faces, t), 'collecting')
            self.assertIsNone(cal.baseline)
            self.assertTrue(all(f['head_angles_relative_degrees'] is None for f in faces))

    def test_new_track_during_collection_does_not_mix_samples(self):
        cal = RecoveringHeadCalibration(duration_ms=200, min_samples=3)
        cal.start()
        cal.update([face(1, 0)], 0)
        cal.update([face(2, 30)], 100)
        cal.update([face(2, 30)], 200)
        self.assertEqual(cal.status, 'collecting')
        cal.update([face(2, 30)], 300)
        self.assertEqual(cal.status, 'ready')
        self.assertAlmostEqual(cal.baseline['yaw'], 30)

    def test_multi_reidentified_long_gap_recalibrates_person_only(self):
        model=MultiFaceGaussianRuntime()
        model.update_identities({1:'a',2:'b'})
        for t in range(0,3301,100): model(frame(t))
        original=deepcopy(model._people['b']['calibration'].baseline)
        for t in range(3400, 6801, 100): model(frame(t, (0,)))
        result=model(frame(6900))
        rows={r['id']:r for r in model.participants()}
        self.assertEqual(rows['a']['calibration_status'],'ready')
        self.assertEqual(rows['b']['calibration_status'],'collecting')
        self.assertIsNone(rows['b']['current_direction_score'])
        self.assertIsNone(model._people['b']['calibration'].baseline)
        self.assertIsNotNone(original)

if __name__ == '__main__': unittest.main()
