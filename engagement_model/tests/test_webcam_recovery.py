"""미디어 저장 없이 실제 파이프라인의 보정→미검출→복귀 경로를 검증한다."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from integration import IntegrationService
from video_recognition.integrated_analysis import VisionPipeline
from video_recognition.head_calibration import HeadCalibration
from engagement_model.gaussian import GaussianRuntime
from engagement_model.webcam_demo import WebcamHeadCalibration, panel_lines, render_tracking, GaussianDiagnostics
from engagement_model.tests.test_runtime import feature_window


def face():
    return {'quality': {'status': 'valid', 'reason_codes': []},
            'head_angles_degrees': {'pitch': 10.0, 'yaw': 5.0, 'roll': 0.0}}


class WebcamRecoveryTests(unittest.TestCase):
    def test_mouse_and_automatic_calibration_requests_work_without_keyboard(self):
        import cv2
        from engagement_model.webcam_demo import CalibrationControls, CALIBRATE_BUTTON
        controls = CalibrationControls(True)
        self.assertTrue(controls.consume(-1))
        self.assertFalse(controls.consume(-1))
        left, top, _, _ = CALIBRATE_BUTTON
        controls.mouse(cv2.EVENT_LBUTTONDOWN, left + 5, top + 5, 0, None)
        self.assertTrue(controls.consume(-1))
        controls.mouse(cv2.EVENT_LBUTTONDOWN, 1000, 1000, 0, None)
        self.assertFalse(controls.consume(-1))
        self.assertTrue(controls.consume(ord(' ')))
        self.assertFalse(controls.consume(ord(' ')))

    def test_drop_snapshot_keeps_baseline_and_angles_after_missing_then_expires(self):
        model = GaussianRuntime()
        diagnostics = GaussianDiagnostics()
        baseline = {'pitch': 179.0, 'yaw': -5.0}
        for ms, yaw in [(0, 0), (100, 70)]:
            feature = feature_window(ms, yaw=yaw)
            face_data = feature['features']['faces'][0]
            face_data['head_angles_degrees'] = {'pitch': 179, 'yaw': yaw-5}
            face_data['head_angles_smoothed_degrees'] = {'pitch': 179, 'yaw': yaw-5}
            model(feature)
            diagnostics.update(feature, model, baseline=baseline)
        self.assertEqual(diagnostics.events[0]['after']['baseline']['yaw'], -5)
        baseline['yaw'] = 99
        self.assertEqual(diagnostics.events[0]['after']['baseline']['yaw'], -5)
        missing = feature_window(200, faces=0)
        model(missing)
        diagnostics.update(missing, model, baseline=baseline)
        self.assertIn('HISTORICAL DROP', '\n'.join(diagnostics.lines))
        self.assertIn('USED=(0.0, 64.75)', '\n'.join(diagnostics.lines))
        changed = feature_window(300, track_key=2)
        model(changed)
        diagnostics.update(changed, model, baseline=baseline)
        self.assertIn('changed=True', '\n'.join(diagnostics.lines))
        missing = feature_window(8200, faces=0)
        model(missing)
        diagnostics.update(missing, model)
        self.assertNotIn('HISTORICAL DROP', '\n'.join(diagnostics.lines))
        self.assertEqual(len(diagnostics.events), 0)
        diagnostics.reset()
        self.assertIsNone(diagnostics.last_track)

    def test_gaussian_diagnostics_uses_same_actual_input_and_preserves_config(self):
        model = GaussianRuntime()
        diagnostics = GaussianDiagnostics()
        initial = feature_window(0)
        model(initial)
        diagnostics.update(initial, model)
        feature = feature_window(100, yaw=70)
        result = model(feature)
        before = dict(result)
        diagnostics.update(feature, model)
        lines = '\n'.join(diagnostics.lines)
        self.assertIn('USED P/Y: 0.00/64.75', lines)
        self.assertIn('0.53/9.74/26.98', lines)
        self.assertIn('driver=Yaw', lines)
        self.assertIn('score=-99.5', lines)
        self.assertEqual(model.config.gaussian_sigma_yaw_degrees, 20)
        self.assertEqual(result, before)
        self.assertNotIn('USED P/Y', '\n'.join(panel_lines(feature, {'gaussian': model}, diagnostics=diagnostics)))
        self.assertIn('USED P/Y', '\n'.join(panel_lines(feature, {'gaussian': model}, debug=True, diagnostics=diagnostics)))

    def test_diagnostics_pitch_drop_missing_and_return(self):
        model = GaussianRuntime()
        diagnostics = GaussianDiagnostics()
        for feature in [feature_window(0), feature_window(100, pitch=-50)]:
            model(feature)
            diagnostics.update(feature, model)
        self.assertIn('driver=Pitch', '\n'.join(diagnostics.lines))
        for feature in [feature_window(200, faces=0), feature_window(300, pitch=30)]:
            model(feature)
            diagnostics.update(feature, model)
            self.assertIn('candidates: unavailable', '\n'.join(diagnostics.lines))
            self.assertIsNone(diagnostics.previous)
        returned = feature_window(400)
        model(returned)
        diagnostics.update(returned, model)
        self.assertIn('100.00/100.00/100.00', '\n'.join(diagnostics.lines))
        self.assertIn('Delta unavailable', '\n'.join(diagnostics.lines))
        diagnostics.reset()
        self.assertEqual(diagnostics.lines, [])

    def ready(self):
        calibration = WebcamHeadCalibration(duration_ms=100, min_samples=2)
        calibration.start()
        calibration.update([face()], 0)
        calibration.update([face()], 100)
        self.assertEqual(calibration.status, 'ready')
        return calibration

    def test_pipeline_recovery_preserves_baseline_and_resumes_first_score(self):
        service = IntegrationService()
        sid = service.start_session('recovery test')
        runtime = GaussianRuntime()
        analyzer = SimpleNamespace(analyze=Mock(), last_face_centers=[(.5, .5)])
        capture = SimpleNamespace(read=lambda: np.zeros((100, 100, 3), dtype=np.uint8))
        pipeline = VisionPipeline(service, sid, capture, analyzer, model_consumer=runtime)
        pipeline.head_calibration = self.ready()
        baseline = dict(pipeline.head_calibration.baseline)
        with patch.object(service, 'elapsed_ms') as elapsed:
            for ms, state in [(200, 'valid'), (300, 'invalid_angles'), (400, 'bad_quality'),
                              (500, 'missing'), (2000, 'missing'),
                              (2100, 'valid'), (2200, 'valid')]:
                elapsed.return_value = ms
                observed = face()
                if state == 'invalid_angles':
                    observed['head_angles_degrees'] = None
                elif state == 'bad_quality':
                    observed['quality']['status'] = 'not_observable'
                analyzer.analyze.return_value = [] if state == 'missing' else [observed]
                feature = pipeline.process_one()
                self.assertIsNotNone(feature)
                self.assertEqual(pipeline.head_calibration.baseline, baseline)
                self.assertEqual(feature['features']['head_calibration_status'], 'ready')
                if state == 'valid':
                    self.assertIsNotNone(feature['features']['faces'][0]['head_angles_relative_degrees'])
                    self.assertEqual(runtime.latest['current_direction_score'], 100)
                    self.assertEqual(runtime.latest['orientation_state'], 'front')
                else:
                    self.assertIsNone(runtime.latest['current_direction_score'])
                    self.assertEqual(runtime.latest['orientation_state'], 'not_observable')
        self.assertEqual(analyzer.analyze.call_count, 7)
        service.stop_session(sid)

    def test_ambiguous_or_invalid_input_requires_recalibration(self):
        for faces, reason in [([face(), face()], 'multiple_faces_detected')]:
            calibration = self.ready()
            calibration.update(faces, 200)
            self.assertIsNone(calibration.baseline)
            self.assertEqual(calibration.last_reason_code, reason)
            calibration.update([face()], 300)
            self.assertEqual(calibration.status, 'waiting')
        calibration = self.ready()
        calibration.baseline['pitch'] = float('nan')
        calibration.update([face()], 200)
        self.assertEqual(calibration.last_reason_code, 'calibration_baseline_invalid')
        self.assertIsNone(calibration.baseline)

    def test_invalid_observation_clears_relative_angles_but_keeps_baseline(self):
        for reason in ('head_angles_unavailable', 'face_quality_not_observable'):
            calibration = self.ready()
            baseline = dict(calibration.baseline)
            observed = face()
            observed['head_angles_relative_degrees'] = {'pitch': 0, 'yaw': 0, 'roll': 0}
            if reason == 'head_angles_unavailable':
                observed['head_angles_degrees'] = None
            else:
                observed['quality']['status'] = 'not_observable'
            calibration.update([observed], 200)
            self.assertEqual(calibration.baseline, baseline)
            self.assertEqual(calibration.last_reason_code, reason)
            self.assertIsNone(observed['head_angles_relative_degrees'])
            returned = face()
            calibration.update([returned], 300)
            self.assertIsNotNone(returned['head_angles_relative_degrees'])
            self.assertIsNone(calibration.last_reason_code)

    def test_tracking_overlay_disappears_and_resumes_without_stale_face(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        detected = {'features': {'faces': [dict(face(), head_angles_relative_degrees={})]}}
        missing = {'features': {'faces': []}}
        landmarks = [[(.3, .4, 0), (.7, .8, 0)]]
        first = render_tracking(frame, detected, landmarks)
        absent = render_tracking(frame, missing, landmarks)
        returned = render_tracking(frame, detected, landmarks)
        self.assertTrue(np.any(first[48:100, 48:114]))
        self.assertFalse(np.any(absent[48:100, 48:114]))
        np.testing.assert_array_equal(first, returned)
        self.assertFalse(np.any(frame))

    def test_collecting_gap_still_restarts_and_team_calibration_is_unchanged(self):
        calibration = WebcamHeadCalibration()
        calibration.start()
        calibration.update([face()], 0)
        calibration.update([], 100)
        self.assertEqual(calibration.samples, [])
        original = HeadCalibration(duration_ms=100, min_samples=2)
        original.start()
        original.update([face()], 0)
        original.update([face()], 100)
        original.update([], 200)
        self.assertIsNone(original.baseline)

    def test_normal_panel_explains_recalibration(self):
        feature = {'features': {'head_calibration_status': 'waiting',
                   'head_calibration_diagnostics': {'last_reason_code': 'multiple_faces_detected'}}}
        lines = panel_lines(feature, {'gaussian': GaussianRuntime()})
        self.assertTrue(any('multiple_faces_detected' in line for line in lines))


if __name__ == '__main__':
    unittest.main()
