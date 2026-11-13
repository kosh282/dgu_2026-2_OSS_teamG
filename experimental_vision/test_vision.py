import json
import tempfile
import unittest
from pathlib import Path

from integration import IntegrationService
from video_recognition.integrated_analysis import VisionPipeline


class Image:
    shape = (480, 640, 3)


class Capture:
    def read(self):
        return Image()


class Analyzer:
    def analyze(self, frame):
        return [{'eye_openness': 0.2}, {'eye_openness': 0.3}]

    def close(self):
        pass


class VisionTests(unittest.TestCase):

    def test_frame_quality_distinguishes_missing_streams(self):
        from video_recognition.integrated_analysis import feature_quality
        face = {'eye_openness': 0.2, 'head_angles_degrees': {'pitch': 0},
                'quality': {'status': 'valid'}}
        self.assertEqual(feature_quality([face])['status'], 'valid')
        self.assertEqual(feature_quality([])['status'], 'not_observable')
        face['eye_openness'] = None
        self.assertEqual(feature_quality([face])['status'], 'partial')

    def test_periodic_angles_remain_continuous_at_wrap(self):
        from video_recognition.face_analysis import periodic_angles
        left = periodic_angles({'pitch': 179.0, 'yaw': 0.0, 'roll': 0.0})
        right = periodic_angles({'pitch': -179.0, 'yaw': 0.0, 'roll': 0.0})
        self.assertAlmostEqual(left['pitch']['cos'], right['pitch']['cos'])
        self.assertLess(abs(left['pitch']['sin'] - right['pitch']['sin']), 0.04)
        self.assertIsNone(periodic_angles(None))

    def test_points_outside_image_are_not_observable(self):
        from types import SimpleNamespace
        from video_recognition.face_analysis import points_in_frame
        points = [SimpleNamespace(x=0.2, y=0.3), SimpleNamespace(x=1.1, y=0.5)]
        self.assertTrue(points_in_frame(points, [0]))
        self.assertFalse(points_in_frame(points, [0, 1]))

    def test_uncalibrated_features_and_quality_reach_model(self):
        from video_recognition.face_analysis import periodic_angles
        angles = {'pitch': 179.0, 'yaw': -27.0, 'roll': 4.0}
        class FaceAnalyzer:
            def analyze(self, frame):
                return [{'eye_openness': 0.2, 'head_angles_degrees': angles,
                         'head_orientation_periodic': periodic_angles(angles),
                         'head_position': {'x': 0.5, 'y': 0.5},
                         'quality': {'status': 'valid', 'reason_codes': []}}]
            def close(self):
                pass
        service = IntegrationService()
        sid = service.start_session('test')
        received = []
        pipeline = VisionPipeline(service, sid, Capture(), FaceAnalyzer(), received.append)
        pipeline.process_one()
        face = received[0]['features']['faces'][0]
        self.assertEqual(received[0]['features']['head_calibration_status'], 'waiting')
        self.assertIsNone(face['head_angles_relative_degrees'])
        self.assertEqual(face['head_angles_degrees'], angles)
        self.assertIsNotNone(face['head_orientation_periodic'])
        self.assertEqual(received[0]['quality']['status'], 'valid')  # 얼굴 관측만으로 품질 판정
        pipeline.close()

    def test_head_calibration_handles_angle_wrap_and_lost_face(self):
        from video_recognition.head_calibration import HeadCalibration
        calibration = HeadCalibration(duration_ms=3000, min_samples=4)
        face = {'quality': {'status': 'valid'}, 'head_angles_degrees': {'pitch': 179.0, 'yaw': -27.0, 'roll': 1.0}}
        self.assertEqual(calibration.update([face], 0), 'waiting')
        self.assertIsNone(face['head_angles_relative_degrees'])
        calibration.start()
        for time_ms, pitch in [(0, 179.0), (1000, -179.0), (2000, 179.0), (3000, -179.0)]:
            face['head_angles_degrees']['pitch'] = pitch
            calibration.update([face], time_ms)
        self.assertEqual(calibration.status, 'ready')
        face['head_angles_degrees'].update(pitch=-160.0, yaw=-17.0)
        calibration.update([face], 3100)
        self.assertAlmostEqual(face['head_angles_relative_degrees']['pitch'], 20.0, places=4)
        self.assertAlmostEqual(face['head_angles_relative_degrees']['yaw'], 10.0, places=4)
        self.assertEqual(calibration.update([], 3200), 'waiting')
        calibration.update([face], 3300)
        self.assertIsNone(face['head_angles_relative_degrees'])

    def test_head_calibration_rejects_multiple_or_unstable_faces(self):
        from video_recognition.head_calibration import HeadCalibration
        calibration = HeadCalibration(duration_ms=3000, min_samples=4)
        calibration.start()
        first = {'quality': {'status': 'valid'}, 'head_angles_degrees': {'pitch': 180.0, 'yaw': 0.0, 'roll': 0.0}}
        calibration.update([first], 0)
        calibration.update([first, dict(first)], 500)
        self.assertEqual(calibration.samples, [])
        for time_ms, yaw in [(1000, 0), (2000, 90), (3000, 0), (4000, 90)]:
            first['head_angles_degrees']['yaw'] = yaw
            calibration.update([first], time_ms)
        self.assertEqual(calibration.status, 'collecting')
        self.assertIsNone(first['head_angles_relative_degrees'])

    def test_head_calibration_centers_all_axes_and_keeps_intermediate_values(self):
        from video_recognition.head_calibration import HeadCalibration
        calibration = HeadCalibration(duration_ms=3000, min_samples=4)
        calibration.start()
        face = {'quality': {'status': 'valid'}, 'head_angles_degrees': {'pitch': 179.0, 'yaw': -27.0, 'roll': 4.0}}
        for capture_ms in (0, 1000, 2000, 3000):
            calibration.update([face], capture_ms)
        for axis in ('pitch', 'yaw', 'roll'):
            self.assertAlmostEqual(face['head_angles_relative_degrees'][axis], 0.0)
        for capture_ms, change in ((3100, -20), (3200, 0), (3300, 20)):
            face['head_angles_degrees']['yaw'] = -27.0 + change
            calibration.update([face], capture_ms)
            self.assertAlmostEqual(face['head_angles_relative_degrees']['yaw'], change)
        face['head_angles_degrees'].update(pitch=-176.0, yaw=-27.0, roll=9.0)
        calibration.update([face], 3400)
        self.assertAlmostEqual(face['head_angles_relative_degrees']['pitch'], 5.0)
        self.assertAlmostEqual(face['head_angles_relative_degrees']['roll'], 5.0)

    def test_model_delivery_status_and_separate_intermediate_file(self):
        service = IntegrationService()
        sid = service.start_session('test')
        received = []
        with tempfile.TemporaryDirectory(dir='.') as directory:
            intermediate = Path(directory) / 'intermediate.jsonl'
            output = Path(directory) / 'session.json'
            pipeline = VisionPipeline(service, sid, Capture(), Analyzer(), received.append, intermediate)
            result = pipeline.process_one()
            self.assertEqual(len(result['features']['faces']), 2)
            self.assertEqual(received, [result])
            from video_recognition.integrated_analysis import _without_transient_tracking
            stored = json.loads(intermediate.read_text(encoding='utf-8'))
            self.assertEqual(stored, _without_transient_tracking(result))
            self.assertNotIn('transient_track_key', json.dumps(stored))
            self.assertNotIn('upper_bodies', result['features'])
            self.assertNotIn('target_pose_hz', result['features']['sampling'])
            self.assertEqual(result['feature_version'], 'experimental-0.9')
            self.assertEqual(service.get_dashboard_snapshot(sid)['status']['vision'], 'running')
            pipeline.close()
            service.stop_session(sid)
            service.save_session(sid, output)
            saved = json.loads(output.read_text(encoding='utf-8'))
            self.assertNotIn('eye_openness', json.dumps(saved))
            self.assertEqual(service.get_dashboard_snapshot(sid)['status']['vision'], 'stopped')

    def test_capture_failure_reports_unavailable(self):
        class BrokenCapture:
            def read(self):
                raise RuntimeError('capture failed')
        service = IntegrationService()
        sid = service.start_session('test')
        pipeline = VisionPipeline(service, sid, BrokenCapture(), Analyzer())
        with self.assertRaises(RuntimeError):
            pipeline.process_one()
        self.assertEqual(service.get_dashboard_snapshot(sid)['status']['vision'], 'unavailable')
        pipeline.close()

    def test_movement_requires_one_detection_and_short_interval(self):
        service = IntegrationService()
        sid = service.start_session('test')
        pipeline = VisionPipeline(service, sid, Capture(), Analyzer())
        one = [{'head_position': {'x': 0.2, 'y': 0.4}}]
        self.assertIsNone(pipeline._movement('head', one, 'head_position', 100))
        two = [{'head_position': {'x': 0.5, 'y': 0.8}}]
        self.assertAlmostEqual(pipeline._movement('head', two, 'head_position', 200)['distance_normalized'], 0.5)
        self.assertIsNone(pipeline._movement('head', two * 2, 'head_position', 300))
        self.assertIsNone(pipeline._movement('head', one, 'head_position', 400))
        self.assertIsNone(pipeline._movement('head', one, 'head_position', 1500))
        pipeline.close()


    def test_preview_uses_same_capture_as_analysis(self):
        import numpy as np
        from video_recognition.webcam_test import PreviewCapture
        image = np.full((4, 5, 3), 73, dtype=np.uint8)
        class Source:
            def read(self):
                return image
        class InspectAnalyzer:
            def analyze(self, frame):
                self.image = frame.image.copy()
                return []
            def close(self):
                pass
        service = IntegrationService()
        sid = service.start_session('test')
        preview = PreviewCapture(Source())
        analyzer = InspectAnalyzer()
        pipeline = VisionPipeline(service, sid, preview, analyzer)
        pipeline.process_one()
        np.testing.assert_array_equal(preview.last_frame, analyzer.image)
        pipeline.close()

    def test_landmark_overlay_preserves_original_and_skips_out_of_frame(self):
        import numpy as np
        from video_recognition.preview import draw_landmarks
        source = np.zeros((100, 100, 3), dtype=np.uint8)
        face = [(0.5, 0.5, 0.0), (1.5, 0.5, 0.0)]
        result = draw_landmarks(source, [face])
        self.assertFalse(source.any())
        self.assertTrue(result[50, 50].any())
        self.assertFalse(result[:, 99].any())

    def test_class_window_preview_renders_and_exits_on_q(self):
        import sys
        import numpy as np
        from unittest.mock import patch
        from video_recognition import integrated_analysis as module
        class Source:
            def __init__(self, hwnd):
                pass
            def read(self):
                return np.zeros((10, 10, 3), dtype=np.uint8)
        class Model:
            def __init__(self, *args, **kwargs):
                self.last_landmarks = []
            def analyze(self, frame):
                return []
            def close(self):
                pass
        with tempfile.TemporaryDirectory(dir='.') as directory:
            output = str(Path(directory) / 'session.json')
            arguments = ['test', '--hwnd', '123', '--session', output, '--quiet']
            with patch.object(sys, 'argv', arguments), \
                 patch.object(module, 'WindowCapture', Source), \
                 patch.object(module, 'FaceAnalyzer', Model), \
                 patch('cv2.namedWindow'), patch('cv2.resizeWindow'), \
                 patch('ctypes.windll', __import__('types').SimpleNamespace(user32=__import__('types').SimpleNamespace(GetSystemMetrics=lambda _: 1000)), create=True), patch('cv2.moveWindow'), patch('cv2.imshow') as display, \
                 patch('cv2.waitKey', return_value=ord('q')), \
                 patch('cv2.destroyAllWindows'):
                module.main()
            self.assertEqual(display.call_count, 1)
            self.assertTrue(Path(output).exists())

    def test_window_capture_waits_for_new_frame_and_reports_minimized(self):
        import threading
        import time
        from video_recognition.window_capture import FrameNotAvailable, WindowCapture
        class FakeUser32:
            minimized = False
            def IsWindow(self, hwnd):
                return True
            def IsIconic(self, hwnd):
                return self.minimized
        class FakeControl:
            def is_finished(self):
                return False
        capture = WindowCapture.__new__(WindowCapture)
        capture.user32 = FakeUser32()
        capture.condition = threading.Condition()
        capture.hwnd = 123
        capture.timeout = 0.001
        capture.closed = capture.stopped = False
        capture.started_at = time.monotonic()
        capture.control = FakeControl()
        capture.latest_frame = object()
        capture.generation, capture.consumed = 1, 0
        self.assertIs(capture.read(), capture.latest_frame)
        with self.assertRaises(FrameNotAvailable):
            capture.read()
        capture.user32.minimized = True
        with self.assertRaisesRegex(RuntimeError, '최소화'):
            capture.read()


if __name__ == '__main__':
    unittest.main()
