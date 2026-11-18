import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from video_recognition.face_analysis import FaceAnalyzer, _layout_changed


class FullFrameScheduleTests(unittest.TestCase):
    def test_layout_comparison_ignores_order_and_small_jitter(self):
        original = [(0, 0, 100, 100), (100, 0, 200, 100)]
        self.assertFalse(_layout_changed(original, [(101, 1, 201, 101), (1, 1, 101, 101)]))
        self.assertTrue(_layout_changed(original, [(0, 0, 100, 100)]))
        self.assertTrue(_layout_changed(original, [(0, 0, 100, 100), (150, 0, 250, 100)]))

    def test_refresh_and_no_stale_full_frame_faces(self):
        analyzer = FaceAnalyzer.__new__(FaceAnalyzer)
        point = SimpleNamespace(x=.5, y=.5, z=0.)
        face = SimpleNamespace(face_landmarks=[[point] * 478], face_blendshapes=[],
                               facial_transformation_matrixes=[])
        empty = SimpleNamespace(face_landmarks=[], face_blendshapes=[], facial_transformation_matrixes=[])
        analyzer.detector = Mock()
        analyzer.detector.detect_for_video.return_value = face
        analyzer.tile_detector = Mock()
        analyzer.tile_detector.detect.return_value = empty
        analyzer.mp = SimpleNamespace(Image=lambda **kwargs: kwargs['data'],
                                      ImageFormat=SimpleNamespace(SRGB=1))
        analyzer.pose_source = 'face-transform'
        analyzer.max_faces = 20
        analyzer.show_landmarks = True
        analyzer.last_timestamp = -1
        analyzer.full_frame_interval_ms = 4000
        analyzer._last_full_frame_ms = None
        analyzer._layout_regions = []
        analyzer._layout_timestamp = None
        analyzer._layout_frame_size = None

        def frame(ms, width=100):
            return SimpleNamespace(image=np.zeros((100, width, 3), dtype=np.uint8),
                                   capture_ms=ms, width=width, height=100)

        with patch('video_recognition.face_analysis.detect_layout_regions',
                   side_effect=[[], [(5, 5, 55, 55)], [(5, 5, 55, 55)], [(5, 5, 55, 55)]]):
            self.assertEqual(len(analyzer.analyze(frame(0))), 1)
            self.assertEqual(analyzer.analyze(frame(100)), [])
            self.assertEqual(analyzer.last_landmarks, [])
            self.assertEqual(len(analyzer.analyze(frame(1000))), 1)
            self.assertEqual(analyzer.analyze(frame(1200)), [])
            self.assertEqual(len(analyzer.analyze(frame(5000))), 1)
            self.assertEqual(len(analyzer.analyze(frame(5100, width=120))), 1)
        self.assertEqual(analyzer.detector.detect_for_video.call_count, 4)
        self.assertEqual(analyzer.tile_detector.detect.call_count, 28)


if __name__ == '__main__':
    unittest.main()
