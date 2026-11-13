import unittest
from unittest.mock import Mock, patch

import numpy as np

from integration import IntegrationService
from video_recognition.integrated_analysis import VisionPipeline
from video_recognition.preview import PreviewCapture


class FrameEfficiencyTests(unittest.TestCase):
    def test_skipped_frames_are_not_resized_and_keep_sample_timing(self):
        service = IntegrationService()
        sid = service.start_session('test')
        clock = [0]
        service.elapsed_ms = lambda _sid: clock[0]
        capture = Mock()
        capture.read.return_value = np.zeros((80, 120, 3), dtype=np.uint8)
        analyzer = Mock()
        analyzer.analyze.return_value = []
        analyzer.last_face_centers = []
        pipeline = VisionPipeline(service, sid, capture, analyzer, max_analysis_dimension=60)
        with patch.object(pipeline, '_analysis_image', wraps=pipeline._analysis_image) as resize:
            first = pipeline.process_one()
            clock[0] = 50
            self.assertIsNone(pipeline.process_one())
            clock[0] = 100
            second = pipeline.process_one()
        self.assertEqual(resize.call_count, 2)
        self.assertEqual(analyzer.analyze.call_count, 2)
        self.assertEqual([call.args[0].frame_id for call in analyzer.analyze.call_args_list], [0, 2])
        self.assertEqual((first['start_ms'], second['start_ms']), (0, 100))
        self.assertEqual(second['features']['sampling']['source_size'], {'width': 120, 'height': 80})
        self.assertEqual(second['features']['sampling']['analysis_size'], {'width': 60, 'height': 40})
        clock[0] = 450
        third = pipeline.process_one()
        self.assertEqual(third['features']['sampling']['last_face_interval_ms'], 350)
        self.assertEqual(third['features']['sampling']['face_gap_over_300ms'], 1)

    def test_preview_can_be_toggled_without_changing_source_frame(self):
        image = np.ones((2, 2, 3), dtype=np.uint8)
        capture = Mock()
        capture.read.return_value = image
        preview = PreviewCapture(capture, keep_frame=False)
        self.assertIs(preview.read(), image)
        self.assertIsNone(preview.last_frame)
        preview.keep_frame = True
        self.assertIs(preview.read(), image)
        self.assertIsNot(preview.last_frame, image)
        image[:] = 0
        self.assertEqual(int(preview.last_frame[0, 0, 0]), 1)
        preview.keep_frame = False
        self.assertIsNone(preview.last_frame)


if __name__ == '__main__':
    unittest.main()
