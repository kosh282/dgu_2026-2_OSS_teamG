"""실제 얼굴 데이터 없이 재식별 상태와 저장 경계를 확인한다."""
import unittest
import numpy as np
from video_recognition.face_reidentification import IdentityGallery


class GalleryTests(unittest.TestCase):
    def confirm(self, gallery, vector, start=0):
        result = None
        for offset in (0, 500, 1000):
            result = gallery.observe(vector, start + offset)
        return result

    def test_returning_face_keeps_id_and_different_face_gets_new_id(self):
        gallery = IdentityGallery()
        first = self.confirm(gallery, [1, 0])
        self.assertEqual(first['state'], 'new')
        gallery.observe(None, 1500)
        returned = self.confirm(gallery, [1, 0], 5000)
        self.assertEqual(returned['identity'], first['identity'])
        self.assertEqual(returned['state'], 'reconnected')
        other = self.confirm(gallery, [0, 1], 8000)
        self.assertNotEqual(first['identity'], other['identity'])

    def test_missing_or_invalid_embedding_never_assigns_id(self):
        gallery = IdentityGallery()
        for vector in (None, [0, 0], [float('nan'), 0], [[1, 0]]):
            self.assertIsNone(gallery.observe(vector, 0)['identity'])

    def test_uncertain_match_is_not_enrolled(self):
        gallery = IdentityGallery()
        self.confirm(gallery, [1, 0])
        for stamp in (2000, 2500, 3000):
            self.assertIsNone(gallery.observe([0.4, 0.9165], stamp)['identity'])
        self.assertEqual(len(gallery), 1)

    def test_ambiguous_candidates_are_deferred(self):
        gallery = IdentityGallery()
        self.confirm(gallery, [1, 0])
        self.confirm(gallery, [0, 1], 2000)
        self.assertIsNone(gallery.observe([1, 1], 4000)['identity'])

    def test_expiration_and_clear(self):
        gallery = IdentityGallery()
        first = self.confirm(gallery, [1, 0])
        later = self.confirm(gallery, [1, 0], 122000)
        self.assertNotEqual(first['identity'], later['identity'])
        gallery.clear()
        self.assertEqual(len(gallery), 0)

    def test_confirmation_must_be_consecutive_and_capacity_is_bounded(self):
        gallery = IdentityGallery(max_identities=1)
        gallery.observe([1, 0], 0)
        gallery.observe(None, 500)
        self.assertIsNone(gallery.observe([1, 0], 1000)['identity'])
        self.confirm(gallery, [1, 0], 2000)
        result = self.confirm(gallery, [0, 1], 4000)
        self.assertIsNone(result['identity'])
        self.assertEqual(len(gallery), 1)


class AdapterTests(unittest.TestCase):
    def make_adapter(self, face):
        from unittest.mock import Mock
        from video_recognition.face_reidentification import FaceReidentifier
        adapter = FaceReidentifier.__new__(FaceReidentifier)
        adapter.app = Mock()
        adapter.app.get.return_value = [face]
        adapter.gallery = IdentityGallery()
        adapter.last_inference_ms = None
        return adapter

    def test_overlay_does_not_modify_input_or_expose_embeddings(self):
        from types import SimpleNamespace
        face = SimpleNamespace(bbox=np.array([10, 10, 110, 110]), det_score=0.95,
                               embedding=np.array([1., 0.]))
        adapter = self.make_adapter(face)
        frame = np.zeros((160, 320, 3), dtype=np.uint8)
        for stamp in (0, 500, 1000):
            overlay = adapter.annotate(frame, stamp, 1)
        self.assertEqual(len(adapter.gallery), 1)
        self.assertTrue(np.any(overlay))
        self.assertFalse(np.any(frame))
        adapter.close()
        self.assertEqual(len(adapter.gallery), 0)
        self.assertIsNone(adapter.app)

    def test_multiple_faces_and_poor_quality_do_not_enroll(self):
        from types import SimpleNamespace
        face = SimpleNamespace(bbox=np.array([10, 10, 30, 30]), det_score=0.95,
                               embedding=np.array([1., 0.]))
        adapter = self.make_adapter(face)
        frame = np.zeros((160, 320, 3), dtype=np.uint8)
        for stamp in (0, 500, 1000):
            adapter.annotate(frame, stamp, 2)
        adapter.app.get.assert_not_called()
        for stamp in (1500, 2000, 2500):
            adapter.annotate(frame, stamp, 1)
        self.assertEqual(len(adapter.gallery), 0)

    def test_inference_is_throttled(self):
        from types import SimpleNamespace
        face = SimpleNamespace(bbox=np.array([10, 10, 110, 110]), det_score=0.95,
                               embedding=np.array([1., 0.]))
        adapter = self.make_adapter(face)
        frame = np.zeros((160, 320, 3), dtype=np.uint8)
        for stamp in (0, 100, 200, 500):
            adapter.annotate(frame, stamp, 1)
        self.assertEqual(adapter.app.get.call_count, 2)

    def test_missing_or_multiple_faces_between_samples_interrupt_confirmation(self):
        from types import SimpleNamespace
        face = SimpleNamespace(bbox=np.array([10, 10, 110, 110]), det_score=0.95,
                               embedding=np.array([1., 0.]))
        frame = np.zeros((160, 320, 3), dtype=np.uint8)
        for count in (0, 2):
            adapter = self.make_adapter(face)
            for stamp, faces in ((0, 1), (250, count), (500, 1), (750, count), (1000, 1)):
                adapter.annotate(frame, stamp, faces)
            self.assertEqual(len(adapter.gallery), 0)
            self.assertEqual(adapter.app.get.call_count, 3)

    def test_demo_keeps_reidentification_out_of_feature_window_and_log(self):
        import json
        import sys
        from io import StringIO
        from unittest.mock import MagicMock, patch
        from video_recognition import webcam_test_mac as demo
        from video_recognition import face_reidentification
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        result = {'start_ms': 0, 'features': {'faces': []}}
        original = json.dumps(result)
        reid = MagicMock()
        reid.annotate.return_value = frame.copy()
        camera = MagicMock()
        pipeline = MagicMock()
        pipeline.process_one.return_value = result
        preview = MagicMock()
        preview.poll_key.return_value = {'calibrate': False, 'quit': False}
        preview.render.return_value = {'calibrate': False, 'quit': True}
        capture = MagicMock(last_frame=frame)
        log = MagicMock()
        with patch.object(sys, 'argv', ['demo', '--reidentify']), \
             patch.object(demo.cv2, 'VideoCapture', return_value=camera), \
             patch.object(demo.cv2, 'destroyAllWindows'), \
             patch.object(face_reidentification, 'FaceReidentifier', return_value=reid), \
             patch.object(demo, 'FaceAnalyzer'), patch.object(demo, 'PoseAnalyzer'), \
             patch.object(demo, 'PreviewCapture', return_value=capture), \
             patch.object(demo, 'VisionPipeline', return_value=pipeline), \
             patch.object(demo, 'DeveloperPreviewOutput', return_value=preview), \
             patch.object(demo, 'open_demo_log', return_value=(log, demo.Path('/tmp/mock-log'))), \
             patch.object(demo, 'write_frame_log') as writer, \
             patch.object(demo, 'log_debug_transition'), patch('sys.stdout', new=StringIO()):
            demo.main()
        self.assertEqual(json.dumps(result), original)
        self.assertIs(writer.call_args.args[1], result)
        reid.annotate.assert_called_once_with(frame, 0, 0)
        reid.close.assert_called_once()
        pipeline.close.assert_called_once()


    def test_windows_demo_reidentification_is_opt_in_and_display_only(self):
        import json
        import sys
        from contextlib import ExitStack
        from io import StringIO
        from unittest.mock import MagicMock, patch
        from video_recognition import webcam_test as demo
        from video_recognition import face_reidentification
        for enabled in (True, False):
            with self.subTest(enabled=enabled), ExitStack() as stack:
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                result = {'start_ms': 0, 'features': {'faces': []}}
                original = json.dumps(result)
                camera = MagicMock()
                camera.read.return_value = (True, frame)
                pipeline = MagicMock()
                pipeline.process_one.return_value = result
                preview = MagicMock()
                preview.poll_key.return_value = {'calibrate': False, 'quit': False}
                preview.render.return_value = {'calibrate': False, 'quit': True}
                capture = MagicMock(last_frame=frame)
                reid = MagicMock()
                annotated = frame.copy()
                reid.annotate.return_value = annotated
                stack.enter_context(patch.object(sys, 'argv', ['demo'] + (['--reidentify'] if enabled else [])))
                stack.enter_context(patch('sys.stdout', new=StringIO()))
                factory = stack.enter_context(patch.object(face_reidentification, 'FaceReidentifier', return_value=reid))
                stack.enter_context(patch.object(demo.cv2, 'VideoCapture', return_value=camera))
                for name in ('namedWindow', 'resizeWindow', 'moveWindow', 'imshow', 'waitKey', 'destroyAllWindows'):
                    stack.enter_context(patch.object(demo.cv2, name))
                windows = stack.enter_context(patch.object(demo, 'list_windows', return_value=[(42, demo.CAMERA_WINDOW)]))
                win_capture = stack.enter_context(patch.object(demo, 'WindowCapture'))
                stack.enter_context(patch.object(demo, 'FaceAnalyzer'))
                stack.enter_context(patch.object(demo, 'PoseAnalyzer'))
                stack.enter_context(patch.object(demo, 'PreviewCapture', return_value=capture))
                stack.enter_context(patch.object(demo, 'VisionPipeline', return_value=pipeline))
                stack.enter_context(patch.object(demo, 'DeveloperPreviewOutput', return_value=preview))
                stack.enter_context(patch.object(demo, 'open_demo_log', return_value=(MagicMock(), demo.Path('/tmp/mock-log'))))
                writer = stack.enter_context(patch.object(demo, 'write_frame_log'))
                stack.enter_context(patch.object(demo, 'log_debug_transition'))
                demo.main()
                windows.assert_called_once()
                win_capture.assert_called_once_with(42)
                self.assertEqual(json.dumps(result), original)
                self.assertIs(writer.call_args.args[1], result)
                if enabled:
                    factory.assert_called_once()
                    reid.annotate.assert_called_once_with(frame, 0, 0)
                    self.assertIs(preview.render.call_args.args[0], annotated)
                    reid.close.assert_called_once()
                else:
                    factory.assert_not_called()
                    self.assertIs(preview.render.call_args.args[0], frame)
                pipeline.close.assert_called_once()
                camera.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
