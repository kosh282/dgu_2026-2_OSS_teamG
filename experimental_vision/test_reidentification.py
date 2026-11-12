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


class MultiFaceTests(unittest.TestCase):
    def make_adapter(self):
        from unittest.mock import Mock
        from types import SimpleNamespace
        from video_recognition.face_reidentification import FaceReidentifier, MultiIdentityGallery
        adapter = FaceReidentifier.__new__(FaceReidentifier)
        adapter.app = Mock()
        def detect(image, **kwargs):
            # 전체 영상에서는 검출되지만 얼굴만 자르면 검출되지 않는 회귀 사례.
            if image.shape[:2] != (200, 600):
                return np.empty((0, 5)), np.empty((0, 5, 2))
            return (np.array([[30, 40, 210, 160, .76], [390, 40, 570, 160, .76]]),
                    np.zeros((2, 5, 2)))
        def recognize(image, face):
            x = int((face.bbox[0] + face.bbox[2]) / 2)
            index = int(image[100, x, 0])
            face.embedding = np.eye(3)[index]
            return face.embedding
        adapter.app.get.return_value = []
        adapter.app.det_model.detect.side_effect = detect
        adapter.app.models = {'recognition': Mock()}
        adapter.app.models['recognition'].get.side_effect = recognize
        adapter.gallery = MultiIdentityGallery()
        adapter.connections = {}
        adapter.last_inference_ms = None
        adapter.last_observation_ms = None
        return adapter

    def scene(self):
        frame = np.zeros((200, 600, 3), dtype=np.uint8)
        frame[:, 300:] = 1
        faces = [{'transient_track_key': 10, 'head_position': {'x': .2, 'y': .5}},
                 {'transient_track_key': 20, 'head_position': {'x': .8, 'y': .5}}]
        boxes = [[.05, .2, .35, .8], [.65, .2, .95, .8]]
        return frame, faces, boxes

    def test_simultaneous_ids_pause_and_return_independently(self):
        adapter = self.make_adapter()
        frame, faces, boxes = self.scene()
        original = frame.copy()
        for stamp in (0, 500, 1000):
            adapter.annotate(frame, stamp, faces, boxes=boxes)
        self.assertEqual(len(adapter.connections), 2)
        ids = dict(adapter.connections)
        self.assertEqual(len(set(ids.values())), 2)
        self.assertEqual(adapter.app.models['recognition'].get.call_count, 6)
        adapter.annotate(frame, 1500, faces[::-1], boxes=boxes[::-1])
        self.assertEqual(adapter.app.models['recognition'].get.call_count, 6)
        adapter.annotate(frame, 2000, faces[:1], boxes=boxes[:1])
        self.assertEqual(adapter.connections, {10: ids[10]})
        faces[1]['transient_track_key'] = 30
        for stamp in (2500, 3000, 3500):
            adapter.annotate(frame, stamp, faces, boxes=boxes)
        self.assertEqual(adapter.connections, {10: ids[10], 30: ids[20]})
        self.assertEqual(adapter.app.models['recognition'].get.call_count, 9)
        np.testing.assert_array_equal(frame, original)
        self.assertNotIn('identity', faces[0])
        adapter.close()
        self.assertFalse(adapter.connections)
        self.assertFalse(adapter.gallery.entries)

    def test_identify_without_render_returns_independent_mapping(self):
        adapter = self.make_adapter()
        frame, faces, boxes = self.scene()
        original = frame.copy()
        for stamp in (0, 500, 1000):
            mapping = adapter.identify(frame, stamp, faces, boxes=boxes)
        self.assertEqual(mapping, {10: 'S001', 20: 'S002'})
        mapping.clear()
        self.assertEqual(adapter.connections, {10: 'S001', 20: 'S002'})
        np.testing.assert_array_equal(frame, original)

    def test_render_uses_cached_results_without_inference_or_state_changes(self):
        import copy
        adapter = self.make_adapter()
        frame, faces, boxes = self.scene()
        for stamp in (0, 500, 1000):
            adapter.identify(frame, stamp, faces, boxes=boxes)
        original = frame.copy()
        before = copy.deepcopy((adapter.connections, adapter.gallery.__dict__,
                                adapter.last_inference_ms, adapter.last_observation_ms,
                                adapter.last_results))
        calls = adapter.app.det_model.detect.call_count
        recognition_calls = adapter.app.models['recognition'].get.call_count
        for _ in range(2):
            rendered = adapter.render(frame, faces, boxes=boxes)
            self.assertFalse(np.array_equal(rendered, original))
        np.testing.assert_array_equal(frame, original)
        self.assertEqual(adapter.app.det_model.detect.call_count, calls)
        self.assertEqual(adapter.app.models['recognition'].get.call_count, recognition_calls)
        after = (adapter.connections, adapter.gallery.__dict__, adapter.last_inference_ms,
                 adapter.last_observation_ms, adapter.last_results)
        self.assertEqual(repr(before), repr(after))
        adapter.close()
        self.assertEqual(adapter.last_results, {})

    def test_annotate_matches_identify_then_render(self):
        wrapped = self.make_adapter()
        split = self.make_adapter()
        frame, faces, boxes = self.scene()
        for stamp in (0, 500, 1000, 1500):
            actual = wrapped.annotate(frame, stamp, faces, boxes=boxes)
            split.identify(frame, stamp, faces, boxes=boxes)
            expected = split.render(frame, faces, boxes=boxes)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(wrapped.connections, split.connections)

    def test_render_before_identification_is_read_only(self):
        adapter = self.make_adapter()
        frame, faces, boxes = self.scene()
        rendered = adapter.render(frame, faces, boxes=boxes)
        self.assertFalse(np.array_equal(rendered, frame))
        adapter.app.det_model.detect.assert_not_called()
        self.assertIsNone(adapter.last_observation_ms)
        self.assertFalse(adapter.connections)

    def test_duplicate_embedding_does_not_assign_same_id_twice(self):
        adapter = self.make_adapter()
        frame, faces, boxes = self.scene()
        frame[:] = 0
        for stamp in (0, 500, 1000):
            adapter.annotate(frame, stamp, faces, boxes=boxes)
        self.assertEqual(adapter.connections, {})

    def test_overlap_and_missing_geometry_defer(self):
        adapter = self.make_adapter()
        frame, faces, boxes = self.scene()
        for stamp in (0, 500, 1000):
            adapter.annotate(frame, stamp, faces, boxes=[boxes[0], boxes[0]])
        adapter.app.det_model.detect.assert_not_called()
        adapter.annotate(frame, 1500, faces, boxes=[])
        self.assertFalse(adapter.connections)

    def test_small_or_off_center_detection_does_not_enroll(self):
        from types import SimpleNamespace
        frame, faces, boxes = self.scene()
        for box in ([0, 0, 10, 10], [0, 0, 85, 85]):
            adapter = self.make_adapter()
            adapter.app.det_model.detect.side_effect = None
            adapter.app.det_model.detect.return_value = (
                np.array([[*box, .95]]), np.zeros((1, 5, 2)))
            for stamp in (0, 500, 1000):
                adapter.annotate(frame, stamp, faces[:1], boxes=boxes[:1])
            self.assertFalse(adapter.connections)

    def test_long_gap_reconfirms_and_continuous_track_does_not_expire(self):
        frame, faces, boxes = self.scene()
        adapter = self.make_adapter()
        for stamp in range(0, 125001, 500):
            adapter.annotate(frame, stamp, faces[:1], boxes=boxes[:1])
        identity = adapter.connections[10]
        self.assertEqual(adapter.app.det_model.detect.call_count, 3)
        adapter.annotate(frame, 127000, faces[:1], boxes=boxes[:1])
        self.assertFalse(adapter.connections)
        for stamp in (127500, 128000):
            adapter.annotate(frame, stamp, faces[:1], boxes=boxes[:1])
        self.assertEqual(adapter.connections[10], identity)

    def test_gallery_expiry_and_reserved_identity(self):
        from video_recognition.face_reidentification import MultiIdentityGallery
        gallery = MultiIdentityGallery()
        for stamp in (0, 500, 1000):
            matched = gallery.observe_many({10: [1, 0]}, stamp, set())
        identity = matched[10]['identity']
        for stamp in (2000, 2500, 3000):
            result = gallery.observe_many({20: [1, 0]}, stamp, {identity})
            self.assertIsNone(result[20]['identity'])
        for stamp in (124000, 124500, 125000):
            result = gallery.observe_many({20: [1, 0]}, stamp, set())
        self.assertNotEqual(identity, result[20]['identity'])


class AdapterTests(unittest.TestCase):
    def test_demo_keeps_reidentification_out_of_feature_window_and_log(self):
        import json
        import sys
        from io import StringIO
        from unittest.mock import MagicMock, patch
        from video_recognition import webcam_test_mac as demo
        from video_recognition import face_reidentification
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        result = {'start_ms': 0, 'features': {'faces': [
            {'transient_track_key': 10}, {'transient_track_key': 20}]}}
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
             patch.object(demo, 'FaceAnalyzer', return_value=MagicMock(last_face_boxes=[])), patch.object(demo, 'PoseAnalyzer'), \
             patch.object(demo, 'PreviewCapture', return_value=capture), \
             patch.object(demo, 'VisionPipeline', return_value=pipeline), \
             patch.object(demo, 'DeveloperPreviewOutput', return_value=preview), \
             patch.object(demo, 'open_demo_log', return_value=(log, demo.Path('/tmp/mock-log'))), \
             patch.object(demo, 'write_frame_log') as writer, \
             patch.object(demo, 'log_debug_transition'), patch('sys.stdout', new=StringIO()):
            demo.main()
        self.assertEqual(json.dumps(result), original)
        self.assertIs(writer.call_args.args[1], result)
        reid.annotate.assert_called_once_with(frame, 0, result['features']['faces'], boxes=[])
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
                result = {'start_ms': 0, 'features': {'faces': [
                    {'transient_track_key': 10}, {'transient_track_key': 20}]}}
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
                stack.enter_context(patch.object(demo, 'FaceAnalyzer', return_value=MagicMock(last_face_boxes=[])))
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
                    reid.annotate.assert_called_once_with(frame, 0, result['features']['faces'], boxes=[])
                    self.assertIs(preview.render.call_args.args[0], annotated)
                    reid.close.assert_called_once()
                else:
                    factory.assert_not_called()
                    self.assertIs(preview.render.call_args.args[0], frame)
                pipeline.close.assert_called_once()
                camera.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
