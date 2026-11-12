import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from video_recognition.face_analysis import FaceAnalyzer
from frontend.live_session import LiveSession

class FrontendFaceTransformTests(unittest.TestCase):
    def analyzer(self, matrices):
        p = SimpleNamespace(x=0.5, y=0.5, z=0.0)
        category = lambda name: SimpleNamespace(category_name=name, score=0.1)
        result = SimpleNamespace(face_landmarks=[[p]*478], facial_transformation_matrixes=matrices,
            face_blendshapes=[[category(k) for k in ('mouthSmileLeft','mouthSmileRight','browInnerUp','jawOpen')]])
        analyzer = FaceAnalyzer.__new__(FaceAnalyzer)
        analyzer.mp = SimpleNamespace(Image=lambda **kwargs: None, ImageFormat=SimpleNamespace(SRGB=0))
        analyzer.detector = SimpleNamespace(detect_for_video=lambda *args: result)
        analyzer.pose_source = 'face-transform'
        analyzer.last_timestamp = -1
        analyzer.show_landmarks = False
        return analyzer

    def test_transform_is_primary_and_invalid_matrix_never_falls_back(self):
        frame = SimpleNamespace(image=np.zeros((20,20,3),dtype=np.uint8), capture_ms=1, width=20, height=20)
        valid = self.analyzer([np.eye(4)]).analyze(frame)[0]
        invalid = self.analyzer([]).analyze(frame)[0]
        self.assertIsNotNone(valid['head_angles_degrees'])
        self.assertIsNone(invalid['head_angles_degrees'])
        self.assertIn('pose_unreliable', invalid['quality']['reason_codes'])

    def test_frontend_runtimes_and_session_metadata_label_estimator(self):
        for multi in (False, True):
            live = LiveSession(1, reidentification_enabled=multi)
            runtime = live._make_model()
            self.assertEqual(runtime.model_version, live._model_identity()['model_version'])
            self.assertIn('face-transform', runtime.latest['model_version'])

if __name__ == '__main__': unittest.main()
