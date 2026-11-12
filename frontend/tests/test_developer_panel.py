import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from frontend.live_session import LiveSession


class DeveloperPanelTests(unittest.TestCase):
    def test_shared_panel_handles_disabled_single_and_group_models(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        feature = {'start_ms': 1000, 'features': {'faces': [], 'upper_bodies': [],
                   'head_calibration_status': 'waiting'},
                   'quality': {'status': 'valid', 'reason_codes': []}}
        face = SimpleNamespace(last_landmarks=[], last_face_boxes=[])
        pose = SimpleNamespace(last_landmarks=[])
        for mode in ('disabled', 'single', 'multi'):
            with self.subTest(mode=mode), patch('frontend.live_session.cv2.namedWindow'), \
                 patch('frontend.live_session.cv2.resizeWindow'), \
                 patch('frontend.live_session.cv2.imshow') as display, \
                 patch('frontend.live_session.cv2.waitKey', return_value=ord('c')):
                live = LiveSession(1, model_enabled=mode != 'disabled',
                                   reidentification_enabled=mode == 'multi')
                if mode != 'disabled':
                    live.model_runtime = live._make_model()
                self.assertTrue(live._render_developer_preview(frame, feature, face, pose, True))
                self.assertEqual(display.call_count, 1)
                self.assertEqual(display.call_args.args[1].shape, (860, 1400, 3))


if __name__ == '__main__':
    unittest.main()
