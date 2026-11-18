"""얼굴 변환행렬의 좌표계·결측·동일 검출 재사용을 검증한다."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from engagement_model.face_transform import transform_angles
from engagement_model.rotation_calibration import euler_rotation
from engagement_model.config import ModelConfig
from engagement_model.tests.test_runtime import feature_window
from engagement_model.tests.test_yaw_calibration import face
from engagement_model.webcam_demo import GuidedYawDiagnostic


def matrix(pitch=0, yaw=0):
    result=np.eye(4)
    result[:3,:3]=euler_rotation(dict(pitch=pitch,yaw=yaw,roll=0))
    return result


class FaceTransformTests(unittest.TestCase):
    def test_stability_guidance_counts_all_nine_stages(self):
        guide=GuidedYawDiagnostic(1,1,include_down=True,front_stability=True)
        self.assertEqual(len(guide.STAGES),9)
        self.assertEqual(guide.STAGES[1:3],('front_small_motion','front_return'))
        guide.state='running'
        guide.started_ms=0
        self.assertIn('9/9',guide.screen_lines(17000)[0])

    def test_camera_axes_pitch_down_positive_yaw_symmetric(self):
        origin=transform_angles(matrix())
        for delta in (-30,30):
            value=transform_angles(matrix(yaw=delta))
            self.assertAlmostEqual(value['yaw'],-delta)
        down=transform_angles(matrix(pitch=20))
        self.assertAlmostEqual((down['pitch']-origin['pitch']+180)%360-180,20)
        scaled=matrix(pitch=20)
        scaled[:3,:3]*=2
        self.assertEqual(transform_angles(scaled),down)

    def test_invalid_and_reflected_transforms_do_not_become_zero_pose(self):
        for bad in (np.zeros((4,4)),np.eye(3),np.full((4,4),np.nan),np.diag([-1,1,1,1]),np.diag([1,1,.01,1])):
            with self.assertRaises(ValueError):
                transform_angles(bad)

if __name__=='__main__':
    unittest.main()
