"""얼굴 변환행렬의 좌표계·결측·동일 검출 재사용을 검증한다."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from engagement_model.face_transform import transform_angles, FaceTransformAnalyzer, FaceTransformComparison
from engagement_model.rotation_calibration import euler_rotation
from engagement_model.config import ModelConfig
from engagement_model.tests.test_runtime import feature_window
from engagement_model.tests.test_yaw_calibration import face
from engagement_model.webcam_demo import WebcamHeadCalibration, GuidedYawDiagnostic, pose_display_view


def matrix(pitch=0, yaw=0):
    result=np.eye(4)
    result[:3,:3]=euler_rotation(dict(pitch=pitch,yaw=yaw,roll=0))
    return result


class FaceTransformTests(unittest.TestCase):
    def test_transform_primary_selects_its_feature_runtime_and_baseline_without_fallback(self):
        pnp=SimpleNamespace(latest={'current_direction_score':100})
        transformed=SimpleNamespace(latest={'current_direction_score':None})
        comparison=SimpleNamespace(runtime=transformed,input_feature={'transform':True},calibration=object())
        models={'gaussian':pnp,'gaussian_face_transform':transformed}
        feature,view,calibration=pose_display_view({},models,object(),comparison,'face-transform')
        self.assertIs(feature,comparison.input_feature)
        self.assertIs(calibration,comparison.calibration)
        self.assertIs(view['gaussian'],transformed)
        self.assertIs(view['gaussian_pnp'],pnp)
        self.assertIsNone(view['gaussian'].latest['current_direction_score'])
        self.assertIs(models['gaussian'],pnp)

    def test_pnp_display_rollback_preserves_original_objects(self):
        feature,models,calibration={}, {}, object()
        actual=pose_display_view(feature,models,calibration,None,'pnp')
        self.assertIs(actual[0],feature)
        self.assertIs(actual[1],models)
        self.assertIs(actual[2],calibration)

    def test_stability_guidance_counts_all_nine_stages(self):
        guide=GuidedYawDiagnostic(1,1,include_down=True,front_stability=True)
        self.assertEqual(len(guide.STAGES),9)
        self.assertEqual(guide.STAGES[1:3],('front_small_motion','front_return'))
        guide.state='running'
        guide.started_ms=0
        self.assertIn('9/9',guide.screen_lines(17000)[0])

    def test_span_handles_euler_wrap_and_tracks_fixed_baseline(self):
        comparison=FaceTransformComparison(WebcamHeadCalibration(),ModelConfig.gaussian())
        rows=[]
        for value in (179.,-179.):
            channels={name:dict(raw=dict(pitch=value,yaw=0),smoothed=dict(pitch=value,yaw=0),
                relative=dict(pitch=0,yaw=0),baseline=dict(pitch=180,yaw=0,roll=0),state='front',current=100)
                for name in ('pnp','face_transform')}
            rows.append(dict(stage='front_initial',phase='hold',channels=channels))
        comparison.rows.extend(rows)
        for sample in comparison.summary()['front_initial'].values():
            self.assertAlmostEqual(sample['angles']['raw']['pitch']['span'],2)
            self.assertEqual(sample['baseline_changes'],0)

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

    def test_single_detection_is_reused_and_pnp_output_preserved(self):
        detector=Mock()
        detector.detect_for_video.return_value=SimpleNamespace(facial_transformation_matrixes=[matrix()])
        class Delegate:
            def __init__(self):
                self.detector=Mock()
                self.last_face_centers=[(.5,.5)]
            def analyze(self, frame):
                self.detector.detect_for_video('same-image',100)
                return [face(dict(pitch=170,yaw=20,roll=0))]
        delegate=Delegate()
        with patch('mediapipe.tasks.python.vision.FaceLandmarker.create_from_options',return_value=detector) as factory:
            analyzer=FaceTransformAnalyzer(delegate)
        result=analyzer.analyze(None)
        self.assertEqual(result[0]['head_angles_degrees']['yaw'],20)
        self.assertAlmostEqual(analyzer.transform_faces[0]['head_angles_degrees']['yaw'],0)
        detector.detect_for_video.assert_called_once_with('same-image',100)
        self.assertTrue(factory.call_args.args[0].output_facial_transformation_matrixes)
        detector.detect_for_video.return_value.facial_transformation_matrixes=[]
        analyzer.analyze(None)
        self.assertIsNone(analyzer.transform_faces[0]['head_angles_degrees'])
        self.assertEqual(analyzer.transform_reason,'face_transform_unavailable')

    def test_comparison_down_null_and_front_recovery_without_input_mutation(self):
        comparison=FaceTransformComparison(WebcamHeadCalibration(duration_ms=100,min_samples=2),ModelConfig.gaussian())
        comparison.start()
        analyzer=SimpleNamespace(last_face_centers=[(.5,.5)])
        for i in range(50):
            pitch=20 if 10<=i<30 else 0
            analyzer.transform_faces=[face(transform_angles(matrix(pitch=pitch)))]
            frame=feature_window(i*100)
            original=deepcopy(frame)
            comparison.update(frame,analyzer,{'source_angles':None},stage='test')
            self.assertEqual(frame,original)
            if i==29:
                self.assertEqual(comparison.runtime.latest['orientation_state'],'looking_down')
                self.assertIsNone(comparison.runtime.latest['current_direction_score'])
        self.assertEqual(comparison.runtime.latest['current_direction_score'],100)
        self.assertEqual(comparison.calibration.status,'ready')
        recovery=comparison.recovery_summary()['face_transform']
        self.assertGreater(recovery['looking_down_frames'],0)
        self.assertEqual(recovery['nonnull_during_down'],0)
        self.assertFalse(recovery['awaiting_front_return'])
        self.assertIsNotNone(recovery['first_front_recoveries'][0]['current'])


if __name__=='__main__':
    unittest.main()
