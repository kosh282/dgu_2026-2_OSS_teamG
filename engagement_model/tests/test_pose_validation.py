"""실제 오류와 같은 음의 깊이 PnP 해를 합성 투영으로 재현한다."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from engagement_model.pose_validation import ValidatedPose
from engagement_model.rotation_calibration import euler_rotation
from engagement_model.tests.test_runtime import feature_window
from engagement_model.tests.test_yaw_calibration import calibrated, observe
from engagement_model.runtime import EngagementRuntime
from engagement_model.gaussian import GaussianRuntime
from engagement_model.config import ModelConfig
from engagement_model.webcam_demo import UncertaintyDiagnostics

MODEL = np.array([[0,0,0],[0,-330,-65],[-225,170,-135],[225,170,-135],[-150,-150,-125],[150,-150,-125]],float)
CAMERA = np.array([[640.,0,320],[0,640.,240],[0,0,1]])


def landmarks(pitch=170, yaw=0, noise=0, rng=None):
    rotation = euler_rotation(dict(pitch=(pitch+180)%360-180,yaw=yaw,roll=0))
    image = cv2.projectPoints(MODEL,cv2.Rodrigues(rotation)[0],np.array([0.,0.,1600.]),CAMERA,np.zeros(4))[0].reshape(-1,2)
    if rng is not None:
        image += rng.normal(0,noise,image.shape)
    points = [(0.,0.,0.)]*292
    for i, p in zip((1,152,33,263,61,291),image):
        points[i] = (p[0]/640,p[1]/480,0.)
    return points, image, rotation


class PoseValidationTests(unittest.TestCase):
    def test_diagnostic_candidate_respects_custom_yaw_boundaries(self):
        cases=((15,20,False),(15,19,False),(1,16,False),(20,30,False),
               (25,30,False),(15,20.000000000000004,True),(15,30,True))
        for front,away,enabled in cases:
            with self.subTest(front=front,away=away):
                config=ModelConfig.gaussian(front_yaw_degrees=front,away_yaw_degrees=away)
                model,control=GaussianRuntime(config),GaussianRuntime(config)
                diagnostic=UncertaintyDiagnostics(config)
                self.assertEqual(diagnostic.candidate is not None,enabled)
                for timestamp in (0,100,200):
                    frame=feature_window(timestamp,pitch=0,yaw=10)
                    result=model(frame)
                    diagnostic.update(frame,result)
                    self.assertEqual(result,control(frame))
                summary=diagnostic.summary()
                self.assertEqual(summary['baseline']['frames'],3)
                self.assertEqual(summary['candidate_front_yaw_20']['enabled'],enabled)
                self.assertEqual(summary['candidate_front_yaw_20']['frames'],3 if enabled else 0)
                self.assertEqual(model.config,config)
                diagnostic.reset()
                self.assertEqual(diagnostic.summary()['baseline']['frames'],0)

    def test_angle_gap_candidate_changes_label_without_changing_gaussian_score(self):
        model=GaussianRuntime()
        diagnostic=UncertaintyDiagnostics(model.config)
        for timestamp in range(0,1000,100):
            frame=feature_window(timestamp,pitch=0,yaw=25)
            diagnostic.update(frame,model(frame))
            self.assertEqual(model.latest['orientation_state'],'ambiguous')
            self.assertEqual(diagnostic.candidate.latest['orientation_state'],'front')
            self.assertEqual(model.latest['current_direction_score'],diagnostic.candidate.latest['current_direction_score'])
        summary=diagnostic.summary()
        self.assertEqual(summary['baseline']['counts']['state:ambiguous'],10)
        self.assertEqual(summary['baseline']['counts']['current_null'],0)

    def test_uncertainty_causes_keep_down_and_failed_pose_deferred(self):
        model=GaussianRuntime()
        diagnostic=UncertaintyDiagnostics(model.config)
        frames=[feature_window(0,pitch=20),feature_window(100,faces=0),feature_window(200)]
        frames[-1]['features']['faces'][0]['quality']['reason_codes']=['pose_unreliable']
        for frame in frames:
            diagnostic.update(frame,model(frame))
        for summary in diagnostic.summary().values():
            self.assertEqual(summary['counts']['current_null'],3)
            self.assertEqual(summary['counts']['null_cause:downward_uncertain'],1)
            self.assertEqual(summary['counts']['null_cause:face_not_found'],1)
            self.assertEqual(summary['counts']['null_cause:pose_unreliable'],1)

    def test_positive_candidates_select_lowest_reprojection_error(self):
        points,_,_=landmarks(yaw=25,noise=2,rng=np.random.default_rng(42))
        angles,info=ValidatedPose().estimate(points,640,480,0)
        self.assertIsNotNone(angles)
        self.assertAlmostEqual(info['reprojection_rmse_pixels'],min(info['candidate_rmse_pixels'].values()))
        self.assertGreater(info['min_depth'],0)

    def test_guided_down_is_optional_and_has_a_front_return(self):
        from engagement_model.webcam_demo import GuidedYawDiagnostic
        extended=GuidedYawDiagnostic(1,1,include_down=True)
        original=GuidedYawDiagnostic(1,1)
        self.assertEqual(len(original.STAGES),5)
        self.assertEqual(extended.STAGES[-2:],('down','front_after_down'))
        extended.started_ms=0
        self.assertEqual(extended.STAGES[extended.phase(11000)[0]],'down')
        self.assertEqual(extended.STAGES[extended.phase(13000)[0]],'front_after_down')

    def test_negative_depth_regression_old_fails_new_stays_in_front(self):
        rng=np.random.default_rng(2026)
        old_bad=new_bad=0
        for pitch in (160,170,179):
            for yaw in range(-80,81,10):
                for noise in (0,1,2,4):
                    points,image,truth=landmarks(pitch,yaw,noise,rng)
                    _,r,t=cv2.solvePnP(MODEL,image,CAMERA,np.zeros(4))
                    R=cv2.Rodrigues(r)[0]
                    old_bad += ((R@MODEL.T).T+t.reshape(3))[:,2].min()<=0
                    angles,info=ValidatedPose().estimate(points,640,480,0)
                    self.assertIsNotNone(angles, info)
                    new_bad += info['min_depth']<=0
                    error=np.degrees(np.arccos(np.clip((np.trace(np.array(info['rotation'])@truth.T)-1)/2,-1,1)))
                    self.assertLess(error,30)
        self.assertEqual(old_bad,14)
        self.assertEqual(new_bad,0)

    def test_euler_calibration_down_side_symmetry_and_return(self):
        for base in (-20,0,20):
            est=ValidatedPose()
            origin=est.estimate(landmarks(yaw=base)[0],640,480,0)[0]
            c=calibrated(origin)
            scores=[]
            for time,yaw in ((200,base-30),(300,base+30)):
                # 독립 관측으로 좌우를 비교한다. 연속성 오차는 별도 시험한다.
                angles,_=ValidatedPose().estimate(landmarks(yaw=yaw)[0],640,480,time)
                observed,result=observe(c,angles,timestamp=time)
                self.assertAlmostEqual(observed['head_angles_relative_degrees']['yaw'],yaw-base,places=4)
                self.assertNotEqual(result['orientation_state'],'looking_down')
                scores.append(result['current_direction_score'])
            self.assertEqual(scores[0],scores[1])
            down,_=ValidatedPose().estimate(landmarks(pitch=190,yaw=base)[0],640,480,400)
            _,result=observe(c,down,timestamp=400)
            self.assertEqual(result['orientation_state'],'looking_down')
            self.assertIsNone(result['current_direction_score'])
            _,result=observe(c,origin,timestamp=500)
            self.assertEqual(result['current_direction_score'],100)

    def test_invalid_landmarks_and_large_rotation_are_not_fabricated(self):
        est=ValidatedPose()
        points,_,_=landmarks()
        self.assertIsNotNone(est.estimate(points,640,480,0)[0])
        turned,_,_=landmarks(yaw=80)
        angles,info=est.estimate(turned,640,480,10)
        self.assertIsNone(angles)
        self.assertEqual(info['reason'],'pose_rotation_discontinuity')
        points[1]=(float('nan'),0,0)
        self.assertIsNone(est.estimate(points,640,480,100)[0])

    def test_nonpositive_depth_is_rejected_even_if_solver_reports_success(self):
        with patch('engagement_model.pose_validation.cv2.solvePnP',
                   return_value=(True,np.zeros((3,1)),np.array([[0.],[0.],[-1600.]]))):
            angles,info=ValidatedPose().estimate(landmarks()[0],640,480,0)
        self.assertIsNone(angles)
        self.assertEqual(info['status'],'pose_unreliable')
        self.assertEqual(info['reason'],'pose_nonpositive_depth')

    def test_short_pose_failure_preserves_away_timer_without_scoring_failure(self):
        model=EngagementRuntime()
        for time in range(0,1100,100):
            model(feature_window(time,pitch=0,yaw=60))
        before=model.latest['away_seconds']
        missing=feature_window(1100)
        face=missing['features']['faces'][0]
        face['quality']['reason_codes']=['pose_unreliable']
        face['head_angles_relative_degrees']=None
        result=model(missing)
        self.assertIsNone(result['score'])
        self.assertIn('pose_unreliable',result['reason_codes'])
        result=model(feature_window(1200,pitch=0,yaw=60))
        self.assertGreaterEqual(result['away_seconds'],before)


if __name__=='__main__':
    unittest.main()
