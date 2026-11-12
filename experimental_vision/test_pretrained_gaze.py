"""카메라나 이미지 저장 없이 사전 학습 시선 모델 연결을 검사한다."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from video_recognition.gaze_estimation import GazeEstimator, GazeDownPolicy, GazeFaceAnalyzer
from engagement_model.gaussian import GaussianRuntime
from engagement_model.tests.test_runtime import feature_window


def logits(index):
    values = np.full((1, 90), -100., dtype=np.float32)
    values[0, index] = 100.
    return values


def session():
    model = Mock()
    model.get_inputs.return_value = [SimpleNamespace(name='input', shape=[1, 3, 448, 448])]
    model.get_outputs.return_value = [SimpleNamespace(name='yaw'), SimpleNamespace(name='pitch')]
    model.run.return_value = [logits(45), logits(45)]
    return model


FACE = {'quality': {'status': 'valid'}, 'transient_track_key': 1, 'eye_openness': .3}
IMAGE = np.zeros((240, 320, 3), dtype=np.uint8)
BOX = [.1, .1, .9, .9]


class PretrainedGazeTests(unittest.TestCase):
    def test_panel_explains_eye_filter_and_each_down_gate_condition(self):
        estimator = GazeEstimator(session=session())
        closed = {**FACE, 'eye_openness': .08}
        estimator.update(IMAGE, [BOX], [closed], 1000)
        self.assertIn('EAR=0.080', '\n'.join(estimator.lines()))
        policy = GazeDownPolicy(estimator)
        estimator.latest = dict(state='forward_candidate', pitch=-12., yaw=0., sample_ms=1000)
        self.assertFalse(policy(feature_window(1000), {'angles': {'pitch': 20.}})[0])
        self.assertIn('head P 20.0 >= 15: yes', policy.diagnostic)
        self.assertIn('gaze P -12.0 <= -30: no', policy.diagnostic)
        estimator.latest = dict(state='unknown', pitch=-24., yaw=0., sample_ms=1100,
                                inference_ms=30., reason='angle_boundary')
        self.assertIn('angle_boundary', '\n'.join(estimator.lines()))

    def test_demo_opt_in_wires_policy_and_releases_resources_without_camera(self):
        from engagement_model.webcam_demo import run
        estimator = GazeEstimator(session=session())
        camera = Mock()
        camera.isOpened.return_value = True
        pipeline = Mock()
        pipeline.process_one.return_value = feature_window(0)
        with patch('video_recognition.gaze_estimation.GazeEstimator', return_value=estimator), \
             patch('engagement_model.webcam_demo.cv2.VideoCapture', return_value=camera), \
             patch('engagement_model.webcam_demo.FaceAnalyzer') as face_factory, \
             patch('engagement_model.webcam_demo.PoseAnalyzer'), \
             patch('engagement_model.webcam_demo.VisionPipeline', return_value=pipeline) as pipeline_factory, \
             patch('engagement_model.webcam_demo.cv2.namedWindow'), \
             patch('engagement_model.webcam_demo.cv2.resizeWindow'), \
             patch('engagement_model.webcam_demo.cv2.setMouseCallback'), \
             patch('engagement_model.webcam_demo.cv2.imshow'), \
             patch('engagement_model.webcam_demo.cv2.waitKey', return_value=ord('q')), \
             patch('engagement_model.webcam_demo.close_window'), \
             patch('engagement_model.webcam_demo.render_panel', return_value=IMAGE):
            run(compare=False, gaze=True)
        runtime = pipeline_factory.call_args.kwargs['model_consumer']
        self.assertIsInstance(runtime.downward_policy, GazeDownPolicy)
        self.assertIn('mobilegaze', runtime.model_version)
        self.assertIsInstance(pipeline_factory.call_args.args[3], GazeFaceAnalyzer)
        self.assertIsNone(estimator.session)
        camera.release.assert_called_once()
        face_factory.return_value.close.assert_called_once()

    def test_small_face_interrupts_cached_result_and_track_change_clears_cache(self):
        estimator = GazeEstimator(session=session())
        estimator.update(IMAGE, [[.1, .1, .4, .9]], [FACE], 0)
        result = estimator.update(IMAGE, [[.1, .1, .32, .9]], [FACE], 100)
        self.assertEqual(result['reason'], 'face_too_small')
        policy = GazeDownPolicy(estimator)
        for ms in (1000, 1400):
            estimator.latest = dict(state='away_candidate', yaw=0., pitch=-32., sample_ms=ms)
            policy(feature_window(ms, track_key=1), {'angles': {'pitch': 20}})
        self.assertIsNotNone(policy.since)
        self.assertFalse(policy(feature_window(1500, track_key=2), {'angles': {'pitch': 20}})[0])
        self.assertIsNone(estimator.latest['pitch'])
        self.assertIsNone(policy.since)

    def test_wrapper_reuses_current_analysis_frame_without_changing_features(self):
        features = [dict(FACE)]
        analyzer = Mock(last_face_boxes=[BOX])
        analyzer.analyze.return_value = features
        estimator = Mock()
        wrapper = GazeFaceAnalyzer(analyzer, estimator)
        frame = SimpleNamespace(image=IMAGE, capture_ms=321)
        self.assertIs(wrapper.analyze(frame), features)
        estimator.update.assert_called_once_with(IMAGE, [BOX], features, 321)
        self.assertEqual(features, [FACE])

    def test_preprocess_and_yaw_pitch_decode_follow_upstream(self):
        model = session()
        model.run.return_value = [logits(50), logits(35)]
        estimator = GazeEstimator(session=model)
        result = estimator.update(IMAGE, [BOX], [FACE], 100)
        self.assertAlmostEqual(result['yaw'], 20.)
        self.assertAlmostEqual(result['pitch'], -40.)
        tensor = model.run.call_args.args[1]['input']
        self.assertEqual(tensor.shape, (1, 3, 448, 448))
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), -.485/.229, places=5)

    def test_front_away_and_boundary_are_candidates(self):
        model = session()
        estimator = GazeEstimator(session=model)
        self.assertEqual(estimator.update(IMAGE, [BOX], [FACE], 0)['state'], 'forward_candidate')
        model.run.return_value = [logits(55), logits(45)]
        self.assertEqual(estimator.update(IMAGE, [BOX], [FACE], 400)['state'], 'away_candidate')
        model.run.return_value = [logits(51), logits(45)]
        self.assertEqual(estimator.update(IMAGE, [BOX], [FACE], 800)['state'], 'unknown')

    def test_sampling_and_missing_face_never_reuse_old_identity(self):
        model = session()
        estimator = GazeEstimator(session=model)
        estimator.update(IMAGE, [BOX], [FACE], 0)
        estimator.update(IMAGE, [BOX], [FACE], 100)
        self.assertEqual(model.run.call_count, 1)
        self.assertEqual(estimator.update(IMAGE, [], [], 200)['state'], 'unknown')
        estimator.update(IMAGE, [BOX], [FACE], 210)
        self.assertEqual(model.run.call_count, 2)
        result = estimator.update(IMAGE, [BOX, BOX], [FACE, FACE], 220)
        self.assertEqual(result['reason'], 'single_face_required')

    def test_small_crop_invalid_output_and_time_reset_are_unknown(self):
        model = session()
        estimator = GazeEstimator(session=model)
        self.assertEqual(estimator.update(IMAGE, [[0, 0, .1, .1]], [FACE], 0)['state'], 'unknown')
        model.run.return_value = [np.full((1, 90), np.nan), logits(45)]
        self.assertEqual(estimator.update(IMAGE, [BOX], [FACE], 100)['state'], 'unknown')
        model.run.return_value = [logits(45), logits(45)]
        self.assertEqual(estimator.update(IMAGE, [BOX], [FACE], 50)['state'], 'forward_candidate')

    def test_both_head_and_fresh_down_gaze_needed_and_unknown_resets(self):
        estimator = GazeEstimator(session=session())
        policy = GazeDownPolicy(estimator)
        result = {'source_angles': {'pitch': 20}, 'angles': {'pitch': 20}}
        for ms in (1000, 1400):
            estimator.latest = dict(state='away_candidate', pitch=-32., yaw=0., sample_ms=ms)
            blocked, _ = policy(feature_window(ms), result)
        self.assertTrue(blocked)
        estimator.latest = dict(state='unknown', pitch=None, yaw=None, sample_ms=1500)
        self.assertFalse(policy(feature_window(1500), result)[0])
        estimator.latest = dict(state='away_candidate', pitch=-32., yaw=0., sample_ms=1600)
        self.assertFalse(policy(feature_window(1600), result)[0])
        self.assertFalse(policy(feature_window(1700), {'angles': {'pitch': 0}})[0])

    def test_gaussian_default_unchanged_and_opt_in_head_only_scores(self):
        estimator = GazeEstimator(session=session())
        default = GaussianRuntime()
        experimental = GaussianRuntime(downward_policy=GazeDownPolicy(estimator))
        for ms in range(0, 3000, 100):
            default(feature_window(ms))
            experimental(feature_window(ms))
        self.assertIsNone(default(feature_window(3000, pitch=20))['current_direction_score'])
        self.assertIsNotNone(experimental(feature_window(3000, pitch=20))['current_direction_score'])
        self.assertIsNone(experimental(feature_window(3100, faces=0))['current_direction_score'])
