"""Gaussian 독립 웹캠의 CLI·표시·자원 반환 검사."""
from pathlib import Path
import tempfile
from engagement_model.config import ModelConfig
import importlib.util
import unittest
from unittest.mock import Mock, patch
from engagement_model.gaussian import GaussianRuntime
from engagement_model.tests.test_runtime import feature_window

@unittest.skipUnless(importlib.util.find_spec('cv2'), '영상 의존성이 설치된 환경에서 렌더러 검증')
class GaussianDemoCompatibilityTests(unittest.TestCase):
    def test_panel_accepts_gaussian_output(self):
        from video_recognition.developer_view import render_feature_panel
        for config in (ModelConfig.gaussian(),):
            feature = feature_window(0, pitch=20)
            result = GaussianRuntime(config)(feature)
            with patch('video_recognition.developer_view.cv2.putText') as draw:
                panel = render_feature_panel(feature, model_result=result)
            self.assertEqual(panel.shape[1:], (1200, 3))
            text = '\n'.join(call.args[1] for call in draw.call_args_list)
            self.assertIn(result['orientation_state'], text)
            self.assertIn(result['status'], text)
            for reason in result['reason_codes']:
                self.assertIn(reason, text)

    def test_demo_records_gaussian_metadata(self):
        from video_recognition.webcam_test import open_demo_log
        import json
        for runtime in (GaussianRuntime(),):
            with tempfile.TemporaryDirectory() as directory:
                handle, path = open_demo_log(Path(directory), 'synthetic', runtime)
                handle.close()
                metadata = json.loads(path.read_text(encoding='utf-8'))
                self.assertEqual(metadata['model_version'], runtime.model_version)
                self.assertEqual(metadata['config']['scoring_method'], runtime.config.scoring_method)


@unittest.skipUnless(importlib.util.find_spec('cv2'), '영상 의존성이 설치된 환경에서 실행기 검증')
class GaussianWebcamRunnerTests(unittest.TestCase):
    def test_normal_screen_hides_debug_data_and_debug_shows_both_scores(self):
        from engagement_model.webcam_demo import panel_lines, render_panel
        from video_recognition.preview import LandmarkOverlaySmoother
        import numpy as np
        from engagement_model.gaussian import GaussianRuntime
        model = GaussianRuntime()
        feature = feature_window(0)
        model(feature)
        models = {'gaussian': model}
        normal = '\n'.join(panel_lines(feature, models))
        self.assertIn('Current direction: 100.0/100', normal)
        self.assertNotIn('Reasons:', normal)
        self.assertNotIn('aggregate', normal)
        self.assertNotIn('Relative pitch/yaw', normal)
        detailed = '\n'.join(panel_lines(feature, models, debug=True))
        self.assertIn('10s aggregate:', detailed)
        self.assertIn('Reasons:', detailed)
        self.assertIn('Relative pitch/yaw', detailed)
        with patch('engagement_model.webcam_demo.render_video_frame') as overlay:
            panel = render_panel(np.zeros((480, 640, 3), dtype=np.uint8), feature, models,
                overlay_smoother=LandmarkOverlaySmoother(), face=Mock(last_landmarks=[]),
                show_overlay=True)
            overlay.assert_not_called()
        self.assertEqual(panel.shape, (860, 1400, 3))

    def test_cli_defaults_to_simple_gaussian_and_debug_is_explicit(self):
        from engagement_model.webcam_demo import main
        for arguments, debug in (([], False), (['--debug'], True)):
            with patch('sys.argv', ['webcam_demo'] + arguments), \
                 patch('engagement_model.webcam_demo.run') as run:
                main()
            self.assertEqual(run.call_args.kwargs['primary'], 'gaussian')
            self.assertEqual(run.call_args.kwargs['debug'], debug)

    def test_summary_fields_stay_fixed_and_debug_pages_preserve_lines(self):
        from engagement_model.webcam_demo import render_panel, _debug_pages
        from engagement_model.gaussian import GaussianRuntime
        import cv2
        import numpy as np
        model = GaussianRuntime()
        feature = feature_window(0)
        model(feature)
        face = Mock(last_landmarks=[])
        diagnostics = Mock(lines=['diagnostic-' + str(i) for i in range(90)])
        pages = _debug_pages(feature, {'gaussian': model}, 'gaussian', diagnostics, face)
        self.assertGreater(len(pages), 1)
        self.assertIn('diagnostic-89', [line for page in pages for line in page])
        with patch('engagement_model.webcam_demo.cv2.putText', wraps=cv2.putText) as draw:
            first = render_panel(np.zeros((480,640,3), dtype=np.uint8), feature, {'gaussian':model},
                overlay_smoother=None, face=face, show_overlay=False, debug=True,
                diagnostics=diagnostics, debug_page=0)
        positions = {call.args[1]: call.args[2] for call in draw.call_args_list}
        self.assertEqual(positions['CURRENT DIRECTION'], (660, 65))
        self.assertEqual(positions['10s AGGREGATE'], (1030, 65))
        self.assertEqual(positions['RELATIVE HEAD PITCH / YAW'], (660, 165))
        with patch('engagement_model.webcam_demo.cv2.putText', wraps=cv2.putText) as draw:
            second = render_panel(np.zeros((480,640,3), dtype=np.uint8), feature, {'gaussian':model},
                overlay_smoother=None, face=face, show_overlay=False, debug=True,
                diagnostics=diagnostics, debug_page=1)
        self.assertEqual(first.shape, second.shape)
        self.assertIn('diagnostic-89', [line for page in pages for line in page])
        self.assertEqual(next(call.args[2] for call in draw.call_args_list if call.args[1]=='CURRENT DIRECTION'), (660, 65))

    def test_gaussian_cli_passes_sigma_settings_without_opening_camera(self):
        from engagement_model.webcam_demo import main
        with patch('sys.argv', ['webcam_demo', '--model', 'gaussian', '--sigma-pitch', '12',
                                '--sigma-yaw', '25', '--single-model']), \
             patch('engagement_model.webcam_demo.run') as run:
            main()
        self.assertEqual(run.call_args.kwargs['primary'], 'gaussian')
        config = run.call_args.kwargs['gaussian_config']
        self.assertEqual(config.gaussian_sigma_pitch_degrees, 12)
        self.assertEqual(config.gaussian_sigma_yaw_degrees, 25)

    def test_deferred_panel_never_uses_stale_display_score(self):
        from engagement_model.webcam_demo import model_lines
        result = GaussianRuntime()(feature_window(0, pitch=20))
        result['display_score'] = 100
        lines = '\n'.join(model_lines('gaussian', result))
        self.assertIn('DEFERRED', lines)
        self.assertIn('looking_down', lines)
        self.assertIn('downward_activity_uncertain', lines)
        self.assertNotIn('100.0/100', lines)

    def test_unavailable_camera_is_released(self):
        from engagement_model.webcam_demo import run
        camera = Mock()
        camera.isOpened.return_value = False
        with patch('engagement_model.webcam_demo.cv2.VideoCapture', return_value=camera):
            with self.assertRaisesRegex(RuntimeError, '웹캠 0'):
                run()
        camera.release.assert_called_once()

    def test_failed_camera_read_closes_models_and_camera(self):
        from engagement_model.webcam_demo import run
        camera, face = Mock(), Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (False, None)
        with patch('engagement_model.webcam_demo.cv2.VideoCapture', return_value=camera), \
             patch('engagement_model.webcam_demo.FaceAnalyzer', return_value=face), \
             patch('engagement_model.webcam_demo.cv2.namedWindow'), \
             patch('engagement_model.webcam_demo.cv2.resizeWindow'), \
             patch('engagement_model.webcam_demo.cv2.setMouseCallback'), \
             patch('engagement_model.webcam_demo.cv2.destroyWindow'):
            with self.assertRaisesRegex(RuntimeError, '웹캠 화면'):
                run()
        camera.release.assert_called_once()
        face.close.assert_called_once()

