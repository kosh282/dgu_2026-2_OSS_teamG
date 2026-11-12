import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engagement_model.webcam_demo import GuidedYawDiagnostic, PoseInputDiagnostics, save_yaw_diagnostic


class YawLoggingTests(unittest.TestCase):
    def test_opt_in_report_contains_aggregates_but_no_individual_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            guided = GuidedYawDiagnostic(report_sink=lambda report, rows, start:
                save_yaw_diagnostic(report, rows, start, folder))
            guided.started_ms = 1000
            guided.state = 'running'
            values = {'pitch': 0.0, 'yaw': -20.0, 'roll': 0.0}
            for ts in (1000, 1100, 2100):
                guided.rows.append({'timestamp_ms': ts, 'stage': 'left', 'phase': 'hold',
                    'face_count': 1, 'quality_status': 'valid', 'window_quality': 'valid',
                    'track_changed': False, 'current_score': 75.0,
                    'pose_diagnostics': {'reprojection_error_px': 2.0},
                    'raw': values, 'smoothed': values, 'relative': values, 'used': values,
                    'relative_formula_error': {'yaw': 0.0}, 'used_formula_error': {'yaw': 0.0},
                    'private_identity': 'MUST NOT BE WRITTEN'})
            with contextlib.redirect_stdout(io.StringIO()):
                guided.finish('완료')
            paths = list(Path(folder).glob('*.json'))
            self.assertEqual(len(paths), 1)
            content = paths[0].read_text(encoding='utf-8')
            self.assertNotIn('MUST NOT BE WRITTEN', content)
            data = json.loads(content)
            self.assertTrue(data['summary']['completed'])
            self.assertEqual([row['sample_count'] for row in data['timeline']], [2, 1])
            self.assertAlmostEqual(data['timeline'][0]['yaw']['raw']['mean'], -20)
            self.assertEqual(data['timeline'][0]['pose_input']['reprojection_error_px']['mean'], 2.0)
            self.assertNotIn('rows', data)

    def test_cli_log_requires_diagnostic_mode(self):
        from engagement_model.webcam_demo import main
        with patch('sys.argv', ['webcam_demo', '--yaw-log']), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)

    def test_pose_diagnostics_handle_absent_and_stable_face(self):
        import numpy as np
        diagnostic = PoseInputDiagnostics()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = [(0.5, 0.5, 0.0)] * 478
        for idx, xy in {1: (0.5, 0.48), 152: (0.5, 0.72), 33: (0.4, 0.4),
                        263: (0.6, 0.4), 61: (0.43, 0.60), 291: (0.57, 0.60)}.items():
            landmarks[idx] = (*xy, 0.0)
        first = diagnostic.observe([landmarks], frame)
        second = diagnostic.observe([landmarks], frame)
        self.assertIsNone(first['landmark_shape_change_ratio'])
        self.assertAlmostEqual(second['landmark_shape_change_ratio'], 0)
        self.assertGreaterEqual(second['reprojection_error_px'], 0)
        diagnostic.observe([], frame)
        self.assertIsNone(diagnostic.observe([landmarks], frame)['landmark_shape_change_ratio'])


if __name__ == '__main__':
    unittest.main()
