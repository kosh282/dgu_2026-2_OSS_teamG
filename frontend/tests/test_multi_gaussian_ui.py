"""다인 점수 화면의 관측 불가와 메모리 전용 표시 경계를 확인한다."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from frontend.dashboard_data import SessionFiles


ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / '공유용_데이터_계약' / 'examples' / 'session.json'


class FakeLive:
    def __init__(self, service, sid):
        self.service = service
        self.sid = sid
        self.video_enabled = True
        self.model_enabled = True
        self.reidentification_enabled = True
        self.reidentification_error = None

    def has_started(self): return True
    def is_running(self): return True
    def snapshot(self): return None, None, False
    def transcription_error(self): return None
    def checkpoint_status(self): return None
    def developer_preview_enabled(self): return False
    def diagnostics(self): return {'stt': None, 'video_result_age_ms': None}
    def calibration_state(self): return {'status': 'ready', 'progress': 1}

    def participant_snapshot(self):
        return [
            {'id': 'S001', 'status': 'scored', 'calibration_status': 'ready',
             'current_direction_score': 87.5, 'score': 82.0},
            {'id': 'S002', 'status': 'missing', 'calibration_status': 'ready',
             'current_direction_score': None, 'score': None},
        ]


class MultiGaussianUITests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        environment = patch.dict('os.environ', {'LECTURER_SESSION_DIR': directory.name})
        environment.start()
        self.addCleanup(environment.stop)
        self.service, self.sid = SessionFiles(directory.name).load(SAMPLE)

    def app(self, *, live=False):
        app = AppTest.from_file(str(ROOT / 'frontend' / 'app.py'), default_timeout=20)
        if live:
            app.session_state['live'] = FakeLive(self.service, self.sid)
            app.session_state['page'] = '라이브 세션'
            app.session_state['saved_sid'] = self.sid
        return app

    def test_reidentification_defaults_off_and_requires_video(self):
        app = self.app().run()
        self.assertFalse(app.exception)
        option = next(c for c in app.checkbox if c.label == '세션 내 얼굴 재식별 사용')
        self.assertFalse(option.value)
        self.assertFalse(option.disabled)
        next(c for c in app.checkbox if c.label == '영상 분석 사용').uncheck().run()
        self.assertFalse(app.exception)
        self.assertTrue(next(c for c in app.checkbox
                             if c.label == '세션 내 얼굴 재식별 사용').disabled)

    def test_live_table_keeps_missing_scores_null(self):
        app = self.app(live=True).run()
        self.assertFalse(app.exception)
        table = next(item.value for item in app.dataframe if '임시 ID' in item.value.columns)
        self.assertEqual(table['임시 ID'].tolist(), ['S001', 'S002'])
        self.assertEqual(table.loc[0, '현재 결합 점수'], 87.5)
        self.assertEqual(table.loc[0, '최근 결합 점수'], 82.0)
        self.assertEqual(table.loc[1, '상태'], '관측 불가')
        self.assertTrue(pd.isna(table.loc[1, '현재 결합 점수']))
        self.assertTrue(pd.isna(table.loc[1, '최근 결합 점수']))

    def test_reidentification_error_is_visible_while_session_continues(self):
        app = self.app(live=True)
        app.session_state['live'].reidentification_error = '재식별 추론 실패: 개인별 점수 보류'
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(any('재식별 추론 실패' in item.value for item in app.warning))
        self.assertTrue(any(button.label == '수업 종료' for button in app.button))

    def test_group_storage_explanation_and_review_hide_personal_table(self):
        app = self.app(live=True).run()
        self.assertFalse(app.exception)
        self.assertTrue(any('저장·재열기 대상은 ID 없는 그룹 집계' in item.value
                            for item in app.caption))
        app.session_state['review'] = self.service, self.sid
        app.session_state['page'] = '사후 검토'
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(any('임시 ID' in item.value.columns for item in app.dataframe))
        self.assertFalse(any(item.value == '세션 내 ID별 Gaussian 점수' for item in app.subheader))


if __name__ == '__main__':
    unittest.main()
