"""고정 카드와 개발자 도구 전환의 화면·입력 보존 검사. 실제 장치는 열지 않는다."""
from pathlib import Path
import ast
import re
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from streamlit.proto.GapSize_pb2 import GapSize
from streamlit.proto.Block_pb2 import Block

from frontend.dashboard_data import SessionFiles
from integration import IntegrationService


ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / '공유용_데이터_계약' / 'examples' / 'session.json'
CARD_HEIGHTS = {'live-overview': 'content', 'live-health': 210,
                'live-transcript': 210, 'live-notes': 250, 'live-assistant': 170}


class FakeMonitor:
    path = Path('test-performance.jsonl')
    error = None

    def __init__(self): self.running = True
    def is_running(self): return self.running


class FakeLive:
    def __init__(self, service, sid):
        self.service, self.sid = service, sid
        self.video_enabled = self.model_enabled = True
        self.reidentification_enabled = False
        self.reidentification_error = None
        self.performance_monitor = None
        self.preview = False
        self.fail_monitor = False
        self.monitor_starts = self.monitor_stops = 0
        self.running = True

    def has_started(self): return True
    def is_running(self): return self.running
    def stop(self): self.running = False
    def snapshot(self):
        return {'start_ms': 1000, 'features': {'faces': [], 'sampling': {'actual_face_hz': 5.92}},
                'quality': {'status': 'valid'}}, None, True
    def transcription_error(self): return None
    def checkpoint_status(self): return None
    def participant_snapshot(self): return []
    def calibration_state(self): return {'status': 'ready', 'progress': 1}
    def request_head_calibration(self, identity=None): return True
    def developer_preview_enabled(self): return self.preview
    def set_developer_preview(self, enabled): self.preview = enabled
    def diagnostics(self):
        return {'video_result_age_ms': 100, 'stt': {
            'source': 'loopback', 'lag_ms_latest': 2562, 'queue_size': 18, 'rtf': 0.6,
            'dropped_chunks': 0, 'input_overflows': 0, 'discontinuities': 0,
            'processed_windows': 14, 'empty_windows': 1}}
    def start_performance_monitor(self, directory):
        if self.fail_monitor:
            raise OSError('테스트 경로 접근 실패')
        self.performance_monitor = FakeMonitor()
        self.monitor_starts += 1
    def stop_performance_monitor(self):
        self.performance_monitor.running = False
        self.monitor_stops += 1


class LiveLayoutTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        environment = patch.dict('os.environ', {'LECTURER_SESSION_DIR': self.folder.name})
        environment.start()
        self.addCleanup(environment.stop)
        service, sid = SessionFiles(self.folder.name).load(SAMPLE)
        self.live = FakeLive(service, sid)
        self.app = AppTest.from_file(str(ROOT / 'frontend' / 'app.py'), default_timeout=20)
        self.app.session_state['live'] = self.live
        self.app.session_state['page'] = '라이브 세션'
        self.app.session_state['saved_sid'] = sid

    def cards(self):
        return {key: (item.proto.id, item.proto.height_config.pixel_height
                      if item.proto.height_config.WhichOneof('height_spec') == 'pixel_height' else 'content')
                for item in self.app if item.type == 'flex_container'
                for key in CARD_HEIGHTS if item.proto.id.endswith('-' + key)}

    def button(self, label):
        return next(item for item in self.app.button if item.label == label)

    def test_overview_and_timeline_share_a_card_beside_assistant_and_notes(self):
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual({key: height for key, (_, height) in self.cards().items()}, CARD_HEIGHTS)
        root = next(item for item in self.app if item.type == 'flex_container'
                    and item.proto.id.endswith('-live-dashboard'))
        rows = [item for item in root.children.values() if item.type == 'flex_container']
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual([item.proto.weight for item in row.children.values()], [0.5, 0.5])
        expected = [(('live-overview',), ('live-assistant', 'live-notes')),
                    (('live-transcript',), ('live-health',))]
        for row, column_keys in zip(rows, expected):
            for column, keys in zip(row.children.values(), column_keys):
                for key in keys:
                    self.assertTrue(any(item.type == 'flex_container' and item.proto.id.endswith('-' + key)
                                        for item in column.children.values()))
        overview = next(item for item in self.app if item.type == 'flex_container'
                        and item.proto.id.endswith('-live-overview'))
        self.assertTrue(any(item.type == 'selectbox' and item.key == 'live-chart-range' for item in overview))
        self.assertTrue(any(item.type == 'subheader' and item.value == '공통 타임라인' for item in overview))

    def test_card_spacing_uses_streamlit_layout_instead_of_css_height_overrides(self):
        self.app.run()
        self.assertFalse(self.app.exception)
        cards = [item for item in self.app if item.type == 'flex_container'
                 and any(item.proto.id.endswith('-' + key) for key in CARD_HEIGHTS)]
        self.assertEqual(len(cards), 5)
        for card in cards:
            gap = GapSize.XXSMALL if card.proto.id.endswith('-live-overview') else GapSize.XSMALL
            self.assertEqual(card.proto.flex_container.gap_config.gap_size, gap)
        # 카드의 측정값과 화면의 간격이 달라졌던 CSS 재정의를 다시 넣지 않는다.
        styles = '\n'.join(item.value for item in self.app.markdown if '<style>' in item.value)
        for body in re.findall(r'\.st-key-live-dashboard[^{}]*\{([^{}]*)\}', styles):
            self.assertNotRegex(body, r'(?:^|;)\s*(?:gap|row-gap|column-gap|line-height|height|font-size|margin(?:-[a-z]+)?|padding(?:-[a-z]+)?)\s*:')

    def test_stop_action_is_in_page_header_outside_fixed_overview(self):
        self.app.run()
        self.assertFalse(self.app.exception)
        overview = next(item for item in self.app if item.type == 'flex_container'
                        and item.proto.id.endswith('-live-overview'))
        rows = [item for item in overview.children.values() if item.type == 'flex_container']
        self.assertEqual(len(rows), 1)
        timeline_columns = list(rows[0].children.values())
        self.assertFalse(any(item.type == 'button' and item.label == '수업 종료' for item in overview))
        self.assertFalse(any(item.type == 'caption' and '진행 시간' in item.value for item in overview))
        self.assertTrue(any('진행 시간' in item.value for item in self.app.caption))
        header = next(item for item in self.app if item.type == 'flex_container'
                      and item.proto.id.endswith('-live-page-header'))
        header_row = next(item for item in header.children.values() if item.type == 'flex_container')
        self.assertEqual(header_row.proto.flex_container.justify, Block.FlexContainer.SPACE_BETWEEN)
        action_group = list(header_row.children.values())[1]
        self.assertEqual([item.label for item in action_group if item.type == 'button'],
                         ['정면 기준 재보정', '수업 종료'])
        self.assertTrue(any(item.type == 'caption' and '진행 시간' in item.value for item in header.children.values()))
        self.assertTrue(any(item.type == 'subheader' and item.value == '공통 타임라인' for item in timeline_columns[0]))
        self.assertTrue(any(item.type == 'metric' and item.label == '최근 실험 구간 점수' for item in timeline_columns[1]))
        footer = next(item for item in overview.children.values()
                      if item.type == 'caption' and '점수 상태:' in item.value)
        self.assertFalse(footer.proto.wrap)
        self.assertEqual(footer.proto.help, footer.value)
        self.assertEqual(overview.proto.height_config.WhichOneof('height_spec'), 'use_content')

    def test_overview_collecting_and_long_title_keep_controls_and_footer(self):
        title = '긴 수업 제목 ' * 30
        self.live.sid = self.live.service.start_session(title)
        self.live.service.stop_session(self.live.sid)
        with patch.object(self.live, 'calibration_state', return_value={'status': 'collecting', 'progress': 0.25}):
            self.app.run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.button('정면 기준 재보정').disabled)
        self.assertFalse(self.button('수업 종료').disabled)
        self.assertTrue(any('정면 기준 수집 중 · 25%' in item.value for item in self.app.caption))
        header = next(item for item in self.app.caption if '진행 시간' in item.value)
        self.assertFalse(header.proto.wrap)
        self.assertEqual(header.proto.help, title)
        self.assertEqual(self.cards()['live-overview'][1], 'content')

    def test_preview_and_log_toggles_preserve_cards_range_and_session(self):
        self.app.run()
        self.app.selectbox(key='live-chart-range').select('최근 10분').run()
        cards = self.cards()
        before = self.live.service.get_dashboard_snapshot(self.live.sid)
        with patch('streamlit.rerun') as rerun:
            for label in ('개발자 모드 켜기', '성능 로그 수집 시작', '개발자 모드 끄기', '성능 로그 수집 중지'):
                self.button(label).click().run()
                self.assertFalse(self.app.exception)
                self.assertEqual(self.cards(), cards)
                self.assertEqual(self.app.selectbox(key='live-chart-range').value, '최근 10분')
                self.assertIs(self.app.session_state['live'], self.live)
            rerun.assert_not_called()
        self.assertEqual(self.live.service.get_dashboard_snapshot(self.live.sid), before)
        self.assertEqual((self.live.monitor_starts, self.live.monitor_stops), (1, 1))

    def test_monitor_failure_does_not_remove_cards_or_stop_session(self):
        self.app.run()
        cards = self.cards()
        self.live.fail_monitor = True
        self.button('성능 로그 수집 시작').click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.cards(), cards)
        self.assertTrue(self.live.is_running())
        self.assertTrue(any('테스트 경로 접근 실패' in item.value for item in self.app.warning))

    def test_note_is_saved_once_after_tool_changes(self):
        self.app.run()
        self.button('개발자 모드 켜기').click().run()
        self.app.chat_input(key='live-note-text').set_value('첫 줄\n둘째 줄').run()
        self.app.run()
        notes = self.live.service.get_dashboard_snapshot(self.live.sid)['notes']
        self.assertEqual(sum(event['payload']['text'] == '첫 줄\n둘째 줄' for event in notes), 1)
        self.assertTrue(any(item.value == '첫 줄\n둘째 줄' for item in self.app.text))

    def test_current_score_warning_and_suggestion_notification_are_not_repeated(self):
        snapshot = self.live.service.get_dashboard_snapshot(self.live.sid)
        suggestion = dict(snapshot['suggestions'][0], created_ms=1000)
        snapshot['suggestions'] = [suggestion]
        snapshot['teaching_suggestions'] = []
        current = {'state': 'current', 'value': 55, 'window': None, 'age_ms': 0}
        self.app.session_state['live_score_thresholds'] = {self.live.sid: 55}
        with patch.object(self.live.service, 'get_dashboard_snapshot', return_value=snapshot), \
             patch('frontend.dashboard_data.live_score_display', return_value=current) as display, \
             patch('streamlit.toast') as toast:
            self.app.run()
            self.assertFalse(self.app.exception)
            self.assertTrue(any('설정 기준 55점 이하' in item.value for item in self.app.warning))
            self.assertTrue(any(item.type == 'flex_container' and item.proto.id.endswith('-score-band-red')
                                for item in self.app))
            self.app.run()
            self.assertEqual(toast.call_count, 1)
            display.return_value = {'state': 'held', 'value': 55, 'window': {'end_ms': 0}, 'age_ms': 6000}
            self.app.run()
            self.assertFalse(self.app.exception)
            self.assertFalse(any('설정 기준 55점 이하' in item.value for item in self.app.warning))
            self.assertTrue(any(item.type == 'flex_container' and item.proto.id.endswith('-score-band-unknown')
                                for item in self.app))

    def test_long_content_and_empty_content_keep_the_same_card_geometry(self):
        self.app.run()
        cards = self.cards()
        before = self.live.service.get_dashboard_snapshot(self.live.sid)
        transcript = before['transcripts'][0]
        payload = {**transcript['payload'], 'text': '긴 전사 ' * 1000, 'is_edited': True,
                   'revision': transcript['payload']['revision'] + 1, 'replaces_event_id': transcript['event_id']}
        self.live.service.receive_event(self.live.service.make_event(self.live.sid, 'transcript_edited', payload,
            transcript['start_ms'], transcript['end_ms'], 'ui'))
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.cards(), cards)
        empty = IntegrationService()
        sid = empty.start_session('빈 테스트 수업')
        empty.stop_session(sid)
        self.live.service, self.live.sid = empty, sid
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.cards(), cards)

    def test_accepting_suggestion_records_evidence_once(self):
        self.app.run()
        cards = self.cards()
        snapshot = self.live.service.get_dashboard_snapshot(self.live.sid)
        suggestion = snapshot['suggestions'][0]
        with patch('streamlit.rerun') as rerun:
            self.button('제안 메모로 저장').click().run()
            rerun.assert_not_called()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.cards(), cards)
        notes = self.live.service.get_dashboard_snapshot(self.live.sid)['notes']
        self.assertEqual(sum(suggestion['event_id'] in event['payload']['linked_event_ids'] for event in notes), 1)
        self.assertIn(suggestion['event_id'], self.app.session_state['hidden_suggestions'])

    def test_only_one_live_refresh_region_and_no_busy_widget_layout_selector(self):
        source = (ROOT / 'frontend' / 'app.py').read_text(encoding='utf-8')
        live_page = next(node for node in ast.parse(source).body
                         if isinstance(node, ast.FunctionDef) and node.name == 'live_page')
        fragments = [node for node in ast.walk(live_page) if isinstance(node, ast.FunctionDef)
                     and any(isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                             and decorator.func.attr == 'fragment' for decorator in node.decorator_list)]
        self.assertEqual(len(fragments), 1)
        self.assertNotIn('stStatusWidget', source)

    def test_stop_session_still_saves_and_opens_review(self):
        self.app.run()
        self.button('수업 종료').click().run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.live.is_running())
        self.assertEqual(self.app.session_state['page'], '사후 검토')
        self.assertTrue(SessionFiles(self.folder.name).path(self.live.sid).exists())


if __name__ == '__main__':
    unittest.main()
