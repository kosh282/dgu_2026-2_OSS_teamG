import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from frontend.dashboard_data import (SessionFiles, chart, live_chart_data, overlap, score, suggestion_phase, transcript_score_link,
                                     related_suggestions, suggestion_in_range)

ROOT=Path(__file__).resolve().parents[2]
SAMPLE=ROOT/'공유용_데이터_계약'/'examples'/'session.json'

class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.environment=patch.dict('os.environ',{'LECTURER_SESSION_DIR':self.directory.name})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_real_timestamps_and_null_at_same_time(self):
        def event(value,eid):
            return dict(event_id=eid,start_ms=180000,end_ms=190000,payload={'model_output':{'score':value}},quality={'status':'valid'})
        windows=[event(70,'a'),event(None,'b')]
        fig=chart(windows,[],7200000)
        self.assertEqual(list(fig.data[0].x),[190,190])
        self.assertEqual(list(fig.data[0].y),[70,None])
        self.assertEqual(fig.layout.xaxis.range[1],7200)

    def test_unknown_score_is_not_zero(self):
        for value in [None,True,float('nan'),101,-1,'70']:
            self.assertIsNone(score({'payload':{'score':value},'quality':{'status':'valid'}}))

    def test_contract_model_output_value(self):
        event = {'payload': {'model_output': {'kind': 'continuous', 'value': 73.5}}, 'quality': {'status': 'valid'}}
        self.assertEqual(score(event), 73.5)

    def test_transcript_uses_completed_score_without_crossing_missing_or_future_windows(self):
        def window(end, value):
            return {'start_ms': max(0, end - 10000), 'end_ms': end,
                    'payload': {'model_output': {'value': value} if value is not None else None},
                    'quality': {'status': 'valid' if value is not None else 'not_observable'}}
        windows = [window(10000, 80), window(12000, 70), window(14000, None), window(16000, 60)]
        link = transcript_score_link({'end_ms': 13000}, windows)
        self.assertEqual((link['value'], link['delta']), (70, -10))
        link = transcript_score_link({'end_ms': 15000}, windows)
        self.assertIsNone(link['value'])
        self.assertEqual(link['reason'], '점수 보류')
        self.assertIsNone(transcript_score_link({'end_ms': 16000}, windows)['delta'])
        self.assertEqual(transcript_score_link({'end_ms': 22000}, windows)['reason'], '최근 점수 구간 없음')
        self.assertEqual(transcript_score_link({'end_ms': 9000}, windows)['reason'], '점수 구간 대기')

    def test_chart_does_not_join_missing_interval(self):
        def event(start, end, eid):
            return dict(event_id=eid, start_ms=start, end_ms=end, payload={'model_output': {'value': 80}}, quality={'status': 'valid'})
        fig = chart([event(0, 1000, 'a'), event(3000, 4000, 'b')], [], 4000)
        self.assertEqual(list(fig.data[0].y), [80, None, 80])

    def test_live_chart_preserves_gaps_and_merges_unobservable_intervals(self):
        def event(start, end, value, eid):
            return dict(event_id=eid, start_ms=start, end_ms=end,
                        payload={'model_output': {'value': value}}, quality={'status': 'valid'})
        windows = [event(0, 1000, 80, 'a'), event(3000, 5000, None, 'b'),
                   event(4000, 6000, None, 'c'), event(6000, 7000, 70, 'd')]
        notes = [{'start_ms': 3500, 'payload': {'text': '확인할 내용'}}]
        data = live_chart_data(windows, notes, 7200000)
        points = data[data['kind'] == 'score']
        self.assertEqual(points['time'].tolist(), [1, 3, 5, 6, 7])
        self.assertEqual(points['value'].notna().tolist(), [True, False, False, False, True])
        intervals = data[data['kind'] == 'unobservable']
        self.assertEqual(list(zip(intervals['time'], intervals['end'])), [(3, 6)])
        self.assertEqual(data[data['kind'] == 'bound']['time'].tolist(), [0, 7200])
        self.assertEqual(data[data['kind'] == 'note']['text'].tolist(), ['확인할 내용'])
        self.assertEqual(len(chart(windows, notes, 7200000).layout.shapes), 1)

    def test_live_chart_clips_range_without_changing_records(self):
        windows = [dict(event_id='a', start_ms=0, end_ms=10000,
                        payload={'model_output': None}, quality={'status': 'not_observable'}),
                   dict(event_id='b', start_ms=5000, end_ms=15000,
                        payload={'model_output': {'value': 80}}, quality={'status': 'valid'})]
        data = live_chart_data(windows, [], 20000, start_ms=8000)
        self.assertEqual(data[data['kind'] == 'bound']['time'].tolist(), [8, 20])
        missing = data[data['kind'] == 'unobservable'].iloc[0]
        self.assertEqual((missing['time'], missing['end']), (8, 10))
        self.assertEqual(windows[0]['start_ms'], 0)

    def test_overlap_includes_crossing_transcript(self):
        self.assertTrue(overlap({'start_ms':100,'end_ms':500},200,300))
        self.assertFalse(overlap({'start_ms':100,'end_ms':200},200,300))

    def test_suggestion_is_linked_by_evidence_across_timeline_and_phases(self):
        transcript = {'event_id': 'transcript', 'start_ms': 100, 'end_ms': 200}
        live = {'event_id': 'suggestion', 'start_ms': 300, 'end_ms': 400,
                'created_ms': 410, 'created_at_utc': None,
                'payload': {'evidence_event_ids': ['transcript']}}
        post = dict(live, created_ms=None, created_at_utc='2026-01-01T00:00:00Z')
        self.assertEqual(suggestion_phase(live), '수업 중 제안')
        self.assertEqual(suggestion_phase(post), '수업 후 제안')
        self.assertEqual(related_suggestions(transcript, [live]), [live])
        self.assertTrue(suggestion_in_range(live, 100, 200, [transcript]))
        self.assertFalse(suggestion_in_range(live, 100, 200, []))
        self.assertTrue(suggestion_in_range(live, 300, 400, []))

    def test_edit_save_reload_and_notes_do_not_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            files=SessionFiles(directory)
            service,sid=files.load(SAMPLE)
            original=service.get_dashboard_snapshot(sid)['transcripts'][0]
            payload=dict(original['payload'],text='수정한 문장',revision=original['payload']['revision']+1,is_edited=True,replaces_event_id=original['event_id'])
            event=service.make_event(sid,'transcript_edited',payload,original['start_ms'],original['end_ms'],'ui')
            service.receive_event(event)
            note=service.make_event(sid,'note_created',{'text':'새 메모','linked_event_ids':[]},1000,1000,'ui')
            service.receive_event(note)
            before=len(service.get_dashboard_snapshot(sid)['notes'])
            files.save(service,sid)
            service,sid=files.load(files.path(sid))
            files.save(service,sid)
            service,sid=files.load(files.path(sid))
            snapshot=service.get_dashboard_snapshot(sid)
            self.assertEqual(snapshot['transcripts'][0]['payload']['text'],'수정한 문장')
            self.assertEqual(len(snapshot['notes']),before)
            self.assertEqual(len(files.list()),1)

    def test_path_cannot_escape_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                SessionFiles(directory).path('../../elsewhere')

    def test_live_ui_uses_shared_snapshot(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as directory:
            service,sid=SessionFiles(directory).load(SAMPLE)
            class FakeLive:
                def __init__(self):
                    self.service=service
                    self.sid=sid
                    self.running=False
                    self.video_enabled=False
                    self.model_enabled=False
                    self.calibration_requests=0
                def has_started(self): return True
                def is_running(self): return self.running
                def snapshot(self): return None,None,False
                def transcription_error(self): return None
                def checkpoint_status(self): return None
                def diagnostics(self):
                    return {'stt': {'dropped_chunks': 0, 'input_overflows': 1, 'discontinuities': 0,
                                    'lag_ms_latest': 54, 'queue_size': 0, 'rtf': 0.3},
                            'video_result_age_ms': None}
                def developer_preview_enabled(self): return False
                def calibration_state(self): return {'status':'collecting','progress':0.5,'last_reason_code':None}
                def request_head_calibration(self): self.calibration_requests+=1
            app=AppTest.from_file(str(ROOT/'frontend'/'app.py'),default_timeout=20)
            app.session_state['live']=FakeLive()
            app.session_state['page']='라이브 세션'
            app.session_state['saved_sid']=sid
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(next(m for m in app.metric if m.label=='최근 실험 구간 점수').value,'측정 불가')
            self.assertTrue(any('수업 종료 · 진행 시간' in caption.value for caption in app.caption))
            elements=list(app)
            score_index=next(i for i, element in enumerate(elements) if element.type=='metric' and element.label=='최근 실험 구간 점수')
            timeline_index=next(i for i, element in enumerate(elements) if element.type=='subheader' and '공통 타임라인' in element.value)
            note_index=next(i for i, element in enumerate(elements) if element.type=='subheader' and element.value=='교수자 메모')
            transcript_index=next(i for i, element in enumerate(elements) if element.type=='subheader' and element.value=='실시간 전사 · 분석 상태')
            self.assertLess(score_index, timeline_index)
            self.assertLess(note_index, transcript_index)
            next(t for t in app.text_input if t.label=='메모 입력').set_value('라이브 메모')
            next(b for b in app.button if b.label=='메모 저장').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any(e['payload']['text']=='라이브 메모' for e in service.get_dashboard_snapshot(sid)['notes']))
            app.session_state['live'].running=True
            app.run()
            self.assertFalse(app.exception)
            elements=list(app)
            stop_index=next(i for i, element in enumerate(elements) if element.type=='button' and element.label=='수업 종료')
            warning_index=next(i for i, element in enumerate(elements) if element.type=='warning' and '음성 입력 누락 가능' in element.value)
            self.assertLess(warning_index, stop_index)
            app.session_state['live'].video_enabled=True
            app.session_state['live'].model_enabled=True
            app.run()
            self.assertTrue(any('정면 기준 수집 중' in c.value for c in app.caption))
            next(b for b in app.button if b.label=='정면 기준 재보정').click().run()
            self.assertEqual(app.session_state['live'].calibration_requests, 1)

    def test_empty_and_sample_ui(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(ROOT/'frontend'/'app.py'),default_timeout=20).run()
        self.assertFalse(app.exception)
        next(b for b in app.button if b.label=='사후 검토').click().run()
        self.assertFalse(app.exception)
        next(b for b in app.button if b.label=='공통 계약 예제로 확인').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value,'점수 없음')
        app.text_input[0].set_value('없는문장검색어').run()
        self.assertFalse(app.exception)

    def test_can_prepare_audio_without_video_window(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(ROOT/'frontend'/'app.py'),default_timeout=20).run()
        next(c for c in app.checkbox if c.label=='영상 분석 사용').uncheck().run()
        next(c for c in app.checkbox if c.label=='실시간 음성 전사 사용').check().run()
        next(t for t in app.text_input if t.label=='수업 제목').set_value('음성 수업').run()
        next(c for c in app.checkbox if c.label.startswith('수업 영상·음성')).check().run()
        next(c for c in app.checkbox if c.label.startswith('전사·메모')).check().run()
        self.assertFalse(app.exception)
        self.assertFalse(next(b for b in app.button if b.label=='세션 시작').disabled)

    def test_window_selection_survives_changing_window_title_and_refresh(self):
        from streamlit.testing.v1 import AppTest
        # 운영체제 판정을 이 화면의 namespace에서만 모의한다. 전역 sys.platform을 바꾸지 않는다.
        with patch('video_recognition.integrated_analysis.list_windows') as list_windows:
            list_windows.side_effect = [[(1, '수업 창'), (2, '다른 창')],
                                        [(2, '다른 창'), (1, '수업 창 새 제목')],
                                        [(2, '다른 창')]]
            script=(ROOT/'frontend'/'app.py').read_text(encoding='utf-8')
            script=script.replace('import sys', "import sys as _real_sys\nfrom types import SimpleNamespace\nsys = SimpleNamespace(platform='win32', path=_real_sys.path)")
            app=AppTest.from_string(script,default_timeout=20).run()
            selector=next(s for s in app.selectbox if s.label=='분석할 수업 창')
            selector.select('수업 창').run()
            self.assertFalse(app.exception)
            self.assertEqual(next(s for s in app.selectbox if s.label=='분석할 수업 창').value, 1)
            self.assertTrue(any('창 선택됨' in item.value for item in app.markdown))
            next(b for b in app.button if b.label=='창 목록 새로고침').click().run()
            self.assertEqual(next(s for s in app.selectbox if s.label=='분석할 수업 창').value, 1)
            next(b for b in app.button if b.label=='창 목록 새로고침').click().run()
            self.assertIsNone(next(s for s in app.selectbox if s.label=='분석할 수업 창').value)

    def test_personal_api_selection_requires_key(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(ROOT/'frontend'/'app.py'),default_timeout=20).run()
        next(s for s in app.selectbox if s.label=='문장 작성 방식').select('OpenAI').run()
        self.assertFalse(app.exception)
        self.assertTrue(any('API 키와 모델 ID' in warning.value for warning in app.warning))
        key=next(t for t in app.text_input if t.label=='개인 API 키')
        key.set_value('test-only-key').run()
        self.assertFalse(app.exception)
        self.assertFalse(any('API 키와 모델 ID' in warning.value for warning in app.warning))

if __name__=='__main__':
    unittest.main()
