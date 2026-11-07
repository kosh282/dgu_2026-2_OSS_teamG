import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from frontend.dashboard_data import SessionFiles, chart, overlap, score

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

    def test_chart_does_not_join_missing_interval(self):
        def event(start, end, eid):
            return dict(event_id=eid, start_ms=start, end_ms=end, payload={'model_output': {'value': 80}}, quality={'status': 'valid'})
        fig = chart([event(0, 1000, 'a'), event(3000, 4000, 'b')], [], 4000)
        self.assertEqual(list(fig.data[0].y), [80, None, 80])

    def test_overlap_includes_crossing_transcript(self):
        self.assertTrue(overlap({'start_ms':100,'end_ms':500},200,300))
        self.assertFalse(overlap({'start_ms':100,'end_ms':200},200,300))

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
            app=AppTest.from_file(str(ROOT/'frontend'/'app.py'),default_timeout=20)
            app.session_state['live']=FakeLive()
            app.session_state['page']='라이브 세션'
            app.session_state['saved_sid']=sid
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(next(m for m in app.metric if m.label=='최근 실험 구간 점수').value,'측정 불가')
            self.assertTrue(any('수업 종료 · 진행 시간' in caption.value for caption in app.caption))
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
            self.assertLess(stop_index, warning_index)

    def test_empty_and_sample_ui(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(ROOT/'frontend'/'app.py'),default_timeout=20).run()
        self.assertFalse(app.exception)
        next(b for b in app.button if b.label=='사후 검토').click().run()
        self.assertFalse(app.exception)
        next(b for b in app.button if b.label=='공통 계약 예제로 확인').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value,'측정 불가')
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

if __name__=='__main__':
    unittest.main()
