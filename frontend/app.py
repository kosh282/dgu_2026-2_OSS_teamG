"""Figma lecturer dashboard: preparation, live session and post-session review."""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from frontend.dashboard_data import SessionFiles, chart, clock, overlap, score

st.set_page_config(page_title='교수자 대시보드', page_icon='📊', layout='wide')
files = SessionFiles(os.environ.get('LECTURER_SESSION_DIR', str(Path.home() / 'lecturer_dashboard' / 'sessions')))
for key, value in {'page':'세션 시작 준비','live':None,'review':None,'saved_sid':None,'hidden_suggestions':set()}.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown('''<style>
.stApp{background:#f5f6f7;color:#1f2328} [data-testid="stSidebar"]{background:white;border-right:1px solid #c9ced4}
.block-container{padding:24px;max-width:1440px} h1{font-size:24px!important;padding-top:0!important} h3{font-size:16px!important}
[data-testid="stVerticalBlockBorderWrapper"]>div{background:white;border-radius:8px}
[data-testid="stBaseButton-primary"]{background:#1f2328;border-color:#1f2328;color:white}
[data-testid="stMetric"]{background:white;border:1px solid #c9ced4;border-radius:8px;padding:12px}
[data-testid="stMetricValue"]{font-size:22px} button{min-height:42px} .stCaption{color:#57606a}
@media(max-width:640px){.block-container{padding:16px}}
</style>''', unsafe_allow_html=True)


def navigate(page):
    st.session_state.page = page
    st.rerun()


def active():
    return st.session_state.live


def persist_live():
    live = active()
    if live and live.has_started() and not live.is_running() and st.session_state.saved_sid != live.sid:
        try:
            files.save(live.service, live.sid)
            st.session_state.saved_sid = live.sid
        except (OSError, ValueError) as exc:
            st.error(f'수업 저장 실패: {exc}')


persist_live()
with st.sidebar:
    st.subheader('교수자 수업 도우미')
    for name in ['세션 시작 준비','라이브 세션','사후 검토']:
        if st.button(name, use_container_width=True, type='primary' if st.session_state.page == name else 'secondary'):
            navigate(name)
    st.divider()
    st.caption('지난 수업 목록')
    for sid, session in files.list():
        if sid is None:
            st.caption(session['title'])
            continue
        if st.button(f"{session['started_at_utc'][:10]} · {session['title']}", key=f'open-{sid}', use_container_width=True):
            st.session_state.review = files.load(files.path(sid))
            navigate('사후 검토')
    uploaded = st.file_uploader('파일에서 열기', type=['json'])
    if uploaded is not None and st.button('세션 JSON 열기'):
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'import.json'
                path.write_bytes(uploaded.getvalue())
                service, sid = files.load(path)
            # Preserve original records when an imported file has the same session ID.
            if files.path(sid).exists():
                st.warning('같은 ID의 수업이 이미 저장되어 있어 기존 수업을 엽니다.')
                service, sid = files.load(files.path(sid))
            else:
                files.save(service, sid)
            st.session_state.review = service, sid
            navigate('사후 검토')
        except Exception as exc:
            st.error(f'공통 계약 JSON을 열 수 없습니다: {exc}')
    st.caption('원본 영상·음성은 저장하지 않습니다. 저장한 전사·메모·집계는 이 컴퓨터에서 다시 확인합니다.')


def event_text(event):
    st.caption(clock(event['start_ms']))
    st.write(event['payload'].get('text', event['payload'].get('description','')))


def add_note(service, sid, text, at, linked=None):
    event = service.make_event(sid,'note_created',{'text':text,'linked_event_ids':linked or []},at,at,'ui')
    service.receive_event(event)


def resources(service, sid, snapshot, readonly=False):
    suggestions = [e for e in snapshot['suggestions'] if e['event_id'] not in st.session_state.hidden_suggestions]
    if not suggestions:
        st.caption('현재 표시할 자료 제안이 없습니다. 자료 검색 모듈의 이벤트를 기다립니다.')
    for event in suggestions:
        payload = event['payload']
        with st.container(border=True):
            st.caption(clock(event['start_ms']))
            st.write(payload['title'])
            st.write(payload['description'])
            source = payload['source']
            st.caption(f"출처: {source['document_name']} · {source.get('page') or '페이지 미지정'}쪽")
            if event.get('evidence_needs_refresh'):
                st.warning('근거 전사가 수정되었습니다. 자료 제안을 다시 확인하세요.')
            if not readonly:
                a,b = st.columns(2)
                if a.button('제안 메모로 저장',key='accept-'+event['event_id'],use_container_width=True):
                    add_note(service,sid,payload['title'],service.elapsed_ms(sid),[event['event_id']])
                    st.session_state.hidden_suggestions.add(event['event_id'])
                    st.rerun()
                if b.button('이번 제안 숨기기',key='hide-'+event['event_id'],use_container_width=True):
                    st.session_state.hidden_suggestions.add(event['event_id'])
                    st.rerun()


def preparation():
    st.title('세션 시작 준비')
    st.caption('분석을 시작하기 전에 입력과 동의를 확인합니다.')
    live = active()
    running = live is not None and live.is_running()
    left,right = st.columns([2,1],gap='medium')
    with left:
        with st.container(border=True):
            st.subheader('1  분석할 수업 창 선택')
            title = st.text_input('수업 제목',key='lesson_title',disabled=running,placeholder='예: 오픈소스 소프트웨어 8주차')
            windows=[]
            window_error=None
            if sys.platform == 'win32':
                try:
                    from video_recognition.integrated_analysis import list_windows
                    windows=list_windows()
                except Exception as exc:
                    window_error=str(exc)
            else:
                window_error='실시간 창 캡처는 Windows 로컬 실행에서 사용할 수 있습니다.'
            selected=st.selectbox('분석할 수업 창',windows,index=None,format_func=lambda x:x[1],disabled=running,placeholder='수업 창을 선택하세요')
            if st.button('창 목록 새로고침',disabled=running):
                st.rerun()
            if window_error:
                st.info(window_error)
            st.caption('미리보기는 선택한 창의 현재 프레임 한 장만 표시하며 파일로 저장하지 않습니다.')
            if selected and st.button('선택한 창 미리보기',disabled=running):
                capture=None
                try:
                    from video_recognition.window_capture import WindowCapture
                    capture=WindowCapture(selected[0],timeout=3)
                    st.image(capture.read(),channels='BGR',use_container_width=True)
                except Exception as exc:
                    st.error(f'미리보기를 가져올 수 없습니다: {exc}')
                finally:
                    if capture:
                        capture.close()
        with st.container(border=True):
            st.subheader('2  수업 음성 입력')
            st.checkbox('시스템 음성 수집',disabled=True)
            st.checkbox('교수자 마이크도 함께 수집',disabled=True)
            st.caption('음성 입력·STT 모듈 미연결 · 입력 레벨을 측정하지 않습니다.')
        with st.container(border=True):
            st.subheader('3  강의 자료 업로드 (선택)')
            st.file_uploader('PDF · PPT 강의 자료',type=['pdf','ppt','pptx'],accept_multiple_files=True,key='materials',disabled=running)
            st.caption('선택한 자료는 현재 브라우저 세션에만 보관됩니다. 검색 모듈 연결 전에는 자동 자료 제안에 사용되지 않습니다.')
        with st.container(border=True):
            st.subheader('4  동의·저장 안내')
            consent=st.checkbox('촬영·분석에 동의한 수업입니다',key='consent',disabled=running)
            storage=st.checkbox('전사·메모·집계를 로컬 JSON으로 저장하고 세션 삭제로 제거함을 확인했습니다',key='storage',disabled=running)
            st.caption('동의를 거부하거나 철회해도 수업 참여에 불이익이 없습니다.')
    with right:
        with st.container(border=True):
            st.subheader('점검 요약')
            for name,state in [('영상 캡처','창 선택됨' if selected else '대기'),('수업 음성','미연결'),('전사 모델 (sherpa-onnx)','미연결'),('영상 분석 모델 (MediaPipe)','설치됨' if importlib.util.find_spec('mediapipe') else '미설치'),('참여 모델','실시간 연결 대기')]:
                st.write(f'{name} · **{state}**')
            st.caption('설치 여부는 실제 모델 구동 성공 여부와 다릅니다.')
        with st.container(border=True):
            if running:
                st.info('진행 중인 수업이 있습니다.')
                if st.button('라이브 세션으로 이동',use_container_width=True):
                    navigate('라이브 세션')
            elif st.button('세션 시작',type='primary',use_container_width=True,disabled=not(selected and title.strip() and consent and storage)):
                try:
                    from frontend.live_session import LiveSession
                    live=LiveSession(selected[0])
                    live.start(title)
                    st.session_state.live=live
                    st.session_state.hidden_suggestions=set()
                    navigate('라이브 세션')
                except Exception as exc:
                    st.error(f'수업을 시작하지 못했습니다: {exc}')
            st.caption('수업 제목·창 선택·동의를 확인하면 영상 분석을 시작합니다. 음성 없이도 시작할 수 있습니다.')


def live_page():
    st.title('라이브 세션')
    live=active()
    if not live or not live.has_started():
        st.info('세션 시작 준비에서 수업 창을 선택하고 수업을 시작하세요.')
        if st.button('세션 시작 준비로 이동'):
            navigate('세션 시작 준비')
        return
    @st.fragment(run_every=1 if live.is_running() else None)
    def panel():
        snapshot=live.service.get_dashboard_snapshot(live.sid)
        duration=snapshot['session']['duration_ms']
        st.caption(snapshot['session']['title'])
        a,b=st.columns(2)
        a.metric('진행 중' if live.is_running() else '수업 종료',clock(duration))
        windows=snapshot['engagement_windows']
        value=score(windows[-1]) if windows else None
        label='측정 불가' if value is None else ('좋음' if value>=70 else '보통' if value>=40 else '나쁨')
        b.metric('현재 참여 상태',label,delta=None if value is None else f'{value:g}점',delta_color='off')
        result,error,_=live.snapshot()
        if error:
            st.error(f'영상 분석 오류: {error}')
        if result:
            st.caption(f"영상 품질 {result['quality']['status']} · 얼굴 후보 {len(result['features']['faces'])}개 (참가자 수·참여 점수가 아닙니다)")
        with st.container(border=True):
            st.subheader('공통 타임라인')
            st.caption('참여 점수 0–100 · 좋음 ≥70 · 보통 40–69 · 나쁨 <40')
            st.plotly_chart(chart(windows,snapshot['notes'],duration),use_container_width=True,key='live-chart')
            if not windows:
                st.caption('참여 모델의 engagement_window 이벤트가 들어오면 그래프와 현재 상태가 함께 갱신됩니다.')
        with st.container(border=True):
            st.subheader('강의 조교 · 자료 제안')
            resources(live.service,live.sid,snapshot)
        with st.container(border=True):
            st.subheader('실시간 전사 · 분석 상태')
            st.caption(' · '.join(f'{key}: {value}' for key,value in snapshot['status'].items()))
            for event in snapshot['transcripts'][-8:]:
                event_text(event)
            for event in snapshot['partial_transcripts']:
                st.caption('인식 중: '+event['payload']['text'])
            if not snapshot['transcripts'] and not snapshot['partial_transcripts']:
                st.caption('전사 모듈의 입력을 기다립니다.')
        with st.container(border=True):
            st.subheader('교수자 메모')
            with st.form('live-note',clear_on_submit=True):
                text=st.text_input('메모 입력',placeholder='이 설명 다시 짚기')
                if st.form_submit_button('메모 저장') and text.strip():
                    add_note(live.service,live.sid,text.strip(),live.service.elapsed_ms(live.sid))
                    if not live.is_running():
                        files.save(live.service,live.sid)
                    st.rerun()
            for event in reversed(snapshot['notes'][-5:]):
                event_text(event)
        if not live.is_running():
            persist_live()
            if st.button('수업 결과 보기',type='primary'):
                st.session_state.review=live.service,live.sid
                navigate('사후 검토')
    panel()
    if live.is_running():
        if st.button('수업 종료',type='primary'):
            try:
                live.stop()
                files.save(live.service,live.sid)
                st.session_state.saved_sid=live.sid
                st.session_state.review=live.service,live.sid
                navigate('사후 검토')
            except Exception as exc:
                st.error(f'수업 종료·저장 실패: {exc}')
        with st.expander('개발자 도구'):
            enabled=live.developer_preview_enabled()
            if st.button('개발자 모드 끄기' if enabled else '개발자 모드 켜기'):
                live.set_developer_preview(not enabled)
                st.rerun()


def review_page():
    st.title('사후 검토')
    review=st.session_state.review
    if not review:
        st.info('왼쪽의 지난 수업을 선택하거나 공통 계약 세션 JSON을 열어주세요.')
        if st.button('공통 계약 예제로 확인'):
            st.session_state.review=files.load(ROOT/'공유용_데이터_계약'/'examples'/'session.json')
            st.rerun()
        return
    service,sid=review
    snapshot=service.get_dashboard_snapshot(sid)
    session=snapshot['session']
    duration=session['duration_ms']
    st.caption(f"{session['title']} · 길이 {clock(duration)}")
    windows=snapshot['engagement_windows']
    measured=[(score(e),max(1,e['end_ms']-e['start_ms'])) for e in windows if score(e) is not None]
    average=sum(s*w for s,w in measured)/sum(w for _,w in measured) if measured else None
    ratios=[(e['payload'].get('observation',{}).get('observable_ratio'),max(1,e['end_ms']-e['start_ms'])) for e in windows]
    ratios=[(r,w) for r,w in ratios if r is not None]
    columns=st.columns(4)
    for column,label,value in zip(columns,['수업 전체 참여 점수','분석 가능률','자료 제안','교수자 메모'],[f'{average:.1f}' if average is not None else '측정 불가',f'{sum(r*w for r,w in ratios)/sum(w for _,w in ratios):.0%}' if ratios else '측정 불가',f"{len(snapshot['suggestions'])}건",f"{len(snapshot['notes'])}개"]):
        column.metric(label,value)
    range_key=f'range-{sid}'
    chart_key=f'review-chart-{sid}'
    def select_chart_range():
        selection=st.session_state.get(chart_key, {}).get('selection', {})
        boxes=selection.get('box', [])
        if boxes and 'x' in boxes[0]:
            bounds=sorted(int(float(x)*1000) for x in boxes[0]['x'])
            st.session_state[range_key]=tuple(max(0,min(duration,x)) for x in bounds)
    with st.container(border=True):
        st.subheader('공통 타임라인')
        st.caption('아래에서 구간을 선택하면 전사·메모·자료·피드백이 함께 좁혀집니다.')
        selected=st.slider('검토 구간 (밀리초)',0,max(1,duration),value=(0,max(1,duration)),key=range_key)
        st.caption(f'{clock(selected[0])} – {clock(selected[1])}')
        if st.button('선택 해제 · 전체 보기'):
            st.session_state.pop(range_key,None)
            st.rerun()
        st.plotly_chart(chart(windows,snapshot['notes'],duration,selected),use_container_width=True,key=chart_key,on_select=select_chart_range,selection_mode='box')
    def visible(event):
        return overlap(event,*selected)
    left,right=st.columns([2,1])
    with left:
        with st.container(border=True):
            st.subheader('전사 · 선택 구간')
            query=st.text_input('전사 검색',key=f'search-{sid}')
            transcripts=[e for e in snapshot['transcripts'] if visible(e) and query.casefold() in e['payload']['text'].casefold()]
            st.caption(f'{len(transcripts)}개 문장 · 펼쳐서 수정할 수 있습니다.')
            for event in transcripts:
                payload=event['payload']
                with st.expander(f"{clock(event['start_ms'])} · {payload['text']}"):
                    with st.form('edit-'+event['event_id']):
                        text=st.text_area('전사 수정',payload['text'])
                        if st.form_submit_button('수정 저장'):
                            try:
                                updated=dict(payload,text=text,revision=payload['revision']+1,is_edited=True,replaces_event_id=event['event_id'])
                                change=service.make_event(sid,'transcript_edited',updated,event['start_ms'],event['end_ms'],'ui')
                                service.receive_event(change)
                                files.save(service,sid)
                                st.rerun()
                            except Exception as exc:
                                st.error(f'전사를 수정하지 못했습니다: {exc}')
            if not transcripts:
                st.caption('이 구간 또는 검색어에 해당하는 확정 전사가 없습니다.')
    with right:
        with st.container(border=True):
            st.subheader('교수자 메모 · 선택 구간')
            notes=[e for e in snapshot['notes'] if visible(e)]
            for event in notes:
                event_text(event)
            if not notes:
                st.caption('저장된 메모가 없습니다.')
        with st.container(border=True):
            st.subheader('사후 피드백 · 선택 구간')
            feedback=[e for e in snapshot['feedback'] if visible(e)]
            for event in feedback:
                event_text(event)
                if event.get('evidence_needs_refresh'):
                    st.warning('근거 전사가 수정되어 피드백 재검토가 필요합니다.')
            if not feedback:
                st.caption('저장된 피드백이 없습니다. 피드백 생성 모듈은 아직 연결되지 않았습니다.')
            st.caption('관찰 신호만으로 이해 부족이나 교수법의 문제를 확정하지 않습니다.')
        with st.container(border=True):
            st.subheader('관련 자료 · 선택 구간')
            filtered=dict(snapshot,suggestions=[e for e in snapshot['suggestions'] if visible(e)])
            resources(service,sid,filtered,readonly=True)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'session.json'
        service.save_session(sid,path)
        st.download_button('세션 JSON 저장',path.read_bytes(),file_name=f'{sid}.json',mime='application/json')
    with st.expander('선택한 수업 삭제'):
        st.caption('이 앱에 저장한 수업 기록을 삭제합니다. 별도로 다운로드한 파일은 남습니다.')
        confirmed=st.checkbox('이 수업의 전사·메모·집계를 삭제합니다',key='delete-'+sid)
        if st.button('수업 삭제',disabled=not confirmed):
            try:
                files.path(sid).unlink(missing_ok=True)
                if active() and active().sid == sid:
                    st.session_state.live=None
                st.session_state.review=None
                st.rerun()
            except OSError as exc:
                st.error(f'삭제 실패: {exc}')


{'세션 시작 준비':preparation,'라이브 세션':live_page,'사후 검토':review_page}[st.session_state.page]()
