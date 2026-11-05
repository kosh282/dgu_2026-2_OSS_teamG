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
.block-container{padding:64px 24px 24px;max-width:1440px} h1{font-size:24px!important;padding-top:0!important} h3{font-size:16px!important}
[data-testid="stVerticalBlockBorderWrapper"]>div{background:white;border-radius:8px}
[data-testid="stBaseButton-primary"]{background:#1f2328;border-color:#1f2328;color:white}
[data-testid="stMetric"]{background:white;border:1px solid #c9ced4;border-radius:8px;padding:12px}
[data-testid="stMetricValue"]{font-size:22px} button{min-height:42px} .stCaption{color:#57606a}
@media(max-width:640px){.block-container{padding:64px 16px 16px}}
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


def stt_readiness(source):
    """선택한 오디오 입력에 필요한 패키지와 로컬 모델 파일을 확인한다."""
    from sherpa_stt.config import SttConfig
    from sherpa_stt.engine import find_transducer_files

    missing = []
    if importlib.util.find_spec('sherpa_onnx') is None:
        missing.append('sherpa-onnx 패키지')
    audio_package = 'soundcard' if source == 'loopback' else 'sounddevice'
    if importlib.util.find_spec(audio_package) is None:
        missing.append(f'{audio_package} 패키지')
    config = SttConfig()
    try:
        find_transducer_files(Path(config.model_dir))
    except (FileNotFoundError, OSError) as exc:
        missing.append(f'한국어 모델 파일 ({exc})')
    return not missing, ', '.join(missing), config


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
            st.subheader('1  영상 분석 설정')
            title = st.text_input('수업 제목',key='lesson_title',disabled=running,placeholder='예: 오픈소스 소프트웨어 8주차')
            video_enabled=st.checkbox('영상 분석 사용',value=True,key='video_enabled',disabled=running)
            model_enabled=st.checkbox('실험 행동 점수 사용',value=True,key='model_enabled',disabled=running or not video_enabled)
            st.caption('실험 점수는 영상 분석 결과를 사용합니다. 수업 시작 전에 기능을 선택하세요.')
            windows=[]
            window_error=None
            if video_enabled and sys.platform == 'win32':
                try:
                    from video_recognition.integrated_analysis import list_windows
                    windows=list_windows()
                except Exception as exc:
                    window_error=str(exc)
            elif video_enabled:
                window_error='실시간 창 캡처는 Windows 로컬 실행에서 사용할 수 있습니다.'
            selected=st.selectbox('분석할 수업 창',windows,index=None,format_func=lambda x:x[1],disabled=running or not video_enabled,placeholder='수업 창을 선택하세요')
            if st.button('창 목록 새로고침',disabled=running or not video_enabled):
                st.rerun()
            if window_error:
                st.info(window_error)
            elif video_enabled and not windows:
                st.info('표시 중이고 제목이 있는 창이 없습니다. 수업 창을 연 뒤 목록을 새로고침하세요.')
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
            stt_enabled=st.checkbox('실시간 음성 전사 사용',key='stt_enabled',disabled=running)
            audio_source=st.selectbox(
                '음성 입력 방식', ['loopback','microphone'],
                format_func=lambda value: '시스템 출력 음성 (루프백)' if value=='loopback' else '교수자 마이크',
                key='stt_source', disabled=running or not stt_enabled,
            )
            if stt_enabled:
                ready,missing,stt_config=stt_readiness(audio_source)
                if ready:
                    st.success('전사 패키지와 한국어 모델을 찾았습니다.')
                else:
                    st.warning('전사 시작 전 준비가 필요합니다: '+missing)
                if audio_source=='loopback':
                    st.caption('기본 출력 장치의 모든 소리를 받습니다. 선택한 Zoom 창의 소리만 분리하지 않습니다.')
                else:
                    st.caption('기본 마이크 입력을 받습니다. 시스템 출력 소리와 동시 수집은 지원하지 않습니다.')
            else:
                st.caption('선택 입력은 수업 세션의 공통 시계로 전사해 계약 이벤트로 전달합니다. 원본 음성 파일은 저장하지 않습니다.')
        with st.container(border=True):
            st.subheader('3  강의 자료 업로드 (선택)')
            st.file_uploader('PDF · PPT 강의 자료',type=['pdf','ppt','pptx'],accept_multiple_files=True,key='materials',disabled=running)
            st.caption('선택한 자료는 현재 브라우저 세션에만 보관됩니다. 검색 모듈 연결 전에는 자동 자료 제안에 사용되지 않습니다.')
        with st.container(border=True):
            st.subheader('4  동의·저장 안내')
            consent=st.checkbox('수업 영상·음성 수집과 분석에 동의한 수업입니다',key='consent',disabled=running)
            storage=st.checkbox('전사·메모·집계를 로컬 JSON으로 저장하고 세션 삭제로 제거함을 확인했습니다',key='storage',disabled=running)
            st.caption('동의를 거부하거나 철회해도 수업 참여에 불이익이 없습니다.')
    with right:
        with st.container(border=True):
            st.subheader('점검 요약')
            stt_ready,stt_missing,_=stt_readiness(audio_source)
            audio_label='사용 안 함' if not stt_enabled else ('출력 루프백' if audio_source=='loopback' else '마이크')
            model_label='미선택' if not stt_enabled else ('준비됨' if stt_ready else '준비 필요')
            for name,state in [('영상 캡처',('창 선택됨' if selected else '대기') if video_enabled else '사용 안 함'),('수업 음성',audio_label),('전사 모델 (sherpa-onnx)',model_label),('영상 분석 모델 (MediaPipe)',('설치됨' if importlib.util.find_spec('mediapipe') else '미설치') if video_enabled else '사용 안 함'),('참여 모델','사용' if video_enabled and model_enabled else '사용 안 함')]:
                st.write(f'{name} · **{state}**')
            if stt_enabled and not stt_ready:
                st.caption(stt_missing)
            st.caption('설치 여부는 실제 모델 구동 성공 여부와 다릅니다.')
        with st.container(border=True):
            if running:
                st.info('진행 중인 수업이 있습니다.')
                if st.button('라이브 세션으로 이동',use_container_width=True):
                    navigate('라이브 세션')
            elif st.button('세션 시작',type='primary',use_container_width=True,disabled=not((selected or not video_enabled) and title.strip() and consent and storage)):
                try:
                    from frontend.live_session import LiveSession
                    live=LiveSession(selected[0] if selected and video_enabled else None,
                                     audio_source=audio_source if stt_enabled else None,
                                     video_enabled=video_enabled, model_enabled=model_enabled)
                    live.start(title)
                    st.session_state.live=live
                    st.session_state.hidden_suggestions=set()
                    navigate('라이브 세션')
                except Exception as exc:
                    st.error(f'수업을 시작하지 못했습니다: {exc}')
            st.caption('수업 제목과 동의를 확인하세요. 영상 사용 시 수업 창을 선택해야 합니다.')


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
        latest_window=max(windows, key=lambda e: (e['end_ms'], e['created_ms'] or 0)) if windows else None
        model_status=snapshot['status'].get('model')
        value=score(latest_window) if latest_window and model_status != 'unavailable' else None
        b.metric('최근 실험 구간 점수', f'{value:g}점' if value is not None else ('측정 불가' if latest_window or model_status in ('stopped','unavailable') else '집계 대기'))
        result,error,_=live.snapshot()
        if error:
            st.error(f'영상 분석 오류: {error}')
        transcription_error=live.transcription_error()
        if transcription_error:
            st.warning(f'음성 전사 오류: {transcription_error}')
        if result:
            st.caption(f"영상 품질 {result['quality']['status']} · 얼굴 후보 {len(result['features']['faces'])}개 (참가자 수·참여 점수가 아닙니다)")
        if latest_window:
            st.caption(f"구간 {clock(latest_window['start_ms'])}~{clock(latest_window['end_ms'])} · 실험 점수는 집중도나 이해도를 나타내지 않습니다.")
            if model_status == 'unavailable':
                st.caption('모델 또는 영상 입력이 중단되어 최근 점수를 현재 점수로 표시하지 않습니다.')
            elif value is None:
                st.caption('점수 보류 사유: '+', '.join(latest_window['quality']['reason_codes']))
                st.caption('실험 모델은 개발자 미리보기에서 얼굴 한 명이 정면을 바라보며 C 키로 기준을 수집해야 합니다.')
        with st.container(border=True):
            st.subheader('공통 타임라인')
            st.caption('계약에 저장된 engagement_window 구간과 메모를 표시합니다.')
            st.plotly_chart(chart(windows,snapshot['notes'],duration),use_container_width=True,key='live-chart')
            if not windows:
                state=snapshot['status'].get('model')
                st.caption('실험 모델을 사용하지 않거나 실행할 수 없습니다.' if state in ('stopped','unavailable') else '첫 구간을 집계하는 중입니다. 기본 설정에서는 수업 시작 후 10초가 필요합니다.')
        with st.container(border=True):
            st.subheader('강의 조교 · 자료 제안')
            resources(live.service,live.sid,snapshot)
        with st.container(border=True):
            st.subheader('실시간 전사 · 분석 상태')
            state_labels={'running':'실행 중','ready':'준비됨','unavailable':'오류 또는 사용 불가','stopped':'꺼짐 또는 종료'}
            names={'vision':'영상','model':'실험 점수','stt':'음성 전사'}
            st.caption(' · '.join(f"{names.get(key,key)}: {state_labels.get(value,value)}" for key,value in snapshot['status'].items()))
            for component, state in snapshot['status'].items():
                if state == 'unavailable' and snapshot.get('status_messages', {}).get(component):
                    st.caption(f"{names.get(component,component)}: {snapshot['status_messages'][component]}")
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
            if st.button('개발자 모드 끄기' if enabled else '개발자 모드 켜기', disabled=not live.video_enabled):
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
    ordered_windows=sorted(windows,key=lambda e:(e['start_ms'],e['end_ms']))
    has_overlap=any(b['start_ms'] < a['end_ms'] for a,b in zip(ordered_windows,ordered_windows[1:]))
    measured=[(score(e),max(1,e['end_ms']-e['start_ms'])) for e in windows if score(e) is not None] if not has_overlap else []
    average=sum(s*w for s,w in measured)/sum(w for _,w in measured) if measured else None
    ratios=[(e['payload'].get('observation',{}).get('observable_ratio'),max(1,e['end_ms']-e['start_ms'])) for e in windows] if not has_overlap else []
    ratios=[(r,w) for r,w in ratios if r is not None]
    columns=st.columns(4)
    for column,label,value in zip(columns,['수업 전체 참여 점수','분석 가능률','자료 제안','교수자 메모'],[f'{average:.1f}' if average is not None else '측정 불가',f'{sum(r*w for r,w in ratios)/sum(w for _,w in ratios):.0%}' if ratios else '측정 불가',f"{len(snapshot['suggestions'])}건",f"{len(snapshot['notes'])}개"]):
        column.metric(label,value)
    if has_overlap:
        st.caption('이동 구간은 서로 겹치므로 전체 평균과 전체 분석 가능률을 계산하지 않습니다.')
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
