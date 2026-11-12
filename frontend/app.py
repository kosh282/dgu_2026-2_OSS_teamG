"""Figma lecturer dashboard: preparation, live session and post-session review."""
import hashlib
import importlib.util
import os
import sys
import tempfile
from urllib.parse import quote
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from frontend.dashboard_data import (SessionFiles, chart, clock, live_chart_data, live_chart_spec, overlap, score, suggestion_phase,
                                     related_suggestions, suggestion_in_range)

st.set_page_config(page_title='교수자 대시보드', page_icon='📊', layout='wide')
files = SessionFiles(os.environ.get('LECTURER_SESSION_DIR', str(Path.home() / 'lecturer_dashboard' / 'sessions')))
for key, value in {'page':'세션 시작 준비','live':None,'review':None,'saved_sid':None,'hidden_suggestions':set(),
                   'assistant':None,'assistant_error':None,'material_cache':{},'material_index':None,
                   'assistant_result':None}.items():
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
            files.clear_recovery(live.sid)
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
    for recovery_sid, recovery in files.recoveries():
        if active() and active().is_running() and active().sid == recovery_sid:
            continue
        if st.button(f"복구: {recovery['title']}", key=f'recover-{recovery_sid}', use_container_width=True):
            try:
                service, sid = files.load(files.recovery_path(recovery_sid))
                if files.path(sid).exists():
                    st.warning('같은 ID의 저장 기록이 있어 복구 파일을 유지합니다. 두 파일을 비교해 주세요.')
                else:
                    files.save(service, sid)
                    files.clear_recovery(sid)
                    st.session_state.review = service, sid
                    navigate('사후 검토')
            except Exception as exc:
                st.error(f'수업 복구 실패: {exc}')
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


def material_index(uploaded):
    """업로드한 강의 자료를 페이지 단위로 읽어 검색 색인을 만든다. 자료는 브라우저 세션 메모리에만 둔다."""
    from ai_agent import MaterialError, MaterialIndex, load_material
    loaded, errors = [], []
    cache = st.session_state.material_cache
    for file in uploaded or []:
        data = file.getvalue()
        key = (file.name, hashlib.sha256(data).hexdigest())
        if key not in cache:
            try:
                cache[key] = load_material(file.name, data)
            except MaterialError as exc:
                cache[key] = exc
        item = cache[key]
        if isinstance(item, Exception):
            errors.append(f'{file.name}: {item}')
        else:
            loaded.append(item)
    return MaterialIndex(loaded), errors


def material_summary(index, errors):
    for error in errors:
        st.warning(error)
    if index.materials:
        empty = sum(m.empty_pages for m in index.materials)
        st.caption(f'자료 {len(index.materials)}개 · {index.page_count}쪽을 읽었습니다.'
                   + (f' 텍스트가 없는 {empty}쪽(스캔 이미지 등)은 검색하지 않습니다.' if empty else ''))


def assistant_options(key, disabled=False):
    """제공자별 키는 화면 세션과 실행 중 작성기에만 둔다."""
    from ai_agent.writer import PROVIDER_DEFAULT_MODELS
    provider = st.selectbox('문장 작성 방식', ['local', 'openai', 'gemini', 'claude'],
                            format_func=lambda x: {'local':'로컬', 'openai':'OpenAI', 'gemini':'Gemini', 'claude':'Claude'}[x],
                            key=f'{key}-provider', disabled=disabled)
    if provider == 'local':
        return {'provider': 'local'}, True
    model = st.text_input('모델 ID', value=PROVIDER_DEFAULT_MODELS[provider],
                          key=f'{key}-{provider}-model', disabled=disabled)
    api_key = st.text_input('개인 API 키', type='password', key=f'{key}-{provider}-key', disabled=disabled)
    st.caption('확정 전사·메모·관련 자료 텍스트가 선택한 제공자에 전송됩니다. API 키는 세션 JSON에 저장하지 않으며 사용량은 해당 계정에 청구될 수 있습니다.')
    if not api_key.strip() or not model.strip():
        st.warning('선택한 제공자의 API 키와 모델 ID를 입력하세요.')
    if st.button('API 연결 확인 (요청 발생)', key=f'{key}-{provider}-check', disabled=disabled or not api_key.strip() or not model.strip()):
        from ai_agent.writer import make_writer, ClaudeUnavailable
        try:
            make_writer(provider=provider, api_key=api_key, model=model).check_connection()
            st.success('API와 모델의 구조화 응답을 확인했습니다.')
        except (ValueError, ClaudeUnavailable) as exc:
            st.error(f'연결 확인 실패: {exc}')
    return {'provider': provider, 'api_key': api_key, 'model': model}, bool(api_key.strip() and model.strip())


def resources(service, sid, snapshot, readonly=False):
    suggestions = [e for e in snapshot['suggestions'] + snapshot.get('teaching_suggestions', [])
                   if readonly or e['event_id'] not in st.session_state.hidden_suggestions]
    suggestions.sort(key=lambda e: (e['start_ms'], e['event_id']), reverse=True)
    if not suggestions:
        st.caption('선택한 구간에 연결된 제안이 없습니다.' if readonly else '설정한 실험 점수 이하의 최근 구간과 확정 전사가 있으면 제안을 생성합니다.')
    for event in suggestions:
        payload = event['payload']
        with st.container(border=True):
            st.caption(f"{clock(event['start_ms'])} · {suggestion_phase(event)}")
            st.write(payload['title'])
            st.write(payload['description'])
            model_id = str(event['provenance'].get('model_id') or '')
            by = '강의 조교 · 로컬 검색' if model_id.startswith('local-') else f'강의 조교 · {model_id or "모델 미상"}'
            if event['event_type'] == 'teaching_suggestion':
                st.caption(f'{by} · 전사 기반 AI 생성 초안 · 출처 내용과 적합성은 교수자가 확인해야 합니다.')
                for source in payload.get('material_sources', []):
                    st.caption(f"참고한 업로드 자료 후보: {source['document_name']} · {source.get('page') or '?'}쪽")
                for source in payload.get('external_sources', []):
                    st.link_button('외부 검색 결과: ' + source['title'], source['url'])
                if payload.get('search_query'):
                    st.link_button('검색어로 외부 자료 찾기', 'https://www.google.com/search?q=' + quote(payload['search_query']))
            else:
                source = payload['source']
                st.caption(f"출처: {source['document_name']} · {source.get('page') or '페이지 미지정'}쪽 · {by}")
            st.caption('실험 점수는 제안 시점의 참고 신호이며 학습자의 집중도·이해도를 뜻하지 않습니다.')
            if event.get('evidence_needs_refresh'):
                st.warning('근거 전사가 수정되었습니다. 자료 제안을 다시 확인하세요.')
            if readonly:
                ids = set(payload.get('evidence_event_ids', []))
                evidence = [t for t in snapshot['transcripts'] if t['event_id'] in ids]
                if evidence:
                    with st.expander(f'근거 강의록 {len(evidence)}개 보기'):
                        for transcript in evidence:
                            st.write(f"{clock(transcript['start_ms'])} · {transcript['payload']['text']}")
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
            windows=st.session_state.get('class_window_list', [])
            window_error=None
            refresh_windows=st.button('창 목록 새로고침',disabled=running or not video_enabled)
            if video_enabled and sys.platform == 'win32' and ('class_window_list' not in st.session_state or refresh_windows):
                try:
                    from video_recognition.integrated_analysis import list_windows
                    windows=list_windows()
                    st.session_state.class_window_list=windows
                except Exception as exc:
                    window_error=str(exc)
            elif video_enabled and sys.platform != 'win32':
                window_error='실시간 창 캡처는 Windows 로컬 실행에서 사용할 수 있습니다.'
            # 창 제목·순서가 바뀌어도 같은 핸들로 선택을 유지한다. 목록은 명시적으로 새로고침한다.
            window_titles=dict(windows)
            if st.session_state.get('class_window_handle') not in window_titles:
                st.session_state.class_window_handle=None
            selected_handle=st.selectbox('분석할 수업 창',list(window_titles),index=None,
                key='class_window_handle',format_func=lambda hwnd: window_titles[hwnd],
                disabled=running or not video_enabled,placeholder='수업 창을 선택하세요')
            selected=(selected_handle, window_titles[selected_handle]) if selected_handle is not None else None
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
            st.subheader('3  강의 자료 · 강의 조교 (선택)')
            uploaded=st.file_uploader('PDF · PPTX 강의 자료',type=['pdf','ppt','pptx'],accept_multiple_files=True,key='materials',disabled=running)
            index,errors=material_index(uploaded)
            material_summary(index,errors)
            writer_options, writer_ready = assistant_options('live-assistant', disabled=running)
            st.caption('외부 AI 사용 시 전사·관련 자료가 제공자에 전달되며, AI가 검색어를 제안하면 한국어 위키백과에도 검색어가 전송됩니다. 검색 결과 제목만 참고하므로 링크 내용을 직접 확인하세요.')
            score_threshold=st.slider('제안 시작 실험 점수 (이하)', 0, 100, 60, key='assistant_score_threshold', disabled=running)
            st.caption('현재 점수는 단일 얼굴 실험값입니다. 점수가 없거나 오래되면 제안하지 않습니다. 외부 AI 선택 시 자료 없이도 전사 기반 제안을 받을 수 있습니다. '
                       '자료는 현재 브라우저 세션 메모리에만 두며 파일로 저장하지 않습니다. 세션 JSON에는 자료 이름·페이지 번호만 남습니다.')
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
            for name,state in [('영상 캡처',('창 선택됨' if selected else '대기') if video_enabled else '사용 안 함'),('수업 음성',audio_label),('전사 모델 (sherpa-onnx)',model_label),('영상 분석 모델 (MediaPipe)',('설치됨' if importlib.util.find_spec('mediapipe') else '미설치') if video_enabled else '사용 안 함'),('참여 모델','사용' if video_enabled and model_enabled else '사용 안 함'),
                               ('강의 조교',f'자료 {len(index.materials)}개 · 점수 {score_threshold} 이하 · {writer_options["provider"]}')]:
                st.write(f'{name} · **{state}**')
            if stt_enabled and not stt_ready:
                st.caption(stt_missing)
            st.caption('설치 여부는 실제 모델 구동 성공 여부와 다릅니다.')
        with st.container(border=True):
            if running:
                st.info('진행 중인 수업이 있습니다.')
                if st.button('라이브 세션으로 이동',use_container_width=True):
                    navigate('라이브 세션')
            elif st.button('세션 시작',type='primary',use_container_width=True,disabled=not((selected or not video_enabled) and title.strip() and consent and storage and writer_ready)):
                try:
                    from frontend.live_session import LiveSession
                    live=LiveSession(selected[0] if selected and video_enabled else None,
                                     audio_source=audio_source if stt_enabled else None,
                                     video_enabled=video_enabled, model_enabled=model_enabled,
                                     recovery_root=files.root / '.recovery')
                    live.start(title)
                    st.session_state.live=live
                    st.session_state.hidden_suggestions=set()
                    st.session_state.assistant=None
                    st.session_state.assistant_error=None
                    # 업로더 상태는 페이지를 옮기면 사라지므로 사후 검토에서 쓸 색인을 따로 보관한다(메모리 전용).
                    st.session_state.material_index=index
                    try:
                        # 강의 조교는 같은 통합부 세션의 확정 전사를 읽어 자료 제안 이벤트를 보낸다.
                        from ai_agent import LiveAssistant, make_writer
                        assistant=LiveAssistant(live.service,live.sid,index,make_writer(**writer_options),score_threshold=score_threshold)
                        assistant.start()
                        st.session_state.assistant=assistant
                    except Exception as exc:  # 조교 실패가 수업 시작을 막지 않는다
                        st.session_state.assistant_error=f'강의 조교를 시작하지 못했습니다: {exc}'
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
        status='진행 중' if live.is_running() else '수업 종료'
        st.caption(f"{snapshot['session']['title']} · {status} · 진행 시간 {clock(duration)}")
        a,b=st.columns(2)
        with a:
            with st.container(border=True):
                st.subheader('강의 조교 · 자료 제안')
                assistant_message=snapshot.get('status_messages',{}).get('assistant')
                if st.session_state.assistant_error:
                    st.warning(st.session_state.assistant_error)
                elif assistant_message:
                    st.caption(assistant_message)
                resources(live.service,live.sid,snapshot)
        windows=snapshot['engagement_windows']
        latest_window=max(windows, key=lambda e: (e['end_ms'], e['created_ms'] or 0)) if windows else None
        model_status=snapshot['status'].get('model')
        value=score(latest_window) if latest_window and model_status != 'unavailable' else None
        b.metric('최근 실험 구간 점수', f'{value:g}점' if value is not None else ('측정 불가' if latest_window or model_status in ('stopped','unavailable') else '집계 대기'))
        if live.video_enabled and getattr(live, 'model_enabled', False):
            calibration=live.calibration_state()
            state=calibration.get('status', 'starting')
            reason=calibration.get('last_reason_code')
            reasons={'no_face_detected':'얼굴이 감지되지 않았습니다.', 'multiple_faces_detected':'얼굴이 여러 개 감지되어 단일 얼굴 보정을 할 수 없습니다.',
                     'face_quality_not_observable':'얼굴을 충분히 관찰할 수 없습니다.', 'head_angles_unavailable':'머리 각도를 계산할 수 없습니다.',
                     'head_pose_unstable':'머리 움직임이 커서 다시 수집합니다.', 'calibration_frame_gap':'영상 입력이 끊겨 다시 수집합니다.',
                     'calibration_timestamp_not_increasing':'영상 시각이 정상적으로 진행되지 않았습니다.'}
            b.caption({'starting':'정면 기준: 영상 분석 준비 중', 'waiting':'정면 기준: 재보정이 필요합니다',
                       'collecting':f"정면 기준 수집 중 · {calibration.get('progress', 0)*100:.0f}%", 'ready':'정면 기준 수집 완료'}.get(state, '정면 기준: 상태 확인 중'))
            if reason:
                b.caption(reasons.get(reason, f'보정 상태: {reason}'))
            b.caption('얼굴 한 명이 정면을 보고 약 3초간 유지해 주세요. 자동 기준은 실제 정면인지 확인하지 못합니다.')
            if b.button('정면 기준 재보정', disabled=not live.is_running(), key='recalibrate-head'):
                live.request_head_calibration()
                st.rerun()
        _,error,_=live.snapshot()
        if error:
            st.error(f'영상 분석 오류: {error}')
        transcription_error=live.transcription_error()
        if transcription_error:
            st.warning(f'음성 전사 오류: {transcription_error}')
        checkpoint_error=live.checkpoint_status()
        if checkpoint_error:
            st.warning(f'수업 복구용 저장 실패: {checkpoint_error}')
        with st.container(border=True):
            st.subheader('공통 타임라인')
            st.caption('계약에 저장된 engagement_window 구간과 메모를 표시합니다.')
            st.vega_lite_chart(live_chart_data(windows,snapshot['notes'],duration), live_chart_spec(),
                               use_container_width=True, key='live-chart')
            if not windows:
                state=snapshot['status'].get('model')
                st.caption('실험 모델을 사용하지 않거나 실행할 수 없습니다.' if state in ('stopped','unavailable') else '첫 구간을 집계하는 중입니다. 기본 설정에서는 수업 시작 후 10초가 필요합니다.')
        with st.container(border=True):
            st.subheader('실시간 전사 · 분석 상태')
            state_labels={'running':'실행 중','ready':'준비됨','unavailable':'오류 또는 사용 불가','stopped':'꺼짐 또는 종료'}
            names={'vision':'영상','model':'실험 점수','stt':'음성 전사','assistant':'강의 조교'}
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
                if st.session_state.assistant is not None:
                    st.session_state.assistant.stop()
                    if not st.session_state.assistant.is_running():
                        st.session_state.assistant = None
                files.save(live.service,live.sid)
                files.clear_recovery(live.sid)
                st.session_state.saved_sid=live.sid
                st.session_state.review=live.service,live.sid
                navigate('사후 검토')
            except Exception as exc:
                st.error(f'수업 종료·저장 실패: {exc}')

    @st.fragment(run_every=1 if live.is_running() else None)
    def diagnostics_panel():
        snapshot=live.service.get_dashboard_snapshot(live.sid)
        windows=snapshot['engagement_windows']
        latest_window=max(windows, key=lambda e: (e['end_ms'], e['created_ms'] or 0)) if windows else None
        model_status=snapshot['status'].get('model')
        value=score(latest_window) if latest_window and model_status != 'unavailable' else None
        result,_,_=live.snapshot()
        diagnostics=live.diagnostics()
        stt_stats=diagnostics['stt']
        if stt_stats:
            dropped=stt_stats['dropped_chunks']
            device_errors=stt_stats['input_overflows'] + stt_stats['discontinuities']
            if dropped or device_errors:
                st.warning(f'음성 입력 누락 가능: 버린 청크 {dropped}개 · 장치 불연속 {device_errors}회')
            st.caption(f"음성 처리 지연 최근 {stt_stats['lag_ms_latest']}ms · 대기 청크 {stt_stats['queue_size']}개 · RTF {stt_stats['rtf']} (처리 시간 / 음성 길이)")
        if diagnostics['video_result_age_ms'] is not None:
            st.caption(f"최근 영상 분석 결과 경과: {diagnostics['video_result_age_ms']}ms")
        if result:
            st.caption(f"영상 품질 {result['quality']['status']} · 얼굴 후보 {len(result['features']['faces'])}개 (참가자 수·참여 점수가 아닙니다)")
        if latest_window:
            st.caption(f"구간 {clock(latest_window['start_ms'])}~{clock(latest_window['end_ms'])} · 실험 점수는 집중도나 이해도를 나타내지 않습니다.")
            if model_status == 'unavailable':
                st.caption('모델 또는 영상 입력이 중단되어 최근 점수를 현재 점수로 표시하지 않습니다.')
            elif value is None:
                st.caption('점수 보류 사유: '+', '.join(latest_window['quality']['reason_codes']))
                st.caption('정면 기준과 관측 상태를 확인하세요. 기준 수집 후 새 점수 구간 집계가 필요합니다.')
    diagnostics_panel()
    if live.is_running():
        with st.expander('개발자 도구'):
            enabled=live.developer_preview_enabled()
            if st.button('개발자 모드 끄기' if enabled else '개발자 모드 켜기', disabled=not live.video_enabled):
                live.set_developer_preview(not enabled)
                st.rerun()
            st.markdown('**성능 진단 로그**')
            st.caption('5초마다 CPU·메모리, 영상 처리 주기, 전사 처리 지연과 상태만 로컬에 기록합니다. 전사 내용, 영상·음성, 얼굴 특징, 점수는 저장하지 않습니다.')
            monitor=getattr(live, 'performance_monitor', None)
            if monitor is not None:
                st.caption(f'로그 위치: {monitor.path}')
                if monitor.error:
                    st.warning(monitor.error)
            if monitor is not None and monitor.is_running():
                if st.button('성능 로그 수집 중지'):
                    live.stop_performance_monitor()
                    st.rerun()
            elif st.button('성능 로그 수집 시작'):
                try:
                    live.start_performance_monitor(ROOT/'frontend'/'diagnostic_logs')
                    st.rerun()
                except (OSError, RuntimeError) as exc:
                    st.error(f'성능 로그를 시작하지 못했습니다: {exc}')


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
    for column,label,value in zip(columns,['수업 전체 참여 점수','분석 가능률','자료·수업 제안','교수자 메모'],[f'{average:.1f}' if average is not None else '측정 불가',f'{sum(r*w for r,w in ratios)/sum(w for _,w in ratios):.0%}' if ratios else '측정 불가',f"{len(snapshot['suggestions']) + len(snapshot.get('teaching_suggestions', []))}건",f"{len(snapshot['notes'])}개"]):
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
    all_suggestions=snapshot['suggestions'] + snapshot.get('teaching_suggestions', [])
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
                    linked=related_suggestions(event, all_suggestions)
                    for suggestion in linked:
                        st.caption(f"{suggestion_phase(suggestion)} · {clock(suggestion['start_ms'])} · {suggestion['payload']['title']}")
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
            st.subheader('AI 강의 조교 · 사후 분석')
            st.caption('확정 전사로 구간별 피드백을 만들고, 낮은 실험 점수 또는 교수자 메모가 있는 5분 구간에 제안을 추가합니다. 전사가 없거나 기존 제안이 있는 구간은 새 제안을 만들지 않습니다.')
            st.caption('이미 조교 사후 피드백을 생성한 수업은 중복 방지를 위해 다시 생성하지 않습니다. 점수는 집중도나 이해도를 의미하지 않습니다.')
            extra=st.file_uploader('강의 자료 추가 (수업 시작 때 올린 자료는 다시 올리지 않아도 됩니다)',type=['pdf','ppt','pptx'],
                                   accept_multiple_files=True,key=f'review-materials-{sid}')
            from ai_agent import MaterialIndex, generate_post_session, make_writer
            extra_index,errors=material_index(extra)
            known={m.document_id:m for m in (st.session_state.material_index.materials if st.session_state.material_index else [])}
            known.update({m.document_id:m for m in extra_index.materials})
            review_index=MaterialIndex(known.values())
            material_summary(review_index,errors)
            if not review_index.materials:
                st.caption('자료가 없으면 전사·메모·실험 점수만으로 피드백을 만듭니다.')
            review_options, review_ready=assistant_options(f'review-assistant-{sid}')
            review_threshold=st.slider('사후 제안 실험 점수 기준 (이하)', 0, 100, 60, key=f'review-threshold-{sid}')
            include_selected=st.checkbox('현재 검토 구간에도 제안 생성', key=f'review-selected-{sid}')
            if include_selected:
                st.caption(f'선택 구간 {clock(selected[0])}–{clock(selected[1])}과 겹치는 5분 구간을 추가합니다. 전체 구간이 선택되어 있으면 전 구간이 대상입니다.')
            st.caption('외부 AI 사용 시 전사·관련 자료가 제공자에 전달되며 생성된 검색어는 한국어 위키백과 검색에 사용될 수 있습니다.')
            if st.button('사후 피드백 · 자료 제안 생성',key=f'assistant-run-{sid}',type='primary',disabled=not review_ready):
                with st.spinner('강의 조교가 수업 기록을 정리하는 중입니다...'):
                    try:
                        result=generate_post_session(service,sid,review_index,make_writer(**review_options),
                            score_threshold=review_threshold, selected_range=selected if include_selected else None)
                        files.save(service,sid)
                        st.session_state.assistant_result=(sid,result)
                    except Exception as exc:
                        st.session_state.assistant_result=(sid,{'error':str(exc)})
                st.rerun()
            outcome=st.session_state.assistant_result
            if outcome and outcome[0]==sid:
                result=outcome[1]
                if result.get('error'):
                    st.error(f"사후 분석 실패: {result['error']}")
                elif result.get('skipped'):
                    st.info(result['skipped'])
                else:
                    st.success(f"피드백 {result['feedback']}건 · 자료 제안 {result['suggestions']}건을 만들었습니다.")
                    if result.get('writer_error'):
                        st.warning(f"외부 AI 작성 실패로 로컬 결과를 사용했습니다: {result['writer_error']}")
        with st.container(border=True):
            st.subheader('사후 피드백 · 선택 구간')
            feedback=[e for e in snapshot['feedback'] if visible(e)]
            for event in feedback:
                st.caption(f"{clock(event['start_ms'])}–{clock(event['end_ms'])} · {'초안' if event['payload']['status']=='draft' else '검토 완료'}")
                st.write(event['payload']['text'])
                if event.get('evidence_needs_refresh'):
                    st.warning('근거 전사가 수정되어 피드백 재검토가 필요합니다.')
            if not feedback:
                st.caption('저장된 피드백이 없습니다. 위의 AI 강의 조교로 만들 수 있습니다.')
            st.caption('관찰 신호만으로 이해 부족이나 교수법의 문제를 확정하지 않습니다.')
        with st.container(border=True):
            st.subheader('수업 중·사후 제안 · 선택 구간')
            st.caption('제안 시각이나 근거 강의록이 선택 구간과 겹치는 항목입니다. 수업 중 숨긴 제안도 강의 기록에 표시합니다.')
            in_range=lambda e: suggestion_in_range(e,*selected,snapshot['transcripts'])
            filtered=dict(snapshot,suggestions=[e for e in snapshot['suggestions'] if in_range(e)],
                          teaching_suggestions=[e for e in snapshot.get('teaching_suggestions', []) if in_range(e)])
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
