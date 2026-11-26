"""Figma lecturer dashboard: preparation, live session and post-session review."""
import hashlib
import importlib.util
import inspect
import os
import sys
import tempfile
from bisect import bisect_left
from contextlib import nullcontext
from urllib.parse import quote
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from frontend.dashboard_data import (SessionFiles, chart, clock, live_chart_data, live_chart_spec, live_score_display, live_score_unavailable_reason, overlap, review_range_options, score, score_band, suggestion_phase, transcript_score_link,
                                     suggestion_anchors, suggestion_in_range)

st.set_page_config(page_title='교수자 대시보드', page_icon='📊', layout='wide')
files = SessionFiles(os.environ.get('LECTURER_SESSION_DIR', str(Path.home() / 'lecturer_dashboard' / 'sessions')))
for key, value in {'page':'세션 시작 준비','live':None,'review':None,'saved_sid':None,'hidden_suggestions':set(),
                   'assistant':None,'assistant_error':None,'material_cache':{},'material_index':None,
                   'assistant_result':None,'live_manual_suggestion':None,'live_writer_options':{},
                   'live_score_thresholds':{},'notified_suggestions':{},'low_score_notified':{},
                   'pending_manual_notice':None}.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown('''<style>
.stApp{background:#f5f6f7;color:#1f2328} [data-testid="stSidebar"]{background:white;border-right:1px solid #c9ced4}
.block-container{padding:64px 24px 24px;max-width:1440px} h1{font-size:24px!important;padding-top:0!important} h3{font-size:16px!important}
[data-testid="stVerticalBlockBorderWrapper"]>div{background:white;border-radius:8px}
[data-testid="stBaseButton-primary"]{background:#1f2328;border-color:#1f2328;color:white}
[data-testid="stMetric"]{background:white;border:1px solid #c9ced4;border-radius:8px;padding:12px}
[data-testid="stMetricValue"]{font-size:22px} button{min-height:42px} .stCaption{color:#57606a}
/* 관찰 점수 알림만 화면 위쪽 가운데에 빨간색으로 표시한다. */
[data-testid="stToastContainer"]:has([data-testid="stToastText"] strong){left:50%!important;right:auto!important;transform:translateX(-50%);margin:0!important}
[data-testid="stToast"]:has([data-testid="stToastText"] strong){background:#b42318!important;color:#fff!important;filter:none!important;max-width:calc(100vw - 32px)}
[data-testid="stToast"]:has([data-testid="stToastText"] strong) [data-testid="stToastDynamicIcon"],
[data-testid="stToast"]:has([data-testid="stToastText"] strong) button{color:#fff!important}
.st-key-calibration-card button{min-height:36px}
@media(max-width:640px){.block-container{padding:64px 16px 16px}}
/* 화면 제목을 사이드바 열기 버튼과 같은 줄에 배치한다. */
body:has(.st-key-page-title) [data-testid="stHeader"]{background:transparent;pointer-events:none}
body:has(.st-key-page-title) [data-testid="stHeader"] button,
body:has(.st-key-page-title) [data-testid="stHeader"] a{pointer-events:auto}
body:has(.st-key-page-title) [data-testid="stMain"]{container-type:inline-size}
body:has(.st-key-page-title) [data-testid="stMain"] .block-container{padding-top:0}
.st-key-page-title{position:relative;left:min(0px, calc((1440px - 100cqw) / 2));margin-left:48px;margin-right:140px}
/* 사이드바가 열리면 본문의 왼쪽에 제목을 맞춘다. */
body:not(:has([data-testid="stExpandSidebarButton"])) .st-key-page-title{left:0;margin-left:0}
@media(min-width:401px){
  .st-key-page-title h1{min-height:3.75rem;display:flex;align-items:center;padding:0!important;margin:0;transform:translateY(-0.75rem)}
}
/* 간격·글자 높이는 Streamlit의 레이아웃 설정에 맡긴다. CSS로 gap을 덮어쓰지 않는다. */
.st-key-live-overview,.st-key-live-health,
.st-key-live-transcript,.st-key-live-notes,.st-key-live-assistant{background-color:#fff}
/* 메모 입력칸은 포커스 전에도 테두리를 표시하고 안쪽 여백을 줄인다. */
.st-key-live-note-text [data-testid="stChatInput"] > div{border:1px solid #a4acb5;padding:4px 10px}
.st-key-live-note-text [data-testid="stChatInput"] > div:focus-within{border-color:#365f91}
/* 요약 카드 안의 점수에는 중첩 테두리·여백을 더하지 않는다. */
.st-key-live-overview [data-testid="stMetric"]{border:0;padding:0;background:transparent}
.st-key-score-band-red [data-testid="stMetric"], [class*="st-key-review-score-red-"]{background:#ffebee;border-radius:8px;padding:6px 10px}
.st-key-score-band-yellow [data-testid="stMetric"], [class*="st-key-review-score-yellow-"]{background:#fff4cc;border-radius:8px;padding:6px 10px}
.st-key-score-band-green [data-testid="stMetric"], [class*="st-key-review-score-green-"]{background:#e1f3e5;border-radius:8px;padding:6px 10px}
.st-key-score-band-unknown [data-testid="stMetric"], [class*="st-key-review-score-unknown-"]{background:#eef0f2;border-radius:8px;padding:6px 10px}
/* 라이브 제목과 버튼의 가로 배치 */
.st-key-live-page-header [data-testid="stHorizontalBlock"]{align-items:center;gap:8px}
.st-key-live-page-header [data-testid="stHorizontalBlock"] [data-testid="stElementContainer"]{width:auto!important;flex:0 0 auto!important}
.st-key-live-page-header .st-key-page-title{position:static;left:auto;margin:0!important;padding:0;flex:0 0 auto}
.st-key-live-page-header .st-key-page-title h1{width:auto;min-height:42px;margin:0!important;transform:translateY(-2px)}
.st-key-live-page-header [data-testid="stHorizontalBlock"] [data-testid="stElementContainer"]:has(.st-key-page-title),
.st-key-live-page-header [data-testid="stHorizontalBlock"] [data-testid="stElementContainer"]:has([data-testid="stBaseButton-primary"]){width:max-content!important;flex:0 0 auto!important}
.st-key-live-page-header [data-testid="stBaseButton-primary"]{position:relative;z-index:2;flex:0 0 auto;width:auto;margin:0!important;pointer-events:auto}
/* 좁은 화면에서도 카드 열이 한 줄을 유지하며 사용 가능한 폭에 맞춰 줄어든다. */
.st-key-live-dashboard [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]){flex-wrap:nowrap!important}
.st-key-live-dashboard [data-testid="stColumn"]{min-width:0!important;flex:1 1 0!important;width:0!important}
/* 타임라인 카드가 좁으면 제목과 점수를 카드 안에서 위아래로 놓는다. */
.st-key-live-overview{container-type:inline-size}
@container (max-width: 300px){
  .st-key-live-overview [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]){flex-wrap:wrap!important}
  .st-key-live-overview [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{flex:1 1 100%!important;width:100%!important}
  .st-key-live-overview h3{font-size:clamp(11px, 8cqw, 16px)!important;white-space:nowrap}
  .st-key-live-overview [data-testid="stMetricLabel"] p{font-size:clamp(11px, 7cqw, 14px)!important;white-space:normal!important;overflow-wrap:anywhere}
  .st-key-live-overview [data-testid="stMetricValue"]{font-size:clamp(15px, 10cqw, 22px)!important}
}
@media(max-width:400px){
  body:has(.st-key-page-title) [data-testid="stMain"] .block-container{padding-top:64px}
  .st-key-page-title{left:0;margin-left:0;margin-right:0}
}
body:has(.st-key-live-page-header) [data-testid="stMain"] .block-container{padding-top:36px}
</style>''', unsafe_allow_html=True)


def page_title(title):
    with st.container(key='page-title', width='content'):
        st.title(title)


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
    text=event['payload'].get('text', event['payload'].get('description',''))
    if event.get('event_type')=='note_created':
        st.text(text)
    else:
        st.write(text)


def add_note(service, sid, text, at, linked=None):
    event = service.make_event(sid,'note_created',{'text':text,'linked_event_ids':linked or []},at,at,'ui')
    service.receive_event(event)


def stt_readiness(source, model='whisper-small'):
    """선택한 오디오 입력과 faster-whisper 전사에 필요한 패키지를 확인한다. 모델 다운로드 여부는 확인하지 않는다."""
    missing = []
    if importlib.util.find_spec('faster_whisper') is None:
        missing.append('faster-whisper 패키지 (python -m pip install -r faster_whisper_stt/requirements.txt)')
    if source == 'loopback':
        if importlib.util.find_spec('soundcard') is None:
            missing.append('SoundCard 패키지')
    elif importlib.util.find_spec('sounddevice') is None:
        missing.append('sounddevice 패키지')
    from faster_whisper_stt.runner import WhisperConfig
    config = WhisperConfig(model=model.removeprefix('whisper-'))
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


def resources(service, sid, snapshot, readonly=False, *, offset=0, limit=None, show_empty=True, inline_details=False):
    def save_suggestion(title, event_id):
        add_note(service,sid,title,service.elapsed_ms(sid),[event_id])
        st.session_state.hidden_suggestions.add(event_id)

    def hide_suggestion(event_id):
        st.session_state.hidden_suggestions.add(event_id)

    suggestions = [e for e in snapshot['suggestions'] + snapshot.get('teaching_suggestions', [])
                   if readonly or e['event_id'] not in st.session_state.hidden_suggestions]
    suggestions.sort(key=lambda e: (e['start_ms'], e['event_id']), reverse=True)
    suggestions = suggestions[offset:offset + limit if limit is not None else None]
    if not suggestions and show_empty:
        st.caption('선택한 구간에 연결된 제안이 없습니다.' if readonly else '설정한 관찰 점수 이하의 최근 구간과 확정 전사가 있으면 제안을 생성합니다.')
    for event in suggestions:
        payload = event['payload']
        with st.container(border=True):
            st.write(payload['title'])
            st.write(payload['description'])
            model_id = str(event['provenance'].get('model_id') or '')
            by = '강의 조교 · 로컬 검색' if model_id.startswith('local-') else f'강의 조교 · {model_id or "모델 미상"}'
            if event.get('evidence_needs_refresh'):
                st.warning('근거 전사가 수정되었습니다. 자료 제안을 다시 확인하세요.')
            with (nullcontext() if inline_details else st.expander('제안 세부 사항')):
                st.caption(f"{clock(event['start_ms'])} · {suggestion_phase(event)}")
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
                st.caption('관찰 점수는 제안 시점의 참고 신호이며 학습자의 집중도·이해도를 뜻하지 않습니다.')
                ids = set(payload.get('evidence_event_ids', []))
                evidence = [t for t in snapshot['transcripts'] if t['event_id'] in ids]
                for transcript in evidence:
                    st.write(f"{clock(transcript['start_ms'])} · {transcript['payload']['text']}")
            if not readonly:
                a,b = st.columns(2)
                a.button('제안 메모로 저장',key='accept-'+event['event_id'],use_container_width=True,
                         on_click=save_suggestion,args=(payload['title'],event['event_id']))
                b.button('이번 제안 숨기기',key='hide-'+event['event_id'],use_container_width=True,
                         on_click=hide_suggestion,args=(event['event_id'],))


def preparation():
    page_title('세션 시작 준비')
    st.caption('분석을 시작하기 전에 입력과 동의를 확인합니다.')
    live = active()
    running = live is not None and live.is_running()
    left,right = st.columns([2,1],gap='medium')
    with left:
        with st.container(border=True):
            st.subheader('1  영상 분석 설정')
            title = st.text_input('수업 제목',key='lesson_title',disabled=running,placeholder='예: 오픈소스 소프트웨어 8주차')
            video_enabled=st.checkbox('영상 분석 사용',value=True,key='video_enabled',disabled=running)
            model_enabled=st.checkbox('관찰 점수 사용',value=True,key='model_enabled',disabled=running or not video_enabled)
            reidentification_enabled=st.checkbox('세션 내 얼굴 재식별 사용',value=False,
                key='reidentification_enabled',disabled=running or not video_enabled)
            st.caption('점수는 Gaussian 모델입니다. 재식별을 켜면 임시 ID별로, 끄면 단기 위치 트랙별로 계산합니다. 위치 트랙은 사람을 식별하지 않습니다.')
            st.caption('관찰 점수는 머리 방향을 사용합니다. 눈 관측값은 별도로 표시하며 점수에 결합하지 않습니다.')
            if reidentification_enabled and video_enabled:
                st.caption('임시 ID·개인별 점수는 현재 수업 화면에만 표시하고 저장하지 않습니다. 처음 확인될 때 모두 정면을 약 3초 유지해 주세요.')
                if importlib.util.find_spec('insightface') is None:
                    st.warning('재식별 패키지 설치가 필요합니다: python -m pip install -r video_recognition/requirements-reidentification.txt. 준비에 실패하면 개인별 점수는 보류됩니다.')
            windows=st.session_state.get('class_window_list', [])
            window_error=None
            window_audio_target=(st.session_state.get('stt_enabled',False)
                                 and st.session_state.get('stt_source','loopback')=='loopback')
            refresh_windows=st.button('창 목록 새로고침',disabled=running or not (video_enabled or window_audio_target))
            if (video_enabled or window_audio_target) and sys.platform == 'win32' and ('class_window_list' not in st.session_state or refresh_windows):
                try:
                    from video_recognition.integrated_analysis import list_windows
                    windows=list_windows()
                    st.session_state.class_window_list=windows
                except Exception as exc:
                    window_error=str(exc)
            elif (video_enabled or window_audio_target) and sys.platform != 'win32':
                window_error='창 영상·오디오 캡처는 Windows 로컬 실행에서 사용할 수 있습니다.'
            # 창 제목·순서가 바뀌어도 같은 핸들로 선택을 유지한다. 목록은 명시적으로 새로고침한다.
            window_titles=dict(windows)
            if st.session_state.get('class_window_handle') not in window_titles:
                st.session_state.class_window_handle=None
            selected_handle=st.selectbox('분석할 수업 창',list(window_titles),index=None,
                key='class_window_handle',format_func=lambda hwnd: window_titles[hwnd],
                disabled=running or not (video_enabled or window_audio_target),placeholder='수업 창을 선택하세요')
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
            stt_model=st.selectbox('음성 전사 모델 (faster-whisper)', ['whisper-small', 'whisper-base'],
                                   format_func=lambda value: {'whisper-small':'small (권장, 더 정확)',
                                       'whisper-base':'base (더 빠름, 덜 정확)'}[value],
                                   key='stt_model', disabled=running or not stt_enabled)
            if stt_enabled:
                ready,missing,stt_config=stt_readiness(audio_source, stt_model)
                if ready:
                    st.success('전사 실행에 필요한 패키지를 확인했습니다.')
                    st.caption('말이 멈추는 지점(8~20초)마다 한 번씩 전사하므로 문장은 십여 초 늦게 나타납니다. 첫 사용 시 모델 다운로드가 필요할 수 있으며, 결과는 구간 수집과 추론 후 표시됩니다.')
                else:
                    st.warning('전사 시작 전 준비가 필요합니다: '+missing)
                if audio_source=='loopback':
                    st.caption('기본 출력 장치의 전체 소리를 받습니다. 수업 소리와 함께 다른 앱의 소리·알림음도 전사될 수 있습니다.')
                    st.caption('수업 음성이 Windows 기본 출력 장치로 재생되는지 확인하세요. 마이크 입력은 포함하지 않습니다.')
                else:
                    st.caption('기본 마이크 입력을 받습니다. 시스템 출력 소리와 동시 수집은 지원하지 않습니다.')
            else:
                st.caption('선택 입력은 수업 세션의 공통 시계로 전사해 계약 이벤트로 전달합니다. 원본 음성 파일은 저장하지 않습니다.')
        with st.container(border=True):
            st.subheader('3  강의 자료 · 강의 조교 (선택)')
            uploaded=st.file_uploader('PDF · PPTX 강의 자료',type=['pdf','ppt','pptx'],accept_multiple_files=True,key='materials',disabled=running)
            index,errors=material_index(uploaded)
            material_summary(index,errors)
            from ai_agent import lecture_terms
            # 자료에 반복해서 나오는 용어를 faster-whisper 전사 힌트로 넘긴다.
            hint_terms=lecture_terms(index) if index else []
            if hint_terms:
                st.caption('전사 힌트 용어 (처리 시간 증가): '+', '.join(hint_terms))
            writer_options, writer_ready = assistant_options('live-assistant', disabled=running)
            st.caption('외부 AI 사용 시 전사·관련 자료가 제공자에 전달되며, AI가 검색어를 제안하면 한국어 위키백과에도 검색어가 전송됩니다. 검색 결과 제목만 참고하므로 링크 내용을 직접 확인하세요.')
            score_threshold=st.slider('제안·화면 알림 기준 관찰 점수 (이하)', 0, 100, 60, key='assistant_score_threshold', disabled=running)
            st.caption('현재 점수는 관측 가능한 얼굴의 관찰 집계값입니다. 점수가 없거나 오래되면 제안하지 않습니다. 외부 AI 선택 시 자료 없이도 전사 기반 제안을 받을 수 있습니다. '
                       '자료는 현재 브라우저 세션 메모리에만 두며 파일로 저장하지 않습니다. 세션 JSON에는 자료 이름·페이지 번호만 남습니다.')
        with st.container(border=True):
            st.subheader('4  동의·저장 안내')
            consent=st.checkbox('수업 영상·음성 수집과 분석에 동의한 수업입니다',key='consent',disabled=running)
            storage=st.checkbox('전사·메모·집계를 로컬 JSON으로 저장하고 세션 삭제로 제거함을 확인했습니다',key='storage',disabled=running)
            st.caption('동의를 거부하거나 철회해도 수업 참여에 불이익이 없습니다.')
    with right:
        with st.container(border=True):
            st.subheader('점검 요약')
            stt_ready,stt_missing,_=stt_readiness(audio_source, stt_model)
            audio_label='사용 안 함' if not stt_enabled else ('출력 루프백' if audio_source=='loopback' else '마이크')
            model_label='미선택' if not stt_enabled else ('준비됨' if stt_ready else '준비 필요')
            for name,state in [('영상 캡처',('창 선택됨' if selected else '대기') if video_enabled else '사용 안 함'),('수업 음성',audio_label),('전사 모델 ('+stt_model+')',model_label),('영상 분석 모델 (MediaPipe)',('설치됨' if importlib.util.find_spec('mediapipe') else '미설치') if video_enabled else '사용 안 함'),('참여 모델','사용' if video_enabled and model_enabled else '사용 안 함'),
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
                    live=LiveSession(selected[0] if selected else None,
                                     audio_source=audio_source if stt_enabled else None,
                                     stt_model=stt_model, stt_hint_terms=hint_terms,
                                     video_enabled=video_enabled, model_enabled=model_enabled,
                                     reidentification_enabled=reidentification_enabled,
                                     recovery_root=files.root / '.recovery')
                    live.start(title)
                    st.session_state.live=live
                    st.session_state.live_score_thresholds[live.sid]=score_threshold
                    st.session_state.notified_suggestions[live.sid]=set()
                    st.session_state.live_writer_options={live.sid:dict(writer_options)}
                    st.session_state.hidden_suggestions=set()
                    st.session_state.assistant=None
                    st.session_state.assistant_error=None
                    st.session_state.live_manual_suggestion=None
                    st.session_state.pending_manual_notice=None
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
            st.caption('수업 제목과 동의를 확인하세요. 영상 분석을 사용하면 수업 창 선택이 필요합니다.')


def live_diagnostics(live, snapshot, result, diagnostics, rows, latest_window, value, stale, on_calibrate):
    """라이브 카드와 같은 시점의 관측값으로 상세 상태를 그린다."""
    model_status=snapshot['status'].get('model')
    st.caption('관측 가능한 얼굴 기준의 관찰 점수 · 집중도나 이해도가 아닙니다.')
    if value is None:
        reason=live_score_unavailable_reason(model_status, getattr(live, 'model_enabled', False), stale, rows, latest_window,
                                             snapshot.get('status_messages', {}).get('model'))
        st.caption('점수 상태: '+reason)
        age=diagnostics.get('video_result_age_ms')
        head_values=[row.get('head_score') for row in rows
                     if row.get('status') not in ('missing','unavailable','not_observable')
                     and isinstance(row.get('head_score'), (int,float))
                     and not isinstance(row.get('head_score'), bool)]
        if model_status == 'running' and age is not None and age <= 1000 and head_values:
            st.caption(f"참고: 현재 머리 방향 평균 {sum(head_values)/len(head_values):.1f}점 · 저장 구간 점수와 집계 기준이 다릅니다")
    stt_stats=diagnostics['stt']
    if stt_stats:
        dropped=stt_stats['dropped_chunks']
        device_errors=stt_stats['input_overflows'] + stt_stats['discontinuities']
        if dropped or device_errors:
            st.warning(f'음성 입력 누락 가능: 버린 청크 {dropped}개 · 장치 불연속 {device_errors}회')
        st.caption(f"음성 처리 지연 최근 {stt_stats['lag_ms_latest']}ms · 대기 청크 {stt_stats['queue_size']}개 · RTF {stt_stats['rtf']} (처리 시간 / 음성 길이)")
        level,gain=stt_stats.get('input_level_dbfs'),stt_stats.get('gain_db')
        capture_level=stt_stats.get('capture_level_dbfs')
        if stt_stats.get('processed_windows') is not None:
            st.caption(f"10초 전사 처리 {stt_stats['processed_windows']}구간 · 빈 결과 {stt_stats['empty_windows']}구간")
        if capture_level is not None:
            level_hint = ' · 최근 약 2초 최대' if stt_stats.get('processed_windows') is not None else ''
            st.caption(f'최근 오디오 입력 신호 {capture_level:g} dBFS{level_hint}')
            if capture_level < -55:
                source=stt_stats.get('source')
                if source=='loopback':
                    st.warning('기본 출력 장치에서 오디오 신호가 거의 들어오지 않습니다. 재생 중인 앱의 출력 장치와 Windows 기본 출력 장치, 음량을 확인하세요.')
                else:
                    st.warning('마이크 입력 신호가 거의 없습니다. 마이크 선택과 Windows 입력 음량을 확인하세요.')
        if level is not None:
            st.caption(f'말소리 입력 크기 {level:g} dBFS · 자동 음량 조절 +{gain:g} dB')
            if level < -50:
                # 자동 조절 최대치(+30dB)로도 인식이 불안정해지는 범위(실측). 원본 크기를 키우는 것이 근본 해결이다.
                st.warning('말소리가 매우 작아 전사가 빠질 수 있습니다. 마이크를 가까이 하거나 Windows 입력·출력 볼륨을 높이세요.')
            elif level < -35:
                # 실측(노트북 마이크 -34 dBFS): 파열음 최고값 때문에 자동 조절이 +10dB 안팎에서 멈춰 오류가 잦았다.
                st.info('말소리가 작은 편이라 인식 오류가 늘 수 있습니다. Windows 마이크 입력 볼륨을 높이거나 마이크에 가까이 말해 주세요.')
    if diagnostics['video_result_age_ms'] is not None:
        st.caption(f"최근 영상 분석 결과 경과: {diagnostics['video_result_age_ms']}ms")
    if result:
        sampling=result.get('features',{}).get('sampling',{})
        if sampling.get('last_face_interval_ms') is not None:
            st.caption(f"영상 분석 간격 최근 {sampling['last_face_interval_ms']}ms · 300ms 초과 누적 {sampling.get('face_gap_over_300ms',0)}회")
    if live.model_enabled:
        reid = getattr(live, 'reidentification_enabled', False)
        st.subheader('세션 내 ID별 Gaussian 점수' if reid else '단기 위치 트랙별 Gaussian 점수')
        st.caption('결합 점수는 머리 방향 70%, 눈 감김 30%를 반영합니다. 관측이 부족한 요소는 0점으로 처리하지 않습니다.')
        st.caption('일시적으로 한 항목을 놓치면 같은 트랙의 이전 값을 최대 2초 사용하며, 아래에 표시합니다. 상단 점수는 최근 5초 집계입니다.')
        st.caption('참고용 관찰 점수입니다. 표는 수업 중에만 표시되며, 저장·재열기 대상은 ID 없는 그룹 집계입니다.')
        if not reid:
            st.caption('위치 트랙은 사람을 식별하지 않으며, 얼굴이 끊기거나 배치가 바뀌면 보정을 다시 시작할 수 있습니다.')
        st.caption('상단 점수는 유효 관측된 ' + ('사람' if reid else '위치 트랙') + '·시간의 가중 집계이며, 전체 수강생 평균이나 참여율이 아닙니다.')
        if getattr(live, 'reidentification_error', None):
            st.warning(live.reidentification_error)
        if rows:
            states={'scored':'관측 중','deferred':'점수 보류','calibrating':'정면 기준 수집 중',
                    'missing':'관측 불가','not_observable':'관측 불가','unavailable':'관측 불가'}
            st.dataframe([{('임시 ID' if reid else '임시 위치 트랙'):row['id'], '상태':states.get(row['status'],row['status']),
                '정면 보정':{'ready':'완료','collecting':'수집 중','waiting':'대기'}.get(row['calibration_status'],row['calibration_status']),
                '현재 결합 점수':row['current_direction_score'],
                '현재 머리 점수':row.get('current_head_score'), '현재 눈 점수':row.get('current_eye_score'),
                '결합에 사용한 머리':row.get('effective_head_score'), '결합에 사용한 눈':row.get('effective_eye_score'),
                '이전 값 사용':', '.join(f"{name} ({age / 1000:.1f}초 전)" for name, age in row.get('held_component_ages_ms', {}).items()),
                '눈 관측률':row.get('eye_coverage'),
                '눈 상태':row.get('eye_state'), '눈 감김 확인 시간(초)':row.get('eye_closed_seconds'),
                '아래보기 상태':row.get('downward_status'), '아래보기 관측 시간(초)':row.get('looking_down_seconds'),
                '눈 진단':row.get('eye_diagnostics', {}).get('reason'),
                '눈 공백 횟수':row.get('eye_diagnostics', {}).get('gap_count'),
                '눈 누락 횟수':row.get('eye_diagnostics', {}).get('missing_eye_count'),
                '보류 사유':', '.join(row.get('reason_codes', []))}
                for row in rows],hide_index=True,use_container_width=True)
            active_ids=[row['id'] for row in rows if row['status'] not in ('missing','not_observable','unavailable')]
            if active_ids and live.is_running():
                target=st.selectbox('개별 재보정 대상', active_ids, key='calibration-participant')
                st.button('선택한 대상 재보정', key='recalibrate-participant',
                          on_click=on_calibrate, args=(target,))
        else:
            st.caption('재식별 ID 확인 대기 중입니다.' if reid else '안정적인 위치 트랙 확인 대기 중입니다.')
    if result:
        st.caption(f"영상 품질 {result['quality']['status']} · 얼굴 후보 {len(result['features']['faces'])}개 (참가자 수·참여 점수가 아닙니다)")
    if latest_window:
        st.caption(f"구간 {clock(latest_window['start_ms'])}~{clock(latest_window['end_ms'])} · 관찰 점수는 집중도나 이해도를 나타내지 않습니다.")
        if model_status == 'unavailable':
            st.caption('모델 또는 영상 입력이 중단되어 최근 점수를 현재 점수로 표시하지 않습니다.')
        elif value is None:
            st.caption('점수 보류 사유: '+', '.join(latest_window['quality']['reason_codes']))
            st.caption('정면 기준과 관측 상태를 확인하세요. 기준 수집 후 새 점수 구간 집계가 필요합니다.')


def live_page():
    live=active()
    if not live or not live.has_started():
        page_title('라이브 세션')
        st.info('세션 시작 준비에서 수업 창을 선택하고 수업을 시작하세요.')
        if st.button('세션 시작 준비로 이동'):
            navigate('세션 시작 준비')
        return

    # 콜백은 상태만 변경한다. 도구 전환으로 전체 앱을 재실행하지 않는다.
    def toggle_preview():
        try:
            live.set_developer_preview(not live.developer_preview_enabled())
            st.session_state.live_action_error=None
        except (OSError, RuntimeError) as exc:
            st.session_state.live_action_error=f'개발자 창 전환 실패: {exc}'

    def toggle_monitor():
        try:
            monitor=getattr(live,'performance_monitor',None)
            if monitor is not None and monitor.is_running():
                live.stop_performance_monitor()
            else:
                live.start_performance_monitor(ROOT/'frontend'/'diagnostic_logs')
            st.session_state.live_action_error=None
        except (OSError, RuntimeError) as exc:
            st.session_state.live_action_error=f'성능 로그 전환 실패: {exc}'

    def calibrate(identity=None):
        accepted=(live.request_head_calibration() if identity is None else live.request_head_calibration(identity))
        st.session_state.live_action_error=('대상이 현재 관측되지 않아 재보정을 요청하지 못했습니다.'
                                            if accepted is False else None)

    def save_note():
        text=(st.session_state.get('live-note-text') or '').strip()
        if text:
            add_note(live.service,live.sid,text,live.service.elapsed_ms(live.sid))
            if not live.is_running():
                files.save(live.service,live.sid)

    def stop_session():
        try:
            live.stop()
            if st.session_state.assistant is not None:
                st.session_state.assistant.stop()
                if not st.session_state.assistant.is_running():
                    st.session_state.assistant=None
            files.save(live.service,live.sid)
            files.clear_recovery(live.sid)
            st.session_state.saved_sid=live.sid
            st.session_state.review=live.service,live.sid
            navigate('사후 검토')
        except Exception as exc:
            st.session_state.live_action_error=f'수업 종료·저장 실패: {exc}'

    # 카드와 도구를 한 갱신 단위에 고정해 서로 다른 시점의 화면이 섞이지 않게 한다.
    @st.fragment(run_every=1 if live.is_running() else None)
    def panel():
        snapshot=live.service.get_dashboard_snapshot(live.sid)
        duration=snapshot['session']['duration_ms']
        running=live.is_running()
        status='진행 중' if running else '수업 종료'
        windows=snapshot['engagement_windows']
        latest_window=max(windows,key=lambda e:(e['end_ms'],e.get('created_ms') or 0)) if windows else None
        model_status=snapshot['status'].get('model')
        stale=bool(running and latest_window and duration-latest_window['end_ms']>5000)
        value=(score(latest_window) if latest_window and model_status!='unavailable' and not stale
               and not (running and model_status=='stopped') else None)
        display_score=live_score_display(windows,duration,running=running,
            model_enabled=getattr(live,'model_enabled',False),model_status=model_status)
        threshold=st.session_state.live_score_thresholds.get(live.sid,60)
        manual_notice=st.session_state.pop('pending_manual_notice',None)
        if manual_notice:
            st.toast('새 강의 제안: '+manual_notice, icon=':material/notifications:')
        current_value=display_score['value'] if display_score['state']=='current' else None
        low_score=current_value is not None and current_value <= threshold
        low_score_notified=st.session_state.low_score_notified
        if running and low_score and not low_score_notified.get(live.sid,False):
            st.toast(f'**관찰 점수 알림**\n\n설정 기준 {threshold}점 이하입니다. 관측 상태와 수업 흐름을 확인하세요. 이 점수는 집중도를 뜻하지 않습니다.',
                     icon=':material/warning:')
            low_score_notified[live.sid]=True
        elif current_value is not None and not low_score:
            low_score_notified[live.sid]=False
        seen=st.session_state.notified_suggestions.setdefault(live.sid,set())
        new_suggestions=[event for event in snapshot['suggestions']+snapshot.get('teaching_suggestions',[])
                         if event['event_id'] not in seen and event.get('created_ms') is not None]
        if running and new_suggestions:
            for event in new_suggestions:
                st.toast('새 강의 제안: '+event['payload']['title'], icon=':material/notifications:')
                seen.add(event['event_id'])
        previous_windows=[e for e in windows if score(e) is not None]
        previous_window=max(previous_windows,key=lambda e:(e['end_ms'],e.get('created_ms') or 0)) if previous_windows else None
        result,error,_=live.snapshot()
        diagnostics=live.diagnostics()
        stats=diagnostics.get('stt') or {}
        rows=live.participant_snapshot() if hasattr(live,'participant_snapshot') else []
        sampling=result.get('features',{}).get('sampling',{}) if result else {}
        monitor=getattr(live,'performance_monitor',None)
        monitoring=monitor is not None and monitor.is_running()
        preview=live.developer_preview_enabled()
        column_options={'wrap':False} if 'wrap' in inspect.signature(st.columns).parameters else {}
        # DOM gap만 바꾸면 Streamlit이 계산한 요소 높이와 어긋나 텍스트가 겹칠 수 있다.
        card_options={'gap':'xsmall'} if 'gap' in inspect.signature(st.container).parameters else {}
        overview_options={'gap':'xxsmall'} if card_options else {}
        caption_options={'wrap':False} if 'wrap' in inspect.signature(st.caption).parameters else {}

        calibration_enabled=live.video_enabled and getattr(live,'model_enabled',False)
        if calibration_enabled:
            calibration=live.calibration_state()
            state=calibration.get('status','starting')
            calibration_text={'starting':'정면 기준 준비 중','waiting':'정면 기준 대기',
                'recovering':'얼굴 재인식 중 · 점수 보류',
                'collecting':f"정면 기준 수집 중 · {calibration.get('progress',0)*100:.0f}%",
                'ready':'정면 기준 수집 완료'}.get(state,'정면 기준 확인 중')
        else:
            calibration_text='영상·관찰 점수 사용 안 함'
        # 제목과 버튼도 같은 주기 갱신 영역에 두어 보정 상태를 함께 갱신한다.
        with st.container(key='live-page-header',**({'gap':None} if card_options else {})):
            with st.container(horizontal=True, horizontal_alignment='distribute', vertical_alignment='center', gap='xsmall'):
                page_title('라이브 세션')
                with st.container(horizontal=True, horizontal_alignment='right', width='content', gap='xsmall'):
                    if calibration_enabled:
                        st.button('정면 기준 재보정',disabled=not running,key='recalibrate-head',on_click=calibrate,
                                  help='모든 대상의 기준과 새 점수 구간을 다시 수집합니다. 정면을 약 3초 유지해 주세요. 한 명만 문제라면 분석 상세 상태에서 해당 대상만 재보정하세요.')
                    st.button('수업 종료',type='primary',key='live-stop',disabled=not running,on_click=stop_session)
            st.caption(f"{status} · 진행 시간 {clock(duration)} · {calibration_text}",
                       help=snapshot['session']['title'],**caption_options)
        with st.container(key='live-dashboard'):
            left,right=st.columns(2,gap='small',**column_options)
            with left, st.container(height='stretch',border=True,key='live-overview',**overview_options):
                if value is None:
                    reason=live_score_unavailable_reason(model_status,getattr(live,'model_enabled',False),stale,rows,
                                                        latest_window,snapshot.get('status_messages',{}).get('model'))
                    detail='점수 상태: '+reason
                    if display_score['state']=='held':
                        detail+=f" · {clock(display_score['window']['end_ms'])} 기준 이전 점수 임시 표시"
                    elif previous_window is not None:
                        detail+=f" · 이전 점수 {score(previous_window):g}점 ({clock(previous_window['end_ms'])} 기준, 현재 관측 보류)"
                else:
                    detail=f"최근 구간 {clock(latest_window['start_ms'])}–{clock(latest_window['end_ms'])} · 관측 가능한 얼굴의 참고 점수"
                timeline_title,summary=st.columns(2,gap='small',**column_options)
                timeline_title.subheader('공통 타임라인')
                display_value=display_score['value']
                with summary, st.container(key='score-band-'+score_band(current_value)):
                    st.metric('최근 관찰 점수',f'{display_value:g}점' if display_value is not None else
                              ('측정 불가' if latest_window or model_status in ('stopped','unavailable') else '집계 대기'))
                if display_score['state']=='held':
                    summary.caption(f"이전 점수 유지 · {display_score['age_ms']/1000:.1f}초 전")
                st.caption(detail,help=detail,**caption_options)
                chart_range=st.selectbox('그래프 범위',['최근 1분','최근 5분','최근 10분','전체 수업'],index=1,
                                         key='live-chart-range',label_visibility='collapsed')
                range_ms={'최근 1분':60000,'최근 5분':300000,'최근 10분':600000}.get(chart_range)
                start_ms=max(0,duration-range_ms) if range_ms else 0
                chart_spec=live_chart_spec(low_score=low_score); chart_spec['height']=225
                st.vega_lite_chart(live_chart_data(windows,snapshot['notes'],duration,start_ms=start_ms,display_score=display_score),chart_spec,
                                   use_container_width=True,key='live-chart')
                if display_score['state']=='held':
                    st.caption('◆ 이전 점수 임시 유지 · 회색은 관측 보류 구간')
                if not windows:
                    st.caption('첫 점수 구간 집계 대기 중입니다.' if model_status not in ('stopped','unavailable')
                               else '실험 모델이 꺼져 있거나 사용할 수 없습니다.')

            with right, st.container(height='stretch',border=True,key='live-assistant',**card_options):
                st.subheader('강의 조교 · 자료 제안')
                assistant_message=snapshot.get('status_messages',{}).get('assistant')
                manual_assistant=st.session_state.assistant
                manual_result=st.session_state.live_manual_suggestion
                visible=[e for e in snapshot['suggestions']+snapshot.get('teaching_suggestions',[])
                         if e['event_id'] not in st.session_state.hidden_suggestions]
                visible.sort(key=lambda e:(e['start_ms'],e['event_id']),reverse=True)
                if manual_result and not manual_result.get('error'):
                    st.markdown(f"**{manual_result['result']['title']}**")
                    st.write(manual_result['result']['description'])
                elif visible:
                    st.markdown(f"**{visible[0]['payload']['title']}**")
                    st.write(visible[0]['payload']['description'])
                    if visible[0].get('evidence_needs_refresh'):
                        st.warning('근거 전사가 수정되었습니다. 자료 제안을 다시 확인하세요.')
                else:
                    st.caption('표시할 제안이 없습니다.')
                with st.expander('제안 상세'):
                    if assistant_message:
                        st.caption(assistant_message)
                    if manual_result:
                        if manual_result.get('error'):
                            st.warning(manual_result['error'])
                        else:
                            st.caption(f"수동 요청 결과 · {manual_result['model_id']} · 기록에는 저장되지 않음")
                    if st.button('제안 요청',key='live-manual-suggestion',
                             disabled=not running or manual_assistant is None,
                             help='최근 60초 확정 전사와 관련 강의자료로 한 번 요청합니다. 점수 기준과 자동 제안 간격은 적용하지 않습니다.'):
                        now_ms=live.service.elapsed_ms(live.sid)
                        recent=[event for event in snapshot['transcripts']
                            if now_ms-60_000 <= event['end_ms'] <= now_ms]
                        texts=[event['payload']['text'] for event in recent]
                        if not texts:
                            st.session_state.live_manual_suggestion={
                            'error':'최근 60초 안에 확정된 전사가 없어 요청하지 않았습니다.'}
                        else:
                            writer=manual_assistant.writer
                            hits=manual_assistant.index.search(' '.join(texts),limit=3)
                            if writer.uses_external_service and hasattr(writer,'suggest_teaching'):
                                with st.spinner('제안을 요청하는 중…'):
                                    suggestion=writer.suggest_teaching(texts,hits)
                                st.session_state.live_manual_suggestion=(
                                {'result':suggestion,'model_id':writer.model_id} if suggestion else
                                {'error':writer.last_error or 'AI가 제안을 반환하지 않았습니다.'})
                            elif hits:
                                suggestion=writer.suggest(texts,hits)
                                st.session_state.live_manual_suggestion=(
                                {'result':{'title':suggestion['title'],'description':suggestion['description']},
                                 'model_id':writer.model_id} if suggestion else
                                {'error':'관련 자료 제안을 찾지 못했습니다.'})
                            else:
                                st.session_state.live_manual_suggestion={
                                'error':'관련 강의자료가 없어 로컬 제안을 만들 수 없습니다.'}
                        if st.session_state.live_manual_suggestion.get('result'):
                            st.session_state.pending_manual_notice=st.session_state.live_manual_suggestion['result']['title']
                        st.rerun(scope='fragment')
                    if visible:
                        resources(live.service,live.sid,snapshot,show_empty=False,inline_details=True)


            with st.container(height=280,border=True,key='live-notes',**card_options):
                st.subheader('교수자 메모')
                st.chat_input('메모를 입력하세요',key='live-note-text',on_submit=save_note)
                if snapshot['notes']:
                    event=snapshot['notes'][-1]
                    st.caption(f"최근 메모 · {clock(event['start_ms'])}")
                    st.text(event['payload']['text'])
                if len(snapshot['notes'])>1:
                    with st.expander('이전 메모 보기'):
                        for event in reversed(snapshot['notes'][:-1]): event_text(event)

            left,right=st.columns(2,gap='small',**column_options)
            with left, st.container(height=210,border=True,key='live-transcript',**card_options):
                st.subheader('실시간 전사 · 분석 상태')
                if snapshot['transcripts']:
                    st.text('\n'.join(f"[{clock(event['start_ms'])}] {event['payload']['text']}"
                                      for event in reversed(snapshot['transcripts'])))
                for event in snapshot['partial_transcripts'][-1:]: st.caption('인식 중: '+event['payload']['text'])
                if not snapshot['transcripts'] and not snapshot['partial_transcripts']:
                    st.caption('전사 모듈의 입력을 기다립니다.')
            with right, st.container(height=210,border=True,key='live-health',**card_options):
                st.subheader('분석 상태 · 개발자 도구')
                state_labels={'running':'실행 중','ready':'준비됨','unavailable':'확인 필요','stopped':'꺼짐'}
                names={'vision':'영상','model':'점수','stt':'전사','assistant':'조교'}
                st.caption(' · '.join(f"{name} {state_labels.get(snapshot['status'].get(key),'대기')}" for key,name in names.items()))
                lag=stats.get('lag_ms_latest'); hz=sampling.get('actual_face_hz')
                st.caption(f"전사 지연 {lag if lag is not None else '—'}ms · 대기 {stats.get('queue_size','—')}개 · 영상 {hz if hz is not None else '—'}Hz")
                tools,logs=st.columns(2,gap='small',**column_options)
                tools.button('개발자 모드 끄기' if preview else '개발자 모드 켜기',key='live-preview-toggle',
                             disabled=not running or not live.video_enabled,on_click=toggle_preview,use_container_width=True)
                logs.button('성능 로그 수집 중지' if monitoring else '성능 로그 수집 시작',key='live-monitor-toggle',
                            disabled=not running,on_click=toggle_monitor,use_container_width=True)
                errors=[error,live.transcription_error(),live.checkpoint_status(),
                        getattr(live,'reidentification_error',None),st.session_state.assistant_error,
                        st.session_state.get('live_action_error'),getattr(monitor,'error',None)]
                for message in errors:
                    if message:
                        st.warning(message)
                if stats.get('dropped_chunks',0) or stats.get('input_overflows',0) or stats.get('discontinuities',0):
                    st.warning('음성 입력 누락 가능성이 있습니다. 분석 상세 상태에서 확인하세요.')
                if monitor is not None:
                    st.caption(f"로그 {'수집 중' if monitoring else '중지'} · {monitor.path}")

            with st.expander('분석 상세 상태'):
                live_diagnostics(live,snapshot,result,diagnostics,rows,latest_window,value,stale,calibrate)
                st.caption('성능 로그는 5초마다 CPU·메모리·처리 지연과 상태만 기록합니다. 전사 내용, 영상·음성, 얼굴 특징과 점수는 저장하지 않습니다.')
            if not running:
                persist_live()
                if st.button('수업 결과 보기',type='primary'):
                    st.session_state.review=live.service,live.sid
                    navigate('사후 검토')
    panel()


def review_page():
    page_title('사후 검토')
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
    measured=([(score(e),1) for e in ordered_windows if score(e) is not None] if has_overlap else
              [(score(e),max(1,e['end_ms']-e['start_ms'])) for e in windows if score(e) is not None])
    average=sum(s*w for s,w in measured)/sum(w for _,w in measured) if measured else None
    valid_scores=[score(e) for e in windows if score(e) is not None]
    columns=st.columns(4)
    for column,label,value in zip(columns,['롤링 구간 점수 평균' if has_overlap else '수업 전체 관찰 점수','점수 있는 구간','자료·수업 제안','교수자 메모'],[f'{average:.1f}' if average is not None else '점수 없음',f'{len(valid_scores)} / {len(windows)}개',f"{len(snapshot['suggestions']) + len(snapshot.get('teaching_suggestions', []))}건",f"{len(snapshot['notes'])}개"]):
        column.metric(label,value)
    if has_overlap:
        st.caption('겹치는 롤링 구간의 유효 점수를 이벤트마다 한 번씩 평균합니다. 수업 시간 전체의 관측률이나 참여도를 뜻하지 않습니다.')
    if valid_scores:
        st.caption(f'유효 구간 점수 범위 {min(valid_scores):.1f}~{max(valid_scores):.1f}점 · 관찰 점수는 집중도나 이해도를 뜻하지 않습니다.')
    else:
        st.caption('유효한 구간 점수가 없습니다. 타임라인에서 관측 불가 구간을 확인하세요.')
    range_key=f'range-{sid}'
    range_slider_key=f'range-slider-{sid}'
    chart_key=f'review-chart-{sid}'
    options=review_range_options(duration)
    def set_review_range(start, end):
        bounds=tuple(max(0,min(duration,value)) for value in (start,end))
        st.session_state[range_key]=bounds
        def closest(value):
            index=bisect_left(options,value)
            if index == len(options): return options[-1]
            if index and value-options[index-1] <= options[index]-value: return options[index-1]
            return options[index]
        st.session_state[range_slider_key]=tuple(closest(value) for value in bounds)
    def select_slider_range():
        start,end=st.session_state[range_slider_key]
        st.session_state[range_key]=(min(start,duration),min(end,duration))
    def select_chart_range():
        selection=st.session_state.get(chart_key, {}).get('selection', {})
        boxes=selection.get('box', [])
        if boxes and 'x' in boxes[0]:
            bounds=sorted(int(float(x)*1000) for x in boxes[0]['x'])
            set_review_range(*bounds)
    def select_score_window(start, end):
        set_review_range(start,end)
    with st.container(border=True):
        st.subheader('공통 타임라인')
        st.caption('아래에서 구간을 선택하면 전사·메모·자료·피드백이 함께 좁혀집니다.')
        st.select_slider('검토 구간',options=options,value=(0,options[-1]),format_func=clock,
                         key=range_slider_key,on_change=select_slider_range,disabled=duration<1000)
        selected=st.session_state.get(range_key,(0,duration))
        if st.button('선택 해제 · 전체 보기'):
            st.session_state.pop(range_key,None)
            st.session_state.pop(range_slider_key,None)
            st.rerun()
        st.plotly_chart(chart(windows,snapshot['notes'],duration,selected),use_container_width=True,key=chart_key,on_select=select_chart_range,selection_mode='box')
    def visible(event):
        return overlap(event,*selected)
    all_suggestions=snapshot['suggestions'] + snapshot.get('teaching_suggestions', [])
    anchored_suggestions=suggestion_anchors(snapshot['transcripts'],all_suggestions)
    left,right=st.columns([2,1])
    with left:
        with st.container(border=True):
            st.subheader('전사 · 선택 구간')
            query=st.text_input('전사 검색',key=f'search-{sid}')
            transcripts=[e for e in snapshot['transcripts'] if visible(e) and query.casefold() in e['payload']['text'].casefold()]
            st.caption(f'{len(transcripts)}개 문장 · 전사 종료 시점까지 확정된 최근 점수와 연결합니다.')
            st.caption('점수는 해당 문장만의 결과가 아닌 이전 분석 구간의 참고값입니다. 점수와 전사의 관계는 인과관계를 뜻하지 않습니다.')
            if transcripts:
                for event in transcripts:
                    payload=event['payload']
                    linked_score=transcript_score_link(event, windows)
                    score_value=linked_score['value']
                    with st.container(key='review-transcript-'+event['event_id']):
                        score_col,transcript_col=st.columns([1,3],gap='medium',vertical_alignment='top')
                        with score_col, st.container(key='review-score-'+score_band(score_value)+'-'+event['event_id']):
                            st.caption(f"{clock(event['start_ms'])}–{clock(event['end_ms'])}")
                            st.markdown(f"### {score_value:.1f}점" if score_value is not None else f"**{linked_score['reason']}**")
                            window=linked_score['window']
                            if linked_score['delta'] is not None:
                                st.caption(f"직전 대비 {linked_score['delta']:+.1f}점")
                            elif score_value is not None:
                                st.caption('직전 비교 없음')
                            if window:
                                quality=window.get('quality',{}).get('status')
                                st.caption({'valid':'관측 가능','partial':'일부 관측',
                                            'not_observable':'관측 불가'}.get(quality,'품질 확인 필요'))
                                reasons=window.get('quality',{}).get('reason_codes',[])
                                if score_value is None and reasons:
                                    st.caption('사유: '+', '.join(reasons))
                                st.button('점수 구간 보기',key='score-window-'+event['event_id'],
                                          on_click=select_score_window,args=(window['start_ms'],window['end_ms']),
                                          width='stretch')
                        with transcript_col:
                            st.write(payload['text'])
                            suggestions=anchored_suggestions.get(event['event_id'],[])
                            if suggestions:
                                st.markdown('**연결 제안**')
                                for suggestion in suggestions:
                                    suggestion_payload=suggestion['payload']
                                    st.markdown(
                                        f"**{suggestion_phase(suggestion)} · {suggestion_payload['title']}**  \n"
                                        f"{suggestion_payload['description']}"
                                    )
                            with st.expander('전사 수정'):
                                with st.form('edit-'+event['event_id']):
                                    text=st.text_area('수정할 전사',payload['text'])
                                    if st.form_submit_button('수정 저장'):
                                        try:
                                            updated=dict(payload,text=text,revision=payload['revision']+1,is_edited=True,replaces_event_id=event['event_id'])
                                            change=service.make_event(sid,'transcript_edited',updated,event['start_ms'],event['end_ms'],'ui')
                                            service.receive_event(change)
                                            files.save(service,sid)
                                            st.rerun()
                                        except Exception as exc:
                                            st.error(f'전사를 수정하지 못했습니다: {exc}')
                    st.write('')
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
            st.caption('확정 전사로 구간별 피드백을 만들고, 낮은 관찰 점수 또는 교수자 메모가 있는 5분 구간에 제안을 추가합니다. 전사가 없거나 기존 제안이 있는 구간은 새 제안을 만들지 않습니다.')
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
                st.caption('자료가 없으면 전사·메모·관찰 점수만으로 피드백을 만듭니다.')
            saved_options=st.session_state.live_writer_options.get(sid)
            use_live_options=(st.checkbox('라이브 수업의 AI 설정 사용',value=True,key=f'review-use-live-ai-{sid}',
                help='이 브라우저 세션에 보관된 제공자·모델·키를 사용합니다. 키는 세션 JSON에 저장하지 않습니다.')
                if saved_options and saved_options.get('provider') != 'local' else False)
            if use_live_options:
                review_options,review_ready=saved_options,True
                st.caption(f"라이브 설정 사용 · {saved_options['provider']} / {saved_options['model']} · 이 브라우저 세션에서만 유지")
            else:
                review_options, review_ready=assistant_options(f'review-assistant-{sid}')
            review_threshold=st.slider('사후 제안 관찰 점수 기준 (이하)', 0, 100, 60, key=f'review-threshold-{sid}')
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
        running_same_session = active() and active().sid == sid and active().is_running()
        if running_same_session:
            st.caption('진행 중인 수업은 라이브 세션에서 종료한 뒤 삭제하세요.')
        confirmed=st.checkbox('이 수업의 전사·메모·집계를 삭제합니다',key='delete-'+sid)
        if st.button('수업 삭제',disabled=not confirmed or bool(running_same_session)):
            try:
                files.delete(sid)
                if active() and active().sid == sid:
                    st.session_state.live=None
                st.session_state.review=None
                st.rerun()
            except OSError as exc:
                st.error(f'삭제 실패: {exc}')


{'세션 시작 준비':preparation,'라이브 세션':live_page,'사후 검토':review_page}[st.session_state.page]()
