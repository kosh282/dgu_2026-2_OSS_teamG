"""Streamlit lecturer dashboard prototype based on the team's Figma live-session panel.
Run: streamlit run app.py
"""
import json
import time
import sys
import tempfile
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frontend.live_session import LiveSession
from video_recognition.integrated_analysis import list_windows

st.set_page_config(page_title="교수자 라이브 세션", page_icon="📊", layout="centered", initial_sidebar_state="collapsed")

SAMPLE_SCORES = [76, 74, 70, 67, 71, 68, 64, 60, 58, 61, 65, 63, 57, 55, 52, None, None, None, 49, 46, 51, 47, 44, 41, 38]
SIGNALS = {
    "얼굴 방향": [(0, 12), (17, 29), (40, 59)],
    "머리 이동": [(0, 7), (20, 28), (38, 50)],
    "표정": [(3, 18), (35, 45)],
    "상체 움직임": [(0, 18), (41, 55)],
}


def initialize():
    defaults = {
        "title": "[수업 제목]", "elapsed": 58 * 60 + 30, "started": False,
        "last_tick": time.time(), "scores": SAMPLE_SCORES.copy(),
        "notes": [{"time_ms": 52 * 60 * 1000 + 40000, "text": "(메모 내용 예시 2)"},
                  {"time_ms": 32 * 60 * 1000 + 5000, "text": "(메모 내용 예시 1)"}],
        "suggestions": [{"time_ms": 58 * 60 * 1000 + 12000,
                         "title": "[강의 자료명] p.[00] · [설명 구간]",
                         "evidence": "(확정된 전사 문장 예시)", "source": "[업로드한 강의 자료]"}],
        "events": [], "source": "예시 데이터", "dismissed": False,
        "transcripts": [], "component_status": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def clock(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02}:{(seconds // 60) % 60:02}:{seconds % 60:02}"


def event_time(ms):
    return clock(ms / 1000)


def status(score):
    if score is None:
        return "분석 보류", "#57606a", "#edf0f2", "#a1aab4", "⚪"
    if score >= 70:
        return "좋음", "#116b43", "#e6f5ed", "#39a56d", "🟢"
    if score >= 40:
        return "보통", "#915b0b", "#fff4db", "#d59a31", "🟡"
    return "나쁨", "#b42318", "#feece9", "#e04f40", "🔴"


def ingest(data):
    """Accept event array, {events:[...]}, or final session JSON."""
    if isinstance(data, list):
        events = data
    elif isinstance(data, dict):
        events = data.get("events", [])
        session = data.get("session", {})
        if isinstance(session, dict):
            st.session_state.title = str(session.get("title") or session.get("name") or st.session_state.title)
    else:
        raise ValueError("JSON 최상위 구조는 객체 또는 배열이어야 해.")
    if not isinstance(events, list):
        raise ValueError("events는 배열이어야 해.")
    scores = []
    notes = []
    suggestions = []
    transcripts = []
    component_status = {}
    valid = []
    for e in events:
        if not isinstance(e, dict):
            continue
        kind = e.get("event_type")
        payload = e.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        start = e.get("start_ms", 0)
        try:
            start = int(start)
        except (ValueError, TypeError):
            start = 0
        if kind == "engagement_window":
            score = payload.get("score", payload.get("engagement_score"))
            if score is not None:
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    score = None
            if score is not None and not 0 <= score <= 100:
                score = None
            scores.append((start, score))
        elif kind == "note_created":
            notes.append({"time_ms": start, "text": str(payload.get("text") or payload.get("content") or "메모")})
        elif kind == "resource_suggestion":
            source = payload.get("source")
            if isinstance(source, dict):
                source = str(source.get("document_name") or "자료") + (f" · {source['page']}쪽" if source.get("page") else "")
            suggestions.append({"time_ms": start,
                                "title": str(payload.get("title") or payload.get("document_title") or "자료 제안"),
                                "evidence": str(payload.get("evidence") or payload.get("transcript") or ""),
                                "source": str(source or "업로드 자료")})
        elif kind in {"transcript_final", "transcript_edited"}:
            transcripts.append({"time_ms": start, "text": str(payload.get("text", ""))})
        elif kind == "analysis_status":
            component_status[str(payload.get("component", "unknown"))] = str(payload.get("state", "unknown"))
        valid.append(e)
    st.session_state.events = valid
    st.session_state.scores = [score for _, score in sorted(scores)] if scores else []
    st.session_state.notes = sorted(notes, key=lambda n: n["time_ms"], reverse=True)
    st.session_state.suggestions = sorted(suggestions, key=lambda n: n["time_ms"], reverse=True)
    st.session_state.transcripts = sorted(transcripts, key=lambda n: n["time_ms"])
    st.session_state.component_status = component_status
    if valid:
        st.session_state.elapsed = max(int(e.get("end_ms") or e.get("start_ms") or 0) for e in valid) // 1000
    if isinstance(data, dict) and isinstance(data.get("session"), dict):
        duration = data["session"].get("duration_ms")
        if isinstance(duration, int) and not isinstance(duration, bool) and duration >= 0:
            st.session_state.elapsed = duration // 1000
    st.session_state.source = "불러온 JSON"
    st.session_state.dismissed = False


def make_chart(scores, notes, show_sample=False):
    fig = go.Figure()
    x = [i * 60 / max(len(scores) - 1, 1) for i in range(len(scores))]
    fig.add_trace(go.Scatter(x=x, y=scores, mode="lines", name="참여 점수", line=dict(color="#2457d6", width=3),
                             connectgaps=False, hovertemplate="%{x:.1f}분 · %{y:.0f}점<extra></extra>"))
    for label, spans in (SIGNALS.items() if show_sample else []):
        row = list(SIGNALS).index(label)
        for a, b in spans:
            fig.add_shape(type="rect", x0=a, x1=b, y0=-17-row*14, y1=-9-row*14,
                          fillcolor="#7f97e0", line_width=0)
    for row, label in enumerate(SIGNALS if show_sample else []):
        fig.add_annotation(x=-3, y=-13-row*14, text=label, showarrow=False, xanchor="right",
                           font=dict(size=11, color="#57606a"))
    if show_sample:
        fig.add_vrect(x0=37, x1=43, fillcolor="#dde1e6", opacity=.55, line_width=0,
                      annotation_text="분석 보류", annotation_position="top left")
        fig.add_vrect(x0=48, x1=57, fillcolor="#b3420f", opacity=.08, line_width=0)
        fig.add_annotation(x=52, y=95, text="⚠ 변화 알림", showarrow=False, font=dict(size=11, color="#b3420f"))
    for n in notes:
        minute = n["time_ms"] / 60000
        if minute <= 60:
            fig.add_trace(go.Scatter(x=[minute], y=[-72], mode="markers", marker=dict(size=9, color="#2457d6"),
                                     showlegend=False, hovertext=n["text"], hoverinfo="text"))
    fig.add_annotation(x=-3, y=-72, text="메모", showarrow=False, xanchor="right", font=dict(size=11, color="#57606a"))
    fig.update_xaxes(range=[-9, 61], tickvals=[0,20,40,60], ticktext=["00:00", "20:00", "40:00", "60:00"],
                     showgrid=False, zeroline=False, fixedrange=True)
    fig.update_yaxes(range=[-82, 110], tickvals=[0,50,100], showgrid=True, gridcolor="#e1e4e8", zeroline=False,
                     fixedrange=True)
    fig.update_layout(height=310, margin=dict(l=12,r=10,t=12,b=8), paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                      showlegend=False, font=dict(family="Arial, sans-serif", size=12, color="#57606a"))
    return fig


initialize()
if "live_session" not in st.session_state:
    st.session_state.live_session = None

st.markdown('<div class="page-eyebrow">온라인 수업 도우미</div><h1 class="page-title">수업 대시보드</h1><p class="page-description">수업 창을 연결하고 분석 상태와 메모를 한곳에서 확인하세요.</p>', unsafe_allow_html=True)

with st.expander("실시간 수업 창 연결", expanded=True):
    st.caption("창 연결은 분석을 시작하지 않습니다. 수업 제목을 확인한 뒤 아래 라이브 세션의 **수업 시작**을 누르면 타이머와 영상 분석이 함께 시작됩니다.")
    live = st.session_state.live_session
    if live is None:
        try:
            st.session_state.title = st.text_input("수업 제목", value=st.session_state.title,
                                                   placeholder="예: 데이터베이스 8주차")
            windows = list_windows()
            selected = st.selectbox("분석할 창", windows, index=None,
                                    placeholder="수업 창을 선택하세요",
                                    format_func=lambda item: f"{item[1]} ({item[0]})") if windows else None
            if not windows:
                st.info("열린 창을 찾지 못했습니다. 수업 창을 연 뒤 화면을 새로고침하세요.")
            if st.button("창 연결", type="primary", disabled=selected is None):
                st.session_state.live_session = LiveSession(selected[0])
                st.session_state.scores = []
                st.session_state.notes = []
                st.session_state.suggestions = []
                st.session_state.events = []
                st.session_state.transcripts = []
                st.session_state.component_status = {}
                st.session_state.source = "연결된 수업 창"
                st.session_state.elapsed = 0
                st.session_state.started = False
                st.rerun()
        except Exception as exc:
            st.error(f"창 목록을 가져오지 못했습니다: {exc}")
    else:
        # 연결 후에도 시작 전에는 제목을 바꿀 수 있다. 시작 순간 이 값이 공통 세션 JSON 제목으로 확정된다.
        st.session_state.title = st.text_input("수업 제목", value=st.session_state.title,
                                               disabled=live.has_started())
        if not live.has_started():
            st.success("수업 창이 연결되었습니다. 아래에서 제목을 확인하고 수업을 시작하세요.")
            if st.button("연결 해제"):
                st.session_state.live_session = None
                st.rerun()
        elif st.button("수업 종료"):
            try:
                live.stop()
                st.session_state.started = False
                st.success("수업과 영상 분석을 종료했습니다.")
            except RuntimeError as exc:
                st.error(str(exc))

        @st.fragment(run_every=1 if live.is_running() else None)
        def live_status():
            if not live.has_started():
                st.caption("수업 시작 전: 아직 타이머·영상 분석·타임라인 이벤트를 만들지 않습니다.")
                return
            result, error, running = live.snapshot()
            snapshot = live.service.get_dashboard_snapshot(live.sid)
            st.write(f"세션 시간: {clock(snapshot['session']['duration_ms'] / 1000)} · 영상: {snapshot['status'].get('vision', '준비 중')}")
            st.write("실험 행동 점수: 개발자 화면 확인용 · 공통 점수 이벤트/타임라인: 미연결 · STT: 미연결")
            if error:
                st.error(f"영상 분석 오류: {error}")
            elif result:
                features = result['features']
                st.write(f"현재 프레임: 얼굴 후보 {len(features['faces'])}개 · 상체 후보 {len(features['upper_bodies'])}개 · 품질 {result['quality']['status']}")
                st.caption("검출 수는 실제 참가자 수가 아닙니다. 개발자 화면의 규칙 기반 실험 점수는 참여도 평가 결과가 아닙니다.")
            elif running:
                st.info("영상 분석을 준비하고 있습니다.")
        live_status()
        if live.has_started() and not live.is_running():
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'session.json'
                live.service.save_session(live.sid, output)
                st.download_button('공통 계약 세션 JSON 저장', output.read_bytes(),
                                   file_name='session.json', mime='application/json')
        if live.has_started() and not live.is_running() and st.button("새 수업 연결"):
            st.session_state.live_session = None
            st.rerun()

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stApp"] {font-family:'Noto Sans KR',sans-serif;}
.stApp {background:#f5f6f7;color:#1f2328;}
.block-container {max-width:920px;padding:34px 28px 72px;}
header[data-testid="stHeader"] {background:transparent;}
[data-testid="stVerticalBlock"] {gap:.8rem;}
.page-eyebrow {color:#355cc9;font-size:12px;font-weight:800;letter-spacing:.04em;margin-bottom:4px;}
.page-title {color:#172033;font-size:30px !important;font-weight:800;line-height:1.2;margin:0 !important;padding:0 !important;}
.page-description {color:#5d697c;font-size:14px;margin:4px 0 18px;}
[data-testid="stExpander"] {background:#fff;border:1px solid #dce3ed;border-radius:14px;overflow:hidden;}
[data-testid="stExpander"] summary {background:#fff;color:#172033 !important;min-height:54px;}
[data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span {color:#172033 !important;font-weight:700;}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {color:#334155;}
[data-testid="stWidgetLabel"] p {color:#334155 !important;font-weight:600;}
[data-baseweb="select"] > div, [data-baseweb="input"] > div {background:#fff !important;color:#172033 !important;border-color:#cbd5e1 !important;}
[data-baseweb="select"] input, [data-baseweb="input"] input {color:#172033 !important;}
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button, [data-testid="stFormSubmitButton"] button {min-height:44px;border-radius:9px;white-space:normal;line-height:1.3;color:#23324b;background:#fff;border:1px solid #cbd5e1;}
[data-testid="stButton"] button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {background:#2457d6 !important;border-color:#2457d6 !important;color:#fff !important;}
[data-testid="stButton"] button p, [data-testid="stDownloadButton"] button p, [data-testid="stFormSubmitButton"] button p {color:inherit !important;}
.section {background:#fff;border:1px solid #c9ced4;border-radius:8px;padding:16px;margin:12px 0;}
.section-title {font-size:16px;font-weight:800;margin:0 0 7px;}
.muted {color:#57606a;font-size:12px;}
.live-head {display:flex;align-items:center;justify-content:space-between;gap:18px;min-height:98px;margin:16px 0 18px;padding:18px;background:#fff;border:1px solid #dce3ed;border-radius:14px;}
.head-left {display:flex;flex-direction:column;justify-content:space-between;min-height:86px;min-width:0;}
.head-title {font-weight:800;font-size:20px;margin-bottom:2px;}
.head-subtitle {font-size:13px;color:#57606a;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:260px;}
.running {display:inline-block;border:1px solid #1f2328;border-radius:100px;padding:6px 14px;font-weight:700;font-size:13px;white-space:nowrap;align-self:flex-start;}
.score-card {flex:0 1 300px;min-width:0;min-height:76px;border:1px solid var(--score-border);background:var(--score-bg);border-radius:12px;display:flex;align-items:center;justify-content:center;color:var(--score-color);font-size:18px;font-weight:800;text-align:center;padding:12px;overflow-wrap:anywhere;}
.alert {border:1px solid #b3420f;border-radius:6px;padding:12px;margin-top:6px;}
.alert-title {color:#b3420f;font-size:14px;font-weight:800;}
.resource {border:1px solid #2457d6;border-radius:6px;padding:12px;margin-top:8px;}
.st-key-alert-actions {padding-top:14px;}
.st-key-resource-actions {padding-top:16px;}
.st-key-resource-actions [data-testid="stHorizontalBlock"] {flex-wrap:nowrap;gap:12px;}
.st-key-resource-actions [data-testid="stColumn"] {min-width:0 !important;}
.pill {background:#e6ebfa;color:#2457d6;font-weight:700;font-size:12px;padding:2px 8px;border-radius:4px;}
.note-item {border-bottom:1px solid #edf0f2;padding:7px 0;font-size:13px;}
@media(max-width:590px){.live-head{align-items:stretch;flex-direction:column}.head-left{min-height:74px;gap:7px}.score-card{flex:initial;width:100%;min-height:68px}.block-container{padding:18px 14px 50px}.page-title{font-size:25px !important}}
</style>""", unsafe_allow_html=True)

with st.expander("⚙️ 세션 설정 · JSON 이벤트 연결", expanded=False):
    st.caption(f"수업 제목: {st.session_state.title} · 수업 시작 전에는 위의 수업 창 연결 영역에서 수정할 수 있습니다.")
    uploaded = st.file_uploader("세션 이벤트 JSON 불러오기", type=["json"])
    if st.button("공통 계약 예제 불러오기", use_container_width=True):
        sample_path = Path(__file__).resolve().parent.parent / "공유용_데이터_계약" / "examples" / "session.json"
        try:
            with sample_path.open(encoding="utf-8") as handle:
                ingest(json.load(handle))
            st.rerun()
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            st.error(f"예제 파일을 읽지 못했어: {exc}")
    if uploaded is not None and st.button("JSON 적용", use_container_width=True):
        try:
            ingest(json.load(uploaded))
            st.success("세션 데이터를 적용했어.")
            st.rerun()
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            st.error(f"파일을 읽지 못했어: {exc}")
    st.caption(f"현재 데이터: {st.session_state.source} · 이벤트 {len(st.session_state.events)}건")
    if st.button("예시 데이터로 초기화"):
        for key in ["scores", "notes", "suggestions", "events", "source", "elapsed", "started", "dismissed", "transcripts", "component_status"]:
            del st.session_state[key]
        initialize()
        st.rerun()

scores = st.session_state.scores
if st.session_state.source == "예시 데이터":
    st.info("아래 점수와 타임라인은 화면 확인용 예시이며 실제 분석 결과가 아닙니다.")
latest = next((x for x in reversed(scores) if x is not None), None)
label, color, bg, border, emoji = status(latest)
if scores and scores[-1] is None:
    label, color, bg, border, emoji = status(None)
score_text = f"{latest:g}점" if latest is not None and label != "분석 보류" else "측정 불가"

@st.fragment(run_every=1 if (st.session_state.live_session is not None and
                              st.session_state.live_session.is_running()) else None)
def live_session_header():
    """수업 시작 시각을 단일 기준으로 표시하고, 시작 동작을 한 곳에 모은다."""
    active_live = st.session_state.live_session
    is_running = active_live is not None and active_live.is_running()
    is_connected = active_live is not None
    # 저장 JSON/예시를 보는 경우에는 기존 세션 길이를 유지하고,
    # 창만 연결한 상태에서는 실제 수업이 아직 시작되지 않았으므로 00:00을 표시한다.
    elapsed = (active_live.elapsed_seconds() if active_live is not None and active_live.has_started()
               else (0 if active_live is not None else st.session_state.elapsed))
    st.markdown(f"""<div class="live-head"><div class="head-left">
<div><div class="head-title">라이브 세션</div><div class="head-subtitle">{escape(st.session_state.title)}</div></div>
<div class="running">{'수업 진행 중' if is_running else ('수업 시작 대기' if is_connected else '창 연결 대기')} · {clock(elapsed)}</div></div>
<div class="score-card" style="--score-bg:{bg};--score-border:{border};--score-color:{color}">
{emoji} 현재 상태 · {label} · {score_text}</div></div>""", unsafe_allow_html=True)
    start_column, refresh_column = st.columns([1, 1])
    with start_column:
        if not active_live or not active_live.has_started():
            if st.button("▶ 수업 시작", use_container_width=True, type="primary", disabled=not is_connected):
                try:
                    active_live.start(st.session_state.title)
                    # 이 순간부터 타이머, 분석 이벤트, 이후의 타임라인 데이터가 같은 세션에 쌓인다.
                    st.session_state.source = "실시간 영상"
                    st.session_state.started = True
                    st.session_state.elapsed = 0
                    st.rerun()
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))
        elif is_running:
            if st.button("■ 수업 종료", use_container_width=True):
                try:
                    active_live.stop()
                    st.session_state.started = False
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))
        else:
            st.button("수업 종료됨", use_container_width=True, disabled=True)
    with refresh_column:
        if st.button("↻ 화면 새로고침", use_container_width=True, disabled=is_running):
            st.rerun()
    if is_running:
        st.caption("수업 진행 중에는 화면 내 새로고침을 막아 분석 상태와 진행 중 세션을 보호합니다.")


live_session_header()

@st.fragment(run_every=1 if (st.session_state.live_session is not None and
                              st.session_state.live_session.is_running()) else None)
def timeline_panel():
    """실시간 세션의 통합 이벤트를 읽어, 시작 후 생성되는 타임라인을 표시한다."""
    active_live = st.session_state.live_session
    timeline_scores = scores
    timeline_notes = st.session_state.notes
    if active_live is not None and active_live.has_started():
        snapshot = active_live.service.get_dashboard_snapshot(active_live.sid)
        # 참여 모델이 연결되면 engagement_window 이벤트가 여기로 들어와 즉시 그래프에 반영된다.
        windows = snapshot['engagement_windows']
        timeline_scores = [event['payload'].get('score', event['payload'].get('engagement_score'))
                           for event in windows]
        timeline_notes = [
            {'time_ms': event['start_ms'], 'text': event['payload']['text']}
            for event in snapshot['notes']
        ]
    st.markdown('<div class="section-title" style="margin-top:15px">공통 타임라인</div><div class="muted">참여 점수 0–100 · 좋음 ≥70 · 보통 40–69 · 나쁨 &lt;40</div>', unsafe_allow_html=True)
    if timeline_scores:
        st.plotly_chart(make_chart(timeline_scores, timeline_notes,
                                   st.session_state.source == "예시 데이터"),
                        use_container_width=True, config={"displayModeBar":False})
    elif active_live is not None and active_live.has_started():
        st.info("수업 타임라인을 생성 중입니다. 참여 모델이 연결되면 점수 구간이 여기에 표시됩니다.")
    else:
        st.info("아직 참여도 데이터가 없어. 이벤트 JSON을 불러오면 타임라인에 표시돼.")


timeline_panel()

if st.session_state.source == "예시 데이터" and not st.session_state.dismissed:
    st.markdown('''<div class="alert"><div class="alert-title">변화 알림 · 00:48 – 00:57</div>
<div class="muted">근거: 얼굴 방향 변화 지속 · 분석 가능률 00% (예시 데이터)</div></div>''', unsafe_allow_html=True)
    with st.container(key="alert-actions"):
        if st.button("변화 알림 닫기", key="dismiss_alert"):
            st.session_state.dismissed = True
            st.rerun()
elif st.session_state.source == "예시 데이터" and st.session_state.dismissed:
    if st.button("변화 알림 열기", key="show_alert"):
        st.session_state.dismissed = False
        st.rerun()

st.markdown('<div class="section-title" style="margin-top:24px">강의 조교 · 자료 제안</div>', unsafe_allow_html=True)
if st.session_state.suggestions:
    s = st.session_state.suggestions[0]
    st.markdown(f'''<div class="resource"><span class="pill">새 제안</span> <span class="muted">{event_time(s['time_ms'])}</span>
<div style="font-size:14px;font-weight:700;margin-top:12px">{escape(s['title'])}</div>
<div class="muted" style="margin-top:6px">근거 전사: {escape(s['evidence'])} · 출처 {escape(s['source'])}</div></div>''', unsafe_allow_html=True)
    # 제안 박스 아래 여백을 두고 두 버튼을 한 줄에 배치한다.
    with st.container(key="resource-actions"):
        save_column, hide_column = st.columns(2, gap="medium")
        with save_column:
            if st.button("제안 메모로 저장", use_container_width=True, type="primary"):
                st.session_state.notes.insert(0, {"time_ms": st.session_state.elapsed*1000, "text": s["title"]})
                st.toast("메모에 저장했어")
                st.rerun()
        with hide_column:
            if st.button("이번 제안 숨기기", use_container_width=True):
                st.session_state.suggestions.pop(0)
                st.rerun()
    if len(st.session_state.suggestions) > 1:
        with st.expander(f"이전 제안 {len(st.session_state.suggestions)-1}건 보기"):
            for old in st.session_state.suggestions[1:]:
                st.write(f"{event_time(old['time_ms'])} · {old['title']}")
else:
    st.caption("현재 표시할 자료 제안이 없어.")

if st.session_state.source != "예시 데이터":
    st.markdown('<div class="section-title" style="margin-top:24px">분석 상태와 전사</div>', unsafe_allow_html=True)
    for component, state in st.session_state.component_status.items():
        st.write(f"{component}: {state}")
    if st.session_state.transcripts:
        for transcript in st.session_state.transcripts:
            st.write(f"{event_time(transcript['time_ms'])} · {transcript['text']}")
    else:
        st.caption("저장된 확정 전사가 없습니다.")

st.markdown('<div class="section-title" style="margin-top:24px">교수자 메모</div>', unsafe_allow_html=True)
note_live = st.session_state.live_session
note_before_start = note_live is not None and not note_live.has_started()
note_time = (st.session_state.live_session.elapsed_seconds()
             if st.session_state.live_session is not None and st.session_state.live_session.has_started()
             else st.session_state.elapsed)
st.caption(f"{clock(note_time)}에 연결")
with st.form("note_form", clear_on_submit=True):
    note = st.text_input("메모 입력", placeholder="이 설명 다시 짚기", label_visibility="collapsed",
                         disabled=note_before_start)
    submitted = st.form_submit_button("메모 저장", use_container_width=True, type="primary",
                                      disabled=note_before_start)
    if submitted and note.strip():
        live = st.session_state.live_session
        if live is not None:
            try:
                now = live.service.elapsed_ms(live.sid)
                event = live.service.make_event(live.sid, 'note_created',
                    {'text': note.strip(), 'linked_event_ids': []}, now, now, 'ui')
                live.service.receive_event(event)
                st.session_state.notes.insert(0, {"time_ms": now, "text": note.strip()})
                st.rerun()
            except Exception as exc:
                st.error(f"메모를 저장하지 못했습니다: {exc}")
        else:
            st.session_state.notes.insert(0, {"time_ms": st.session_state.elapsed*1000, "text": note.strip()})
            st.rerun()
st.markdown('<div class="muted">Enter로 저장 · 저장하면 타임라인에 메모 점이 찍혀.</div>', unsafe_allow_html=True)
if note_before_start:
    st.caption("연결된 수업 창의 메모와 타임라인 기록은 수업 시작 후 활성화됩니다.")
st.markdown('<div style="font-size:13px;font-weight:700;margin-top:13px">최근 메모</div>', unsafe_allow_html=True)
for n in st.session_state.notes[:5]:
    st.markdown(f'<div class="note-item"><span style="color:#57606a;margin-right:12px">{event_time(n["time_ms"])}</span>{escape(n["text"])}</div>', unsafe_allow_html=True)

export = {"schema_version":"1.0", "session":{"title":st.session_state.title},
          "events":st.session_state.events + [
              {"schema_version":"1.0","session_id":"local-demo","event_id":f"note-{i}",
               "event_type":"note_created","start_ms":n["time_ms"],"end_ms":n["time_ms"],
               "created_ms":int(time.time()*1000),"payload":{"text":n["text"]},
               "quality":{},"provenance":{"source":"streamlit"}}
              for i,n in enumerate(st.session_state.notes)]}
if st.session_state.live_session is None:
    st.caption("내보내기는 프로토타입 형식이며 공통 세션 계약과 아직 호환되지 않습니다.")
    st.download_button("⬇️ 세션 JSON 내보내기", json.dumps(export,ensure_ascii=False,indent=2),
                       file_name="lecturer_session.json", mime="application/json", use_container_width=True)
st.caption("참여 모델, STT와 자료 검색은 아직 연결되지 않았습니다.")

# 개발자 확인 기능은 일반 수업 조작과 분리해 화면의 맨 아래에 둔다.
st.divider()
st.markdown('<div class="section-title">개발자 도구</div>', unsafe_allow_html=True)
active_live = st.session_state.live_session
if active_live is not None and active_live.is_running():
    # 운영 화면과 분리된 OpenCV 창이다. 실제 분석에 이미 쓰이는 프레임 복사본만 표시한다.
    # 개발 확인용 원시 좌표·랜드마크는 이 창 밖으로 저장하거나 전송하지 않는다.
    preview_on = active_live.developer_preview_enabled()
    if st.button("개발자 모드 끄기" if preview_on else "개발자 모드 켜기",
                 key="developer_preview", type="secondary"):
        active_live.set_developer_preview(not preview_on)
        st.rerun()
    st.caption(
        "별도 OpenCV 창에 실제 분석 입력 화면과 특징값을 표시합니다. 얼굴별 Vision relative yaw는 보정 전 값이며, "
        "Model relative yaw가 데드밴드·민감도 보정을 거친 실제 모델 입력값입니다. "
        "V: 점 표시 전환 · C: 3초 정면 보정 시작 · Q: 개발자 창 숨기기"
    )
else:
    st.caption("영상 분석을 시작하면 개발자 모드를 사용할 수 있습니다.")
