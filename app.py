"""Streamlit lecturer dashboard prototype based on the team's Figma live-session panel.
Run: streamlit run app.py
"""
import json
import time
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
            suggestions.append({"time_ms": start,
                                "title": str(payload.get("title") or payload.get("document_title") or "자료 제안"),
                                "evidence": str(payload.get("evidence") or payload.get("transcript") or ""),
                                "source": str(payload.get("source") or "업로드 자료")})
        valid.append(e)
    st.session_state.events = valid
    st.session_state.scores = [score for _, score in sorted(scores)] if scores else []
    st.session_state.notes = sorted(notes, key=lambda n: n["time_ms"], reverse=True)
    st.session_state.suggestions = sorted(suggestions, key=lambda n: n["time_ms"], reverse=True)
    if valid:
        st.session_state.elapsed = max(int(e.get("end_ms") or e.get("start_ms") or 0) for e in valid) // 1000
    st.session_state.source = "불러온 JSON"
    st.session_state.dismissed = False


def make_chart(scores, notes):
    fig = go.Figure()
    x = [i * 60 / max(len(scores) - 1, 1) for i in range(len(scores))]
    fig.add_trace(go.Scatter(x=x, y=scores, mode="lines", name="참여 점수", line=dict(color="#2457d6", width=3),
                             connectgaps=False, hovertemplate="%{x:.1f}분 · %{y:.0f}점<extra></extra>"))
    for label, spans in SIGNALS.items():
        row = list(SIGNALS).index(label)
        for a, b in spans:
            fig.add_shape(type="rect", x0=a, x1=b, y0=-17-row*14, y1=-9-row*14,
                          fillcolor="#7f97e0", line_width=0)
    for row, label in enumerate(SIGNALS):
        fig.add_annotation(x=-3, y=-13-row*14, text=label, showarrow=False, xanchor="right",
                           font=dict(size=11, color="#57606a"))
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
if st.session_state.started:
    now = time.time()
    st.session_state.elapsed += max(0, int(now - st.session_state.last_tick))
    st.session_state.last_tick = now

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stApp"] {font-family:'Noto Sans KR',sans-serif;}
.stApp {background:#f5f6f7;color:#1f2328;}
.block-container {max-width:680px;padding:18px 20px 65px;}
header[data-testid="stHeader"] {background:transparent;}
[data-testid="stVerticalBlock"] {gap:.55rem;}
.section {background:#fff;border:1px solid #c9ced4;border-radius:8px;padding:16px;margin:12px 0;}
.section-title {font-size:16px;font-weight:800;margin:0 0 7px;}
.muted {color:#57606a;font-size:12px;}
.live-head {display:flex;align-items:center;justify-content:space-between;gap:12px;min-height:98px;}
.head-left {display:flex;flex-direction:column;justify-content:space-between;min-height:86px;min-width:0;}
.head-title {font-weight:800;font-size:20px;margin-bottom:2px;}
.head-subtitle {font-size:13px;color:#57606a;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:260px;}
.running {display:inline-block;border:1px solid #1f2328;border-radius:100px;padding:6px 14px;font-weight:700;font-size:13px;white-space:nowrap;align-self:flex-start;}
.score-card {flex:0 0 284px;min-height:76px;border:2px solid var(--score-border);background:var(--score-bg);border-radius:14px;display:flex;align-items:center;justify-content:center;color:var(--score-color);font-size:21px;font-weight:800;text-align:center;padding:8px;}
.alert {border:1px solid #b3420f;border-radius:6px;padding:12px;margin-top:6px;}
.alert-title {color:#b3420f;font-size:14px;font-weight:800;}
.resource {border:1px solid #2457d6;border-radius:6px;padding:12px;margin-top:8px;}
.pill {background:#e6ebfa;color:#2457d6;font-weight:700;font-size:12px;padding:2px 8px;border-radius:4px;}
.note-item {border-bottom:1px solid #edf0f2;padding:7px 0;font-size:13px;}
[data-testid="stButton"] button[kind="primary"] {background:#1f2328;border-color:#1f2328;}
@media(max-width:590px){.live-head{align-items:stretch;flex-direction:column}.head-left{min-height:74px;gap:7px}.score-card{flex:initial;width:100%;min-height:76px}.block-container{padding:12px}}
</style>""", unsafe_allow_html=True)

with st.expander("⚙️ 세션 설정 · JSON 이벤트 연결", expanded=False):
    title = st.text_input("수업 제목", value=st.session_state.title)
    st.session_state.title = title
    uploaded = st.file_uploader("세션 이벤트 JSON 불러오기", type=["json"])
    if uploaded is not None and st.button("JSON 적용", use_container_width=True):
        try:
            ingest(json.load(uploaded))
            st.success("세션 데이터를 적용했어.")
            st.rerun()
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            st.error(f"파일을 읽지 못했어: {exc}")
    st.caption(f"현재 데이터: {st.session_state.source} · 이벤트 {len(st.session_state.events)}건")
    if st.button("예시 데이터로 초기화"):
        for key in ["scores", "notes", "suggestions", "events", "source", "elapsed", "started", "dismissed"]:
            del st.session_state[key]
        initialize()
        st.rerun()

scores = st.session_state.scores
latest = next((x for x in reversed(scores) if x is not None), None)
label, color, bg, border, emoji = status(latest)
if scores and scores[-1] is None:
    label, color, bg, border, emoji = status(None)
score_text = f"{latest:g}점" if latest is not None and label != "분석 보류" else "측정 불가"

st.markdown(f"""<div class="live-head"><div class="head-left">
<div><div class="head-title">라이브 세션</div><div class="head-subtitle">{escape(st.session_state.title)}</div></div>
<div class="running">{'분석 중' if st.session_state.started else '분석 대기'} · {clock(st.session_state.elapsed)}</div></div>
<div class="score-card" style="--score-bg:{bg};--score-border:{border};--score-color:{color}">
{emoji} 현재 상태 · {label} · {score_text}</div></div>""", unsafe_allow_html=True)

c1, c2 = st.columns([1,1])
with c1:
    if st.button("⏸ 분석 일시정지" if st.session_state.started else "▶ 분석 시작", use_container_width=True, type="primary"):
        st.session_state.started = not st.session_state.started
        st.session_state.last_tick = time.time()
        st.rerun()
with c2:
    if st.button("↻ 화면 새로고침", use_container_width=True):
        st.rerun()

st.markdown('<div class="section-title" style="margin-top:15px">공통 타임라인</div><div class="muted">참여 점수 0–100 · 좋음 ≥70 · 보통 40–69 · 나쁨 &lt;40</div>', unsafe_allow_html=True)
if scores:
    st.plotly_chart(make_chart(scores, st.session_state.notes), use_container_width=True, config={"displayModeBar":False})
else:
    st.info("아직 참여도 데이터가 없어. 이벤트 JSON을 불러오면 타임라인에 표시돼.")

if not st.session_state.dismissed:
    st.markdown('''<div class="alert"><div class="alert-title">변화 알림 · 00:48 – 00:57</div>
<div class="muted">근거: 얼굴 방향 변화 지속 · 분석 가능률 00% (예시 데이터)</div></div>''', unsafe_allow_html=True)
    if st.button("변화 알림 닫기", key="dismiss_alert"):
        st.session_state.dismissed = True
        st.rerun()

st.markdown('<div class="section-title" style="margin-top:24px">강의 조교 · 자료 제안</div>', unsafe_allow_html=True)
if st.session_state.suggestions:
    s = st.session_state.suggestions[0]
    st.markdown(f'''<div class="resource"><span class="pill">새 제안</span> <span class="muted">{event_time(s['time_ms'])}</span>
<div style="font-size:14px;font-weight:700;margin-top:12px">{escape(s['title'])}</div>
<div class="muted" style="margin-top:6px">근거 전사: {escape(s['evidence'])} · 출처 {escape(s['source'])}</div></div>''', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        if st.button("제안 메모로 저장", use_container_width=True, type="primary"):
            st.session_state.notes.insert(0, {"time_ms": st.session_state.elapsed*1000, "text": s["title"]})
            st.toast("메모에 저장했어")
            st.rerun()
    with b:
        if st.button("이번 제안 숨기기", use_container_width=True):
            st.session_state.suggestions.pop(0)
            st.rerun()
    if len(st.session_state.suggestions) > 1:
        with st.expander(f"이전 제안 {len(st.session_state.suggestions)-1}건 보기"):
            for old in st.session_state.suggestions[1:]:
                st.write(f"{event_time(old['time_ms'])} · {old['title']}")
else:
    st.caption("현재 표시할 자료 제안이 없어.")

st.markdown('<div class="section-title" style="margin-top:24px">교수자 메모</div>', unsafe_allow_html=True)
st.caption(f"{clock(st.session_state.elapsed)}에 연결")
with st.form("note_form", clear_on_submit=True):
    note = st.text_input("메모 입력", placeholder="이 설명 다시 짚기", label_visibility="collapsed")
    submitted = st.form_submit_button("메모 저장", use_container_width=True, type="primary")
    if submitted and note.strip():
        st.session_state.notes.insert(0, {"time_ms": st.session_state.elapsed*1000, "text": note.strip()})
        st.rerun()
st.markdown('<div class="muted">Enter로 저장 · 저장하면 타임라인에 메모 점이 찍혀.</div>', unsafe_allow_html=True)
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
st.download_button("⬇️ 세션 JSON 내보내기", json.dumps(export,ensure_ascii=False,indent=2),
                   file_name="lecturer_session.json", mime="application/json", use_container_width=True)
st.caption("프로토타입: 실제 영상 캡처·MediaPipe·sherpa-onnx·자료 검색은 별도 백엔드 연동이 필요해.")
