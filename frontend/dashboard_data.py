"""Common-contract view adapters. UI state is never a second event store."""
import math
from pathlib import Path
from uuid import UUID
import plotly.graph_objects as go
from integration import IntegrationService


def clock(ms):
    seconds = max(0, int(ms / 1000))
    return f'{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}'


def score(event):
    payload = event['payload']
    output = payload.get('model_output') or {}
    value = payload.get('score', payload.get('engagement_score', output.get('value', output.get('score', output.get('engagement_score')))))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value if 0 <= value <= 100 and event['quality']['status'] != 'not_observable' else None


def overlap(event, start, end):
    a, b = event['start_ms'], event['end_ms']
    return start <= a <= end if a == b else a < end and b > start


def suggestion_phase(event):
    """생성 시각의 공통 계약 필드로 실시간과 종료 후 생성을 구분한다."""
    if event.get('created_ms') is not None:
        return '수업 중 제안'
    if event.get('created_at_utc'):
        return '수업 후 제안'
    return '생성 시점 미상'


def related_suggestions(transcript, suggestions):
    """시각이 아닌 명시적 근거 ID로 전사와 제안을 연결한다."""
    return [event for event in suggestions
            if transcript['event_id'] in event['payload'].get('evidence_event_ids', [])]


def suggestion_in_range(event, start, end, transcripts):
    """제안 시각 또는 명시적 근거 전사가 범위에 있으면 표시한다."""
    if overlap(event, start, end):
        return True
    ids = set(event['payload'].get('evidence_event_ids', []))
    return any(t['event_id'] in ids and overlap(t, start, end) for t in transcripts)


def chart(windows, notes, duration, selected=None):
    fig = go.Figure()
    ordered = sorted(windows, key=lambda e: (e['start_ms'], e['end_ms'], e['event_id']))
    xs, ys = [], []
    previous_end = None
    for event in ordered:
        if previous_end is not None and event['start_ms'] > previous_end:
            xs.append(event['start_ms'] / 1000)
            ys.append(None)
        xs.append(event['end_ms'] / 1000)
        ys.append(score(event))
        previous_end = max(previous_end or 0, event['end_ms'])
    fig.add_trace(go.Scatter(x=xs, y=ys,
                            mode='lines+markers', connectgaps=False, name='참여 점수',
                            marker={'size': 4}, line={'color': '#2457d6', 'width': 2},
                            hovertemplate='%{x:.1f}초 · %{y:.1f}점<extra></extra>'))
    for event in ordered:
        if score(event) is None:
            fig.add_vrect(x0=event['start_ms']/1000, x1=event['end_ms']/1000,
                          fillcolor='#e6e9ed', opacity=.7, line_width=0, layer='below')
    fig.add_trace(go.Scatter(x=[e['start_ms']/1000 for e in notes], y=[-12]*len(notes),
                            mode='markers', marker={'color':'#1f2328','size':7}, name='메모',
                            text=[e['payload']['text'] for e in notes], hovertemplate='%{text}<extra>메모</extra>'))
    if selected:
        fig.add_vrect(x0=selected[0]/1000, x1=selected[1]/1000, fillcolor='#2457d6', opacity=.1, line_width=0)
    end = max(duration/1000, 1)
    ticks = [end*i/6 for i in range(7)]
    fig.update_xaxes(range=[0,end], tickvals=ticks, ticktext=[clock(t*1000) for t in ticks], showgrid=False)
    fig.update_yaxes(range=[-20,105], tickvals=[0,50,100], gridcolor='#e6e9ed')
    fig.update_layout(height=300, margin={'l':30,'r':10,'t':15,'b':20}, showlegend=False,
                      paper_bgcolor='white', plot_bgcolor='white', dragmode='select',
                      font={'color':'#57606a'}, uirevision='timeline')
    return fig


class SessionFiles:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, sid):
        return self.root / f'{UUID(sid)}.json'

    def save(self, service, sid):
        return service.save_session(sid, self.path(sid))

    def recovery_path(self, sid):
        return self.root / '.recovery' / f'{UUID(sid)}.json'

    def clear_recovery(self, sid):
        self.recovery_path(sid).unlink(missing_ok=True)

    def recoveries(self):
        result = []
        for path in (self.root / '.recovery').glob('*.json'):
            try:
                service, sid = self.load(path)
                if path == self.recovery_path(sid):
                    result.append((sid, service.get_dashboard_snapshot(sid)['session']))
            except Exception:
                continue
        return result

    def load(self, path):
        service = IntegrationService()
        sid = service.load_session(path)
        return service, sid

    def list(self):
        result = []
        for path in sorted(self.root.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                service, sid = self.load(path)
                result.append((sid, service.get_dashboard_snapshot(sid)['session']))
            except Exception as exc:
                # A corrupt saved file must not hide other usable sessions.
                result.append((None, {'title': f'읽을 수 없는 파일: {path.name}', 'error': str(exc)}))
        return result
