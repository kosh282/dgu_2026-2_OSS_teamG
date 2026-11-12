"""Common-contract view adapters. UI state is never a second event store."""
import math
from pathlib import Path
from uuid import UUID
import pandas as pd
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


def unobservable_intervals(windows):
    """점수가 없는 구간을 합쳐 겹친 부분이 진해지지 않게 한다."""
    intervals = sorted((e['start_ms'], e['end_ms']) for e in windows if score(e) is None)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def live_chart_data(windows, notes, duration):
    """실시간 차트의 시간 경계, 점수, 관측 불가 구간과 메모를 만든다."""
    rows = [{'kind': 'bound', 'time': 0.0},
            {'kind': 'bound', 'time': max(duration / 1000, 1)}]
    ordered = sorted(windows, key=lambda e: (e['start_ms'], e['end_ms'], e['event_id']))
    previous_end = None
    sequence = 0
    for event in ordered:
        if previous_end is not None and event['start_ms'] > previous_end:
            rows.append({'kind': 'score', 'time': event['start_ms'] / 1000,
                         'value': None, 'sequence': sequence})
            sequence += 1
        rows.append({'kind': 'score', 'time': event['end_ms'] / 1000,
                     'time_label': clock(event['end_ms']), 'value': score(event),
                     'sequence': sequence})
        sequence += 1
        previous_end = max(previous_end or 0, event['end_ms'])
    for start, end in unobservable_intervals(windows):
        rows.append({'kind': 'unobservable', 'time': start / 1000, 'end': end / 1000})
    for event in notes:
        rows.append({'kind': 'note', 'time': event['start_ms'] / 1000, 'value': -12,
                     'time_label': clock(event['start_ms']), 'text': event['payload']['text']})
    return pd.DataFrame(rows, columns=['kind', 'time', 'end', 'value', 'sequence', 'time_label', 'text'])


def live_chart_spec():
    """데이터와 분리된 고정 차트 설정으로 실시간 갱신 시 요소를 유지한다."""
    time_axis = {
        'title': None, 'grid': False, 'tickCount': 7,
        'labelExpr': "(floor(datum.value / 3600) < 10 ? '0' : '') + floor(datum.value / 3600) + ':' + (floor(datum.value / 60) % 60 < 10 ? '0' : '') + floor(datum.value / 60) % 60 + ':' + (floor(datum.value) % 60 < 10 ? '0' : '') + floor(datum.value) % 60",
    }
    x = {'field': 'time', 'type': 'quantitative', 'scale': {'zero': False, 'nice': False}, 'axis': time_axis}
    y = {'field': 'value', 'type': 'quantitative', 'scale': {'domain': [-20, 105]},
         'axis': {'title': None, 'values': [0, 50, 100], 'gridColor': '#e6e9ed'}}
    return {
        'height': 265, 'background': 'white',
        'layer': [
            {'transform': [{'filter': "datum.kind === 'bound'"}],
             'mark': {'type': 'point', 'opacity': 0}, 'encoding': {'x': x}},
            {'transform': [{'filter': "datum.kind === 'unobservable'"}],
             'mark': {'type': 'rect', 'color': '#e6e9ed', 'opacity': 0.7},
             'encoding': {'x': x, 'x2': {'field': 'end'}}},
            {'transform': [{'filter': "datum.kind === 'score'"}],
             'mark': {'type': 'line', 'color': '#2457d6', 'strokeWidth': 2,
                      'point': {'size': 16}, 'invalid': 'break-paths-show-domains'},
             'encoding': {'x': x, 'y': y, 'order': {'field': 'sequence', 'type': 'quantitative'},
                          'tooltip': [{'field': 'time_label', 'type': 'nominal', 'title': '수업 시간'},
                                      {'field': 'value', 'type': 'quantitative', 'title': '실험 점수'}]}},
            {'transform': [{'filter': "datum.kind === 'note'"}],
             'mark': {'type': 'point', 'filled': True, 'color': '#1f2328', 'size': 49},
             'encoding': {'x': x, 'y': y,
                          'tooltip': [{'field': 'time_label', 'type': 'nominal', 'title': '수업 시간'},
                                      {'field': 'text', 'type': 'nominal', 'title': '메모'}]}},
        ],
        'config': {'view': {'stroke': None}, 'axis': {'labelColor': '#57606a'}},
    }


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
    for start, end in unobservable_intervals(ordered):
        fig.add_vrect(x0=start/1000, x1=end/1000,
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
