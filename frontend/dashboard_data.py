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


def review_range_options(duration_ms):
    """초 단위 표시가 중복되지 않는 검토 선택기 눈금을 제공한다."""
    end = max(1000, duration_ms)
    step = max(1000, ((end + 9_999_999) // 10_000_000) * 1000)
    options = list(range(0, end, step))
    if options[-1] // 1000 == end // 1000:
        options.pop()
    return options + [end]


def live_score_unavailable_reason(model_status, model_enabled, stale, rows, latest_window, status_message=None):
    """점수가 없을 때 화면에 표시할 관측 가능한 사유를 우선순위대로 고른다."""
    reason_codes = latest_window.get('quality', {}).get('reason_codes', []) if latest_window else []
    if model_status == 'unavailable':
        return status_message or '모델 또는 영상 입력을 확인하세요.'
    if model_status == 'stopped' or not model_enabled:
        return '관찰 점수 모델이 꺼져 있습니다.'
    if stale:
        return '최근 점수 구간이 5초 이상 갱신되지 않았습니다.'
    if any(row.get('calibration_status') == 'collecting' for row in rows):
        return '정면 기준을 수집 중입니다. 관측된 얼굴이 정면을 유지해 주세요.'
    if any('eye_observation_unavailable' in row.get('reason_codes', []) for row in rows):
        return '눈 관측이 부족하거나 수집 중입니다. 얼굴 크기와 가림을 확인하세요.'
    if 'insufficient_scoreable_time' in reason_codes:
        return '최근 구간의 점수화 가능한 관측 시간이 부족합니다.'
    if 'insufficient_observation' in reason_codes:
        return '최근 구간의 얼굴 관측 시간이 부족합니다.'
    return '얼굴 관측과 첫 점수 구간을 기다리는 중입니다.'


def score(event):
    payload = event['payload']
    if 'model_output' in payload:
        output = payload.get('model_output') or {}
        value = output.get('value', output.get('score', output.get('engagement_score')))
    else:
        value = payload.get('score', payload.get('engagement_score'))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value if 0 <= value <= 100 and event['quality']['status'] != 'not_observable' else None


def live_score_display(windows, now_ms, *, running, model_enabled, model_status, hold_ms=15000):
    """관측 구간은 유지하고 최근 유효값만 제한 시간 동안 화면에 표시한다."""
    if not windows:
        return {'state': 'waiting', 'value': None, 'window': None, 'age_ms': None}
    ordered = sorted(windows, key=lambda e: (e['end_ms'], e.get('created_ms') or 0, e.get('event_id', '')))
    latest = ordered[-1]
    age = now_ms - latest['end_ms']
    current = score(latest)
    if current is not None and 0 <= age <= 5000 and model_status not in ('unavailable', 'stopped') and running and model_enabled:
        return {'state': 'current', 'value': current, 'window': latest, 'age_ms': age}
    if running and model_enabled and model_status not in ('unavailable', 'stopped'):
        for event in reversed(ordered):
            value = score(event)
            age = now_ms - event['end_ms']
            if value is not None and 0 <= age <= hold_ms:
                return {'state': 'held', 'value': value, 'window': event, 'age_ms': age}
    return {'state': 'unavailable', 'value': None, 'window': latest, 'age_ms': None}


def transcript_score_link(transcript, windows, *, max_age_ms=5000):
    """전사 종료 시점까지 확정된 최신 구간만 연결한다. 결측을 건너뛰지 않는다."""
    end = transcript['end_ms']
    completed = sorted((e for e in windows if e['end_ms'] <= end),
                       key=lambda e: (e['end_ms'], e['start_ms'], e.get('created_ms') or 0))
    if not completed:
        return {'window': None, 'value': None, 'delta': None, 'reason': '점수 구간 대기'}
    latest = completed[-1]
    if end - latest['end_ms'] > max_age_ms:
        return {'window': None, 'value': None, 'delta': None, 'reason': '최근 점수 구간 없음'}
    value = score(latest)
    if value is None:
        return {'window': latest, 'value': None, 'delta': None, 'reason': '점수 보류'}
    previous = completed[-2] if len(completed) > 1 else None
    previous_value = score(previous) if previous else None
    delta = (value - previous_value if previous_value is not None
             and 0 < latest['end_ms'] - previous['end_ms'] <= max_age_ms else None)
    return {'window': latest, 'value': value, 'delta': delta, 'reason': None}


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


def suggestion_anchors(transcripts, suggestions):
    """근거가 여러 개인 제안도 도착 시점에 가장 가까운 근거 전사 한 곳에만 배치한다."""
    ordered = sorted(transcripts, key=lambda event: (event['start_ms'], event['end_ms'], event['event_id']))
    by_id = {event['event_id']: event for event in ordered}
    result = {event['event_id']: [] for event in ordered}
    for suggestion in suggestions:
        evidence = [by_id[event_id] for event_id in suggestion['payload'].get('evidence_event_ids', [])
                    if event_id in by_id]
        if not evidence:
            continue
        arrival = suggestion.get('created_ms')
        if arrival is not None:
            completed = [event for event in evidence if event['end_ms'] <= arrival]
            anchor = (max(completed, key=lambda event: (event['end_ms'], event['start_ms']))
                      if completed else min(evidence, key=lambda event: (event['start_ms'], event['end_ms'])))
        else:
            anchor = min(evidence, key=lambda event: (event['start_ms'], event['end_ms']))
        result[anchor['event_id']].append(suggestion)
    return result


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


def live_chart_data(windows, notes, duration, *, start_ms=0, display_score=None):
    """실시간 차트의 시간 경계, 점수, 관측 불가 구간과 메모를 만든다."""
    start_ms = max(0, min(start_ms, duration))
    rows = [{'kind': 'bound', 'time': start_ms / 1000},
            {'kind': 'bound', 'time': max(duration / 1000, 1)}]
    ordered = sorted((e for e in windows if start_ms <= e['end_ms'] <= duration),
                     key=lambda e: (e['end_ms'], e.get('created_ms') or 0, e['event_id']))
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
        if end >= start_ms and start <= duration:
            rows.append({'kind': 'unobservable', 'time': max(start, start_ms) / 1000,
                         'end': min(end, duration) / 1000})
    for event in notes:
        if start_ms <= event['start_ms'] <= duration:
            rows.append({'kind': 'note', 'time': event['start_ms'] / 1000, 'value': -12,
                         'time_label': clock(event['start_ms']), 'text': event['payload']['text']})
    if display_score and display_score.get('state') == 'held' and display_score.get('value') is not None:
        rows.append({'kind': 'held', 'time': duration / 1000, 'value': display_score['value'],
                     'time_label': clock(duration),
                     'text': f"이전 점수 유지 · {display_score['age_ms']/1000:.1f}초 전 측정"})
    elif display_score and display_score.get('state') == 'current' and display_score.get('value') is not None:
        rows.append({'kind': 'current', 'time': duration / 1000, 'value': display_score['value'],
                     'time_label': clock(duration), 'text': '점수 카드와 같은 최신 구간 점수'})
    return pd.DataFrame(rows, columns=['kind', 'time', 'end', 'value', 'sequence', 'time_label', 'text'])


def score_band(value):
    """점수를 경고 기준과 무관한 고정 표시 구간으로 분류한다."""
    if value is None:
        return 'unknown'
    if value < 60:
        return 'red'
    return 'yellow' if value < 80 else 'green'


def live_chart_spec(*, low_score=False):
    """데이터와 분리된 고정 차트 설정으로 실시간 갱신 시 요소를 유지한다."""
    time_axis = {
        'title': None, 'grid': False, 'tickCount': 7,
        'labelExpr': "(floor(datum.value / 3600) < 10 ? '0' : '') + floor(datum.value / 3600) + ':' + (floor(datum.value / 60) % 60 < 10 ? '0' : '') + floor(datum.value / 60) % 60 + ':' + (floor(datum.value) % 60 < 10 ? '0' : '') + floor(datum.value) % 60",
    }
    x = {'field': 'time', 'type': 'quantitative', 'scale': {'zero': False, 'nice': False}, 'axis': time_axis}
    y = {'field': 'value', 'type': 'quantitative', 'scale': {'domain': [-20, 105]},
         'axis': {'title': None, 'values': [0, 50, 100], 'gridColor': '#e6e9ed'}}
    line_color = '#c62828' if low_score else '#2457d6'
    return {
        'height': 175, 'background': 'white',
        'layer': [
            {'transform': [{'filter': "datum.kind === 'bound'"}],
             'mark': {'type': 'point', 'opacity': 0}, 'encoding': {'x': x}},
            {'transform': [{'filter': "datum.kind === 'unobservable'"}],
             'mark': {'type': 'rect', 'color': '#e6e9ed', 'opacity': 0.7},
             'encoding': {'x': x, 'x2': {'field': 'end'}}},
            {'transform': [{'filter': "datum.kind === 'score'"}],
             'mark': {'type': 'line', 'color': line_color, 'strokeWidth': 2,
                      'point': {'size': 16}, 'invalid': 'break-paths-show-domains'},
             'encoding': {'x': x, 'y': y, 'order': {'field': 'sequence', 'type': 'quantitative'},
                          'tooltip': [{'field': 'time_label', 'type': 'nominal', 'title': '수업 시간'},
                                      {'field': 'value', 'type': 'quantitative', 'title': '관찰 점수'}]}},
            {'transform': [{'filter': "datum.kind === 'current'"}],
             'mark': {'type': 'point', 'filled': True, 'color': line_color, 'size': 90},
             'encoding': {'x': x, 'y': y,
                          'tooltip': [{'field': 'time_label', 'type': 'nominal', 'title': '표시 시각'},
                                      {'field': 'value', 'type': 'quantitative', 'title': '현재 표시 점수'},
                                      {'field': 'text', 'type': 'nominal', 'title': '상태'}]}},
            {'transform': [{'filter': "datum.kind === 'held'"}],
             'mark': {'type': 'point', 'shape': 'diamond', 'filled': True, 'color': '#b26a00', 'size': 90},
             'encoding': {'x': x, 'y': y,
                          'tooltip': [{'field': 'time_label', 'type': 'nominal', 'title': '표시 시각'},
                                      {'field': 'value', 'type': 'quantitative', 'title': '이전 점수'},
                                      {'field': 'text', 'type': 'nominal', 'title': '상태'}]}},
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

    def delete(self, sid):
        """복구본까지 삭제한다. 복구본 삭제 실패 시 일반 기록은 유지한다."""
        self.clear_recovery(sid)
        self.path(sid).unlink(missing_ok=True)

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
