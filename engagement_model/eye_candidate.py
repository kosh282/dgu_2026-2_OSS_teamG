"""눈 관측 점수와 머리 점수의 실험 결합. 개인별 값은 최종 이벤트에 저장하지 않는다."""
from collections import deque
import math


class EyeCandidate:
    """10초 관측 중 2초를 넘긴 연속 감김만 계산하는 검증 전 실험."""
    def __init__(self, window_ms=10000, max_gap_ms=300, min_coverage=0.7,
                 closed_threshold=0.21, prolonged_ms=2000):
        self.window_ms, self.max_gap_ms = window_ms, max_gap_ms
        self.min_coverage, self.threshold, self.prolonged_ms = min_coverage, closed_threshold, prolonged_ms
        self.reset()

    def reset(self):
        self.segments = deque()
        self.previous_time = None
        self.previous_closed = False
        self.closed_since = None
        self.track = None

    def update(self, feature, head_result):
        timestamp = feature.get('start_ms')
        faces = feature.get('features', {}).get('faces', [])
        face = faces[0] if len(faces) == 1 else {}
        track = face.get('transient_track_key')
        quality = face.get('quality') or {}
        left, right = face.get('eye_aspect_ratio_left'), face.get('eye_aspect_ratio_right')
        values_ok = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                        and math.isfinite(v) and v >= 0 for v in (left, right))
        usable = (isinstance(timestamp, int) and not isinstance(timestamp, bool)
                  and isinstance(track, int) and not isinstance(track, bool)
                  and feature.get('quality', {}).get('status') in ('valid', 'partial')
                  and quality.get('status') in ('valid', 'partial') and values_ok
                  and head_result.get('angles') is not None)
        if not usable:
            self.previous_time = self.closed_since = None
            self.previous_closed = False
            return {'eye_score': None, 'candidate_total': None, 'head_score': head_result.get('score'),
                    'coverage': None, 'closed_seconds': None, 'eye_state': 'unavailable',
                    'continuous_closed_seconds': None, 'status': 'eye_unavailable'}
        if self.track is not None and self.track != track:
            self.reset()
        self.track = track
        closed = left < self.threshold and right < self.threshold
        previous = self.previous_time
        if previous is not None and 0 < timestamp - previous <= self.max_gap_ms:
            prolonged_start = (max(previous, self.closed_since + self.prolonged_ms)
                               if self.previous_closed and closed and self.closed_since is not None else timestamp)
            self.segments.append((previous, timestamp, max(0, timestamp - prolonged_start)))
        else:
            self.closed_since = None
        if closed:
            if self.closed_since is None:
                self.closed_since = timestamp
        else:
            self.closed_since = None
        self.previous_time, self.previous_closed = timestamp, closed
        cutoff = timestamp - self.window_ms
        while self.segments and self.segments[0][1] <= cutoff:
            self.segments.popleft()
        observed = prolonged = 0.0
        for start, end, closed_ms in self.segments:
            clipped_start = max(start, cutoff)
            duration = max(0, end - clipped_start)
            observed += duration
            prolonged += min(duration, closed_ms)
        coverage = min(1.0, observed / self.window_ms)
        eye_score = round(100 * (1 - prolonged / observed), 1) if coverage >= self.min_coverage and observed else None
        head = head_result.get('score')
        total = round(0.8 * head + 0.2 * eye_score, 1) if head is not None and eye_score is not None else None
        return {'eye_score': eye_score, 'candidate_total': total, 'head_score': head,
                'coverage': round(coverage, 3), 'closed_seconds': round(prolonged / 1000, 2),
                'eye_state': 'closed_candidate' if closed else 'open_candidate',
                'continuous_closed_seconds': round((timestamp - self.closed_since) / 1000, 2)
                if self.closed_since is not None else 0.0,
                'status': 'candidate' if total is not None else 'collecting'}
