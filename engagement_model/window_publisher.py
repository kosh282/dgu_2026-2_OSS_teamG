"""실험 모델의 시점별 결과를 공통 계약의 시간 구간 이벤트로 변환한다."""

from collections import deque

from .runtime import MODEL_ID, MODEL_VERSION


class WindowPublisher:
    """관측이 끊긴 시간은 결측으로 두고 시간 가중 구간 점수를 발행한다.

    특징과 임시 추적 정보는 보관하지 않는다. 각 시점의 점수는 다음 관측이나
    최대 관측 유지 시간까지만 유효하며, 이후는 관측 불가로 처리한다.
    """

    def __init__(self, service, session_id, runtime, *, start_ms=0):
        self.service = service
        self.session_id = session_id
        self.runtime = runtime
        self.config = runtime.config
        self.window_ms = round(self.config.window_seconds * 1000)
        self.interval_ms = round(self.config.output_interval_seconds * 1000)
        self.hold_ms = round(self.config.max_observation_hold_seconds * 1000)
        if min(self.window_ms, self.interval_ms, self.hold_ms) < 1:
            raise ValueError('구간, 출력 간격, 관측 유지 시간은 1ms 이상이어야 합니다')
        self._segments = deque()
        self._start_ms = start_ms
        self._cursor = start_ms
        self._sample = None
        self._next_end = start_ms + self.window_ms
        self._last_end = start_ms

    def __call__(self, feature_window):
        timestamp = feature_window['start_ms']
        self.advance(timestamp)
        result = self.runtime(feature_window)
        self._sample = (timestamp, result.get('angles') is not None, result.get('score'))
        return result

    def _extend(self, end):
        if self._sample is not None:
            timestamp, observable, score = self._sample
            start = max(self._cursor, timestamp)
            stop = min(end, timestamp + self.hold_ms)
            if stop > start:
                self._segments.append((start, stop, observable, score))
        self._cursor = end

    def advance(self, now_ms):
        """새 관측이 없어도 마감된 구간을 발행한다."""
        if now_ms < self._cursor:
            raise ValueError('모델 관측 시각은 역행할 수 없습니다')
        while self._next_end <= now_ms:
            self._extend(self._next_end)
            self._publish(self._next_end, partial=False)
            self._next_end += self.interval_ms
        self._extend(now_ms)
        cutoff = now_ms - self.window_ms
        while self._segments and self._segments[0][1] <= cutoff:
            self._segments.popleft()

    def finish(self, now_ms):
        """마지막 정기 출력 이후 구간이 있으면 세션 종료 전에 발행한다."""
        self.advance(now_ms)
        if now_ms > self._last_end:
            self._publish(now_ms, partial=now_ms - self._start_ms < self.window_ms)

    def _publish(self, end, partial):
        start = max(self._start_ms, end - self.window_ms)
        duration = end - start
        observable_ms = scoreable_ms = weighted = 0
        for left, right, observable, score in self._segments:
            elapsed = max(0, min(end, right) - max(start, left))
            if observable:
                observable_ms += elapsed
            if score is not None:
                scoreable_ms += elapsed
                weighted += elapsed * score
        observable_ratio = observable_ms / duration
        scoreable_ratio = scoreable_ms / duration
        reasons = []
        if observable_ratio < self.config.min_observable_ratio:
            reasons.append('insufficient_observation')
        if scoreable_ratio < self.config.min_scoreable_ratio:
            reasons.append('insufficient_scoreable_time')
        if not scoreable_ms:
            reasons.append('no_scoreable_time')
        value = round(weighted / scoreable_ms, 1) if not reasons else None
        quality = 'not_observable' if value is None else ('partial' if partial or scoreable_ms < duration else 'valid')
        payload = {
            'model_output': None if value is None else {
                'kind': 'continuous', 'value': value, 'scale_or_labels': '0-100',
            },
            # 참가자 수는 알 수 없으므로 공통 계약의 분모를 임의로 채우지 않는다.
            'observation': {'valid_count': None, 'eligible_count': None,
                            'observable_ratio': None, 'count_unit': None},
            'signals': {},
            'model_extensions': {MODEL_ID: {
                'version': 'window-1', 'model_config_version': self.config.config_version,
                'observable_duration_ms': observable_ms,
                'scoreable_duration_ms': scoreable_ms, 'window_duration_ms': duration,
                'observable_time_ratio': round(observable_ratio, 4),
                'scoreable_time_ratio': round(scoreable_ratio, 4), 'partial_window': partial,
            }},
        }
        event = self.service.make_event(
            self.session_id, 'engagement_window', payload, start, end, 'model',
            quality_status=quality, reason_codes=reasons,
            model_id=MODEL_ID, model_version=MODEL_VERSION,
        )
        self.service.receive_event(event)
        self._last_end = end
