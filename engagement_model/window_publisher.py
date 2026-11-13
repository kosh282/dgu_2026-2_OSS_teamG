"""실험 모델의 시점별 결과를 공통 계약의 시간 구간 이벤트로 변환한다."""

from collections import deque

from .gaussian import gaussian_window_score


class WindowPublisher:
    """관측이 끊긴 시간은 결측으로 두고 시간 가중 구간 점수를 발행한다.

    Gaussian 계산에 필요한 값만 이동 구간 동안 메모리에 유지하고 저장하지 않는다.
    임시 추적 정보는 보관하지 않는다. 각 시점의 점수는 다음 관측이나
    최대 관측 유지 시간까지만 유효하며, 이후는 관측 불가로 처리한다.
    """

    def __init__(self, service, session_id, runtime, *, start_ms=0):
        self.service = service
        self.session_id = session_id
        self.runtime = runtime
        self.config = runtime.config
        self.model_id = runtime.model_id
        self.model_version = runtime.model_version
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
        reset_reasons = {'tracking_key_changed', 'baseline_unavailable', 'non_monotonic_timestamp'}
        single_reset = (not callable(getattr(self.runtime, 'aggregate_window', None))
                        and bool(set(result.get('reason_codes', ())) & reset_reasons))
        if (result.get('status') == 'input_error' or single_reset
                or 'position_association_reset' in result.get('reason_codes', ())):
            # 재보정/대상 변경 전후의 상대각을 같은 통계에 섞지 않는다.
            self._segments.clear()
        self._sample = (timestamp, result.get('angles') is not None, result.get('score'),
                        result.get('gaussian_observation'))
        return result

    def _extend(self, end):
        if self._sample is not None:
            timestamp, observable, score, primitive = self._sample
            start = max(self._cursor, timestamp)
            stop = min(end, timestamp + self.hold_ms)
            if stop > start:
                self._segments.append((start, stop, observable, score, primitive))
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
        aggregate = getattr(self.runtime, 'aggregate_window', None)
        # 개인별 집계 또는 단일 Gaussian 적분을 한 번만 수행한다.
        summary = (aggregate(self._segments, start, end) if callable(aggregate) else
                   gaussian_window_score(self._segments, start, end, self.config))
        observable_ms = summary['observable_duration_ms']
        scoreable_ms = summary['scoreable_duration_ms']
        observable_ratio = observable_ms / duration
        scoreable_ratio = scoreable_ms / duration
        value, reasons = summary['score'], summary['reason_codes']
        quality = 'not_observable' if value is None else ('partial' if partial or scoreable_ms < duration else 'valid')
        payload = {
            'model_output': None if value is None else {
                'kind': 'continuous', 'value': value, 'scale_or_labels': '0-100',
            },
            # 참가자 수는 알 수 없으므로 공통 계약의 분모를 임의로 채우지 않는다.
            'observation': {'valid_count': None, 'eligible_count': None,
                            'observable_ratio': None, 'count_unit': None},
            'signals': {},
            'model_extensions': {self.model_id: {
                'version': 'window-1', 'model_config_version': self.config.config_version,
                'observable_duration_ms': observable_ms,
                'scoreable_duration_ms': scoreable_ms, 'window_duration_ms': duration,
                'observable_time_ratio': round(observable_ratio, 4),
                'scoreable_time_ratio': round(scoreable_ratio, 4), 'partial_window': partial,
                'scoring_method': self.config.scoring_method,
            }},
        }
        if self.config.scoring_method == 'gaussian':
            payload['model_extensions'][self.model_id].update({
                'version': 'angle-time-window-4',
                'angle_mode': self.config.gaussian_angle_mode,
                'pitch_deadband_degrees': self.config.gaussian_pitch_deadband_degrees,
                'pitch_small_angle_scale': self.config.gaussian_pitch_small_angle_scale,
                'pitch_full_scale_start_degrees': self.config.front_pitch_degrees,
                'yaw_deadband_degrees': self.config.yaw_neutral_deadband_degrees,
                'yaw_small_angle_scale': self.config.yaw_small_angle_scale,
                'yaw_full_scale_start_degrees': self.config.yaw_full_scale_start_degrees,
                'sigma_pitch_degrees': self.config.gaussian_sigma_pitch_degrees,
                'sigma_yaw_degrees': self.config.gaussian_sigma_yaw_degrees,
                'down_transition_pitch_degrees': self.config.gaussian_down_transition_pitch_degrees,
                'down_confirm_seconds': self.config.gaussian_down_confirm_seconds,
                'down_grace_seconds': self.config.down_grace_seconds,
                'down_decay_seconds': self.config.down_decay_seconds,
                'down_penalty': self.config.down_penalty,
                'window_seconds': self.config.window_seconds,
                'temporal_weighting': 'linear-time-integral',
                'max_observation_hold_seconds': self.config.max_observation_hold_seconds,
                'min_observable_ratio': self.config.min_observable_ratio,
                'min_scoreable_ratio': self.config.min_scoreable_ratio,
            })
        if callable(aggregate):
            payload['model_extensions'][self.model_id].update({
                'aggregation': getattr(self.runtime, 'aggregation_label', 'reidentified-person-time-weighted-gaussian'),
                'scoreable_person_duration_ms': summary['scoreable_person_duration_ms'],
            })
        if getattr(self.runtime, 'eye_fusion', False):
            payload['model_extensions'][self.model_id].update({
                'version': 'head-eye-time-window-1',
                'head_weight': 0.8, 'eye_weight': 0.2,
                'eye_window_seconds': 10, 'eye_min_coverage': 0.7,
                'eye_closed_ear_threshold': 0.21, 'eye_prolonged_seconds': 2,
                'eye_score_policy': 'prolonged_bilateral_closure_observable_ratio',
            })
            if getattr(self.runtime, 'eye_fusion', False):
                payload['model_extensions'][self.model_id].update({
                    'head_weight': 0.8, 'eye_weight': 0.2,
                    'eye_window_seconds': 10, 'eye_min_coverage': 0.7,
                    'eye_closed_ear_threshold': 0.21, 'eye_prolonged_seconds': 2,
                    'eye_score_policy': 'prolonged_bilateral_closure_observable_ratio',
                })
        event = self.service.make_event(
            self.session_id, 'engagement_window', payload, start, end, 'model',
            quality_status=quality, reason_codes=reasons,
            model_id=self.model_id, model_version=self.model_version,
        )
        self.service.receive_event(event)
        self._last_end = end
