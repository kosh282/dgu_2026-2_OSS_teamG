"""독립 웹캠용 머리·눈 결합 런타임. 각도 원본은 메모리에만 남는다."""
from collections import deque
from dataclasses import replace
from .gaussian import GaussianRuntime, gaussian_window_score


class EyeFusionGaussianRuntime(GaussianRuntime):
    eye_fusion = True
    def __init__(self, config=None, **kwargs):
        self._fusion_segments = deque()
        self._fusion_sample = None
        self._fusion_start = None
        super().__init__(config, **kwargs)
        self.config = replace(self.config, config_version=self.config.config_version + '+eye-fusion')
        self.model_version = 'angle-time-0.5+eye'
        self.latest = self._empty_result('waiting_for_observation')

    def _reset_time_state(self, **kwargs):
        super()._reset_time_state(**kwargs)
        self._fusion_segments.clear()
        self._fusion_sample = None
        self._fusion_start = None

    def __call__(self, feature):
        result = super().__call__(feature)
        if result.get('status') == 'input_error':
            return result
        timestamp = feature['start_ms']
        if self._fusion_start is None:
            self._fusion_start = timestamp
        if self._fusion_sample is not None:
            before, observable, primitive = self._fusion_sample
            end = min(timestamp, before + round(self.config.max_observation_hold_seconds * 1000))
            if end > before:
                self._fusion_segments.append((before, end, observable, None, primitive))
        head = result.get('current_direction_score')
        eye = (result.get('eye_candidate') or {}).get('eye_score')
        combined = 0.8 * head + 0.2 * eye if head is not None and eye is not None else None
        primitive = (None, None, combined) if combined is not None else None
        observable = result.get('angles') is not None
        self._fusion_sample = (timestamp, observable, primitive)
        start = max(self._fusion_start, timestamp - round(self.config.window_seconds * 1000))
        while self._fusion_segments and self._fusion_segments[0][1] <= start:
            self._fusion_segments.popleft()
        summary = (gaussian_window_score(self._fusion_segments, start, timestamp, self.config)
                   if timestamp > start else {'score': None, 'reason_codes': ['gaussian_warmup']})
        # 현재 관측을 못하면 오래된 값은 현재값으로 내보내지 않는다.
        score = summary['score'] if combined is not None else None
        result['head_direction_score'] = head
        result['gaussian_observation'] = primitive
        result['current_direction_score'] = round(combined, 1) if combined is not None else None
        result['score'] = score
        result['status'] = 'scored' if score is not None else 'deferred'
        exclude = {'insufficient_observation', 'insufficient_scoreable_time', 'no_scoreable_time', 'gaussian_warmup'}
        reasons = [reason for reason in result.get('reason_codes', []) if reason not in exclude]
        if eye is None:
            reasons.append('eye_observation_unavailable')
        result['reason_codes'] = list(dict.fromkeys(reasons + summary['reason_codes']))
        if score is None:
            self._last_display_score = self._last_display_score_ms = None
        result.update(self._display_fields(timestamp, score))
        return result
