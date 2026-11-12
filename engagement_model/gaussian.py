"""Li·Liu(2024) 기반 각도 공간·실제 시간 변형. 화면 투영 원식의 재현은 아니다."""

from collections import deque
import math

from .config import ModelConfig
from .scoring import _number, _validate_angles
from .input_adapter import HeadAngles, soften_angle
from .runtime import EngagementRuntime


def gaussian_angle_score(angles, config):
    """상대각 원점에서 100점. sigma는 도 단위 실험 설정이며 최적값이 아니다."""
    _validate_angles(angles)
    distance = math.hypot(angles.pitch / config.gaussian_sigma_pitch_degrees,
                          angles.yaw / config.gaussian_sigma_yaw_degrees)
    return 100 * math.exp(-0.5 * distance * distance)


def gaussian_score_angles(angles, config):
    """상태 판정용 상대각은 유지하고 점수 입력에만 선택적 중립 완화를 적용한다."""
    _validate_angles(angles)
    if config.gaussian_angle_mode == 'relative':
        return angles
    return HeadAngles(
        soften_angle(angles.pitch, config.gaussian_pitch_deadband_degrees,
                     config.gaussian_pitch_small_angle_scale, config.front_pitch_degrees),
        soften_angle(angles.yaw, config.yaw_neutral_deadband_degrees,
                     config.yaw_small_angle_scale, config.yaw_full_scale_start_degrees), angles.roll)


def li_liu_rank_average(scores):
    """논문 식 (10)의 기록 순서 가중평균. 입력은 10초 내 오래된 순서의 유효 점수."""
    values = [_number(score, 'score') for score in scores]
    if not values:
        return None
    if any(not 0 <= value <= 100 for value in values):
        raise ValueError('score는 0~100이어야 합니다')
    return sum(k * value for k, value in enumerate(values, 1)) / (len(values) * (len(values) + 1) / 2)


def gaussian_window_score(segments, start, end, config):
    """시간 변형: 선형 최근성 커널의 구간 적분. 표본 개수에 의존하지 않는다.

    w(t)=(t-(end-window_ms))/window_ms. 각 구간은 최대 관측 유지 시간 이내만 제공한다.
    반환값/구간 계약은 Huang 집계기와 동일하며 원시 각도는 저장 결과에 넣지 않는다.
    """
    from .input_adapter import HeadAngles

    duration = end - start
    if duration <= 0:
        raise ValueError('집계 구간은 양수여야 합니다')
    window_ms = round(config.window_seconds * 1000)
    if window_ms < 1 or duration > window_ms:
        raise ValueError('집계 구간은 설정한 시간 범위 안이어야 합니다')
    origin = end - window_ms
    observable_ms = scoreable_ms = 0
    numerator = denominator = 0.0
    for left, right, observable, _, primitive in segments:
        left, right = max(start, left), min(end, right)
        if right <= left:
            continue
        if observable:
            observable_ms += right - left
        if not observable or primitive is None:
            continue
        score = gaussian_angle_score(HeadAngles(primitive[0], primitive[1], 0), config)
        # 정규화된 시간 커널 적분: (v²-u²)/2. 뺄셈 오차를 줄이도록 인수분해한다.
        u, v = (left - origin) / window_ms, (right - origin) / window_ms
        weight = (v - u) * (v + u) / 2
        numerator += weight * score
        denominator += weight
        scoreable_ms += right - left
    reasons = []
    if observable_ms / duration < config.min_observable_ratio:
        reasons.append('insufficient_observation')
    if scoreable_ms / duration < config.min_scoreable_ratio:
        reasons.append('insufficient_scoreable_time')
    if not scoreable_ms or denominator <= 0:
        reasons.append('no_scoreable_time')
    score = min(100.0, max(0.0, numerator / denominator)) if not reasons else None
    return dict(score=round(score, 1) if score is not None else None,
                observable_duration_ms=observable_ms, scoreable_duration_ms=scoreable_ms,
                reason_codes=reasons)


class GaussianRuntime(EngagementRuntime):
    """기존 어댑터/자세/시간 상태를 재사용하며 Gaussian 점수만 별도로 계산한다."""

    def __init__(self, config=None, *, downward_policy=None, **kwargs):
        self.downward_policy = downward_policy
        config = config or ModelConfig.gaussian()
        if config.scoring_method != 'gaussian':
            raise ValueError('GaussianRuntime에는 gaussian 설정이 필요합니다')
        self._gaussian_segments = deque()
        self._gaussian_intervals = deque()
        self._gaussian_sample = None
        self._gaussian_start_ms = None
        self._down_confirm_start_ms = None
        super().__init__(config, **kwargs)
        self.model_id = 'gaussian-head-direction'
        self.model_version = 'angle-time-0.3' if config.gaussian_angle_mode == 'softened' else 'angle-time-0.2'
        self.latest = self._empty_result('waiting_for_observation')

    def _reset_time_state(self, **kwargs):
        super()._reset_time_state(**kwargs)
        self._gaussian_segments.clear()
        self._gaussian_intervals.clear()
        self._gaussian_sample = None
        self._gaussian_start_ms = None
        self._down_confirm_start_ms = None

    def _empty_result(self, reason):
        # 실행 중/미리보기 전용 값이다. JSON 이벤트의 score 의미는 변경하지 않는다.
        return {**super()._empty_result(reason), 'current_direction_score': None,
                'downward_status': None}

    def _thresholds(self):
        return {**super()._thresholds(),
                'gaussian_sigma_pitch_degrees': self.config.gaussian_sigma_pitch_degrees,
                'gaussian_sigma_yaw_degrees': self.config.gaussian_sigma_yaw_degrees,
                'gaussian_down_transition_pitch_degrees': self.config.gaussian_down_transition_pitch_degrees,
                'gaussian_down_confirm_seconds': self.config.gaussian_down_confirm_seconds,
                'gaussian_window_seconds': self.config.window_seconds}

    def sampling_diagnostics(self):
        """관측 간격은 메모리 진단값이다. regular는 허용오차 없이 정확히 같은 간격을 뜻한다."""
        intervals = [delta for _, delta in self._gaussian_intervals]
        return dict(interval_count=len(intervals),
                    min_interval_ms=min(intervals) if intervals else None,
                    max_interval_ms=max(intervals) if intervals else None,
                    regular=len(set(intervals)) == 1 if intervals else None)

    def __call__(self, feature_window):
        result = super().__call__(feature_window)
        if result['status'] == 'input_error':
            self._last_display_score = self._last_display_score_ms = None
            result.update(self._display_fields(None))
            return result
        timestamp = feature_window['start_ms']
        if self._gaussian_start_ms is None:
            self._gaussian_start_ms = timestamp
        if self._gaussian_sample is not None:
            previous, observable, primitive = self._gaussian_sample
            stop = min(timestamp, previous + round(self.config.max_observation_hold_seconds * 1000))
            if stop > previous:
                self._gaussian_segments.append((previous, stop, observable, None, primitive))
                self._gaussian_intervals.append((timestamp, timestamp - previous))
        observable = result['angles'] is not None
        angles = result.get('source_angles') or result['angles']
        # 관측 선택 정책만 변경한다. 전이는 처음부터 제외하여 발행된 과거 점수를 수정하지 않는다.
        transition = observable and angles['pitch'] > self.config.gaussian_down_transition_pitch_degrees
        down = observable and result['orientation_state'] == 'looking_down'
        if observable and angles['pitch'] >= self.config.down_pitch_degrees:
            if self._down_confirm_start_ms is None:
                self._down_confirm_start_ms = timestamp
        else:
            self._down_confirm_start_ms = None
        confirmed = (self._down_confirm_start_ms is not None and
                     timestamp - self._down_confirm_start_ms >= self.config.gaussian_down_confirm_seconds * 1000)
        result['downward_status'] = (
            'looking_down_confirmed' if confirmed else
            'looking_down_uncertain' if down else
            'transition_uncertain' if transition else None)
        if self.downward_policy is not None:
            blocked, gaze_status = self.downward_policy(feature_window, result)
            transition, down = False, observable and blocked
            result['downward_status'] = gaze_status
            if observable:
                result['orientation_state'] = ('looking_down' if blocked else
                    'head_lowered' if angles['pitch'] > self.config.gaussian_down_transition_pitch_degrees
                    else result['orientation_state'])
        primitive = None
        if observable and not (transition or down):
            score_angles = gaussian_score_angles(HeadAngles(**angles), self.config)
            primitive = (score_angles.pitch, score_angles.yaw)
        result['gaussian_observation'] = primitive
        result['current_direction_score'] = (
            round(gaussian_angle_score(HeadAngles(primitive[0], primitive[1], 0), self.config), 1)
            if primitive is not None else None
        )
        self._gaussian_sample = (timestamp, observable, primitive)
        start = max(self._gaussian_start_ms, timestamp - round(self.config.window_seconds * 1000))
        while self._gaussian_segments and self._gaussian_segments[0][1] <= start:
            self._gaussian_segments.popleft()
        while self._gaussian_intervals and self._gaussian_intervals[0][0] <= start:
            self._gaussian_intervals.popleft()
        summary = (gaussian_window_score(self._gaussian_segments, start, timestamp, self.config)
                   if timestamp > start else dict(score=None, reason_codes=['gaussian_warmup']))
        reasons = list(result['reason_codes']) + summary['reason_codes']
        if not observable:
            summary['score'] = None
        elif transition or down:
            summary['score'] = None
            reasons.append('downward_activity_uncertain' if down else 'downward_transition_uncertain')
        result['score'] = summary['score']
        result['status'] = 'scored' if result['score'] is not None else 'deferred'
        result['reason_codes'] = list(dict.fromkeys(reasons))
        if result['score'] is not None:
            self._previous_stable_score = result['score']
        else:
            # 보류 시 과거 평균을 current/held 점수로 노출하지 않는다.
            self._last_display_score = self._last_display_score_ms = None
        result['previous_stable_score'] = self._previous_stable_score
        result.update(self._display_fields(timestamp, result['score']))
        return result
