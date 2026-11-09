"""기존 실험 점수와 Huang(2024) 원식 및 온라인 변형. 시간 상태는 runtime이 관리한다."""

import math
from statistics import median

from .config import ModelConfig
from .input_adapter import HeadAngles, Observation


def huang_head_up(mean_pitch, pitches, reference_pitch, amplitude_degrees=5.0, *,
                  neutral_on_degenerate=False):
    """원문 식 (1)~(3). 위쪽 양수, 도 단위. reference_pitch는 호출자가 제공한다.

    원식에 정의되지 않은 분모 0은 None으로 보류한다. 시퀀스 진폭 차이는
    max-min으로 해석한다. 온라인 호출자는 전체 영상 중앙값 대신 보정 정면 0을 쓴다.
    온라인 전용 neutral_on_degenerate=True는 p=q인 퇴화 구간만 P=.5로 확장한다.
    이는 논문에 없는 중립 처리이며 분모에 임의 epsilon을 더하지 않는다.
    """
    p = _number(mean_pitch, "mean_pitch")
    reference = _number(reference_pitch, "reference_pitch")
    threshold = _number(amplitude_degrees, "amplitude_degrees")
    if threshold < 0:
        raise ValueError("amplitude_degrees는 0 이상이어야 합니다")
    values = [_number(value, "pitch") for value in pitches]
    if not values:
        return None
    low, high = min(values), max(values)
    q = median(values) if high - low >= threshold else reference
    if not math.isfinite(q):
        return None
    difference = p - q
    denominator = (high if p >= q else low) - q
    if not math.isfinite(difference) or not math.isfinite(denominator):
        return None
    if denominator == 0:
        return 0.5 if neutral_on_degenerate and p == q else None
    a = abs(difference / denominator)
    z = a * difference
    # 같은 sigmoid를 부호별로 계산해 큰 각도에서 exp overflow를 피한다.
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1 + exp_z)


def huang_concentration(head_up, front_facing):
    """원문 식 (7). 두 입력은 백분율이 아닌 0~1 비율이며 출력도 0~1이다."""
    p, y = _number(head_up, "head_up"), _number(front_facing, "front_facing")
    if not (0 <= p <= 1 and 0 <= y <= 1):
        raise ValueError("Huang 결합 입력은 0~1이어야 합니다")
    return 1 - math.hypot(y - 1, p - 1) / math.sqrt(2)


def huang_front_facing(yaw, head_x, head_y, width, max_yaw=90.0):
    """원문 식 (4)~(6)의 비율 표현. 교실 공간좌표용이며 타일 좌표에 적용하지 않는다.

    원식의 부호/범위는 그대로 반환한다. 온라인 모델은 별도로 상대 yaw와 clip을 쓴다.
    """
    yaw = _number(yaw, "yaw")
    x, y = _number(head_x, "head_x"), _number(head_y, "head_y")
    width, maximum = _number(width, "width"), _number(max_yaw, "max_yaw")
    if y <= 0 or width <= 0 or maximum <= 0:
        raise ValueError("head_y, width, max_yaw는 양수여야 합니다")
    cs = math.degrees(math.atan(abs(x - width / 2) / y))
    beta = abs(yaw) - cs if x >= width / 2 else abs(yaw) + cs
    return 1 - beta / maximum


def huang_window_score(segments, start, end, config):
    """온라인 변형: 관측 시간으로 가중한 작은 구간 P/Y를 결합한 뒤 C를 한 번 계산한다.

    segments: (시작, 끝, 관측 가능, 기존 점수, (위쪽 양수 pitch, 상대 yaw) 또는 None).
    하향 보류는 primitive=None으로 제외한다. 입력은 메모리 전용이며 저장하지 않는다.
    """
    duration = end - start
    if duration <= 0:
        raise ValueError("집계 구간은 양수여야 합니다")
    clipped = [(max(start, a), min(end, b), observable, primitive)
               for a, b, observable, _, primitive in segments if a < end and b > start]
    observable_ms = sum(b - a for a, b, observable, _ in clipped if observable)
    pitches = [primitive[0] for _, _, _, primitive in clipped if primitive is not None]
    small_ms = round(config.huang_small_seconds * 1000)
    scoreable_ms = weighted_p = weighted_y = 0
    undefined_ms = 0
    # 끝의 미완료 작은 구간은 새 점수로 위장하지 않고 제외한다.
    for left in range(start, end - small_ms + 1, small_ms):
        right = left + small_ms
        rows = [(max(0, min(b, right) - max(a, left)), primitive)
                for a, b, _, primitive in clipped if primitive is not None]
        rows = [(elapsed, primitive) for elapsed, primitive in rows if elapsed]
        elapsed = sum(ms for ms, _ in rows)
        if elapsed / small_ms < config.min_scoreable_ratio or not elapsed:
            continue
        mean_pitch = sum(ms * primitive[0] for ms, primitive in rows) / elapsed
        p = huang_head_up(mean_pitch, pitches, 0.0, config.huang_amplitude_degrees,
                          neutral_on_degenerate=True)
        if p is None:
            undefined_ms += elapsed
            continue
        y = sum(ms * _clip(1 - abs(primitive[1]) / config.huang_max_yaw_degrees)
                for ms, primitive in rows) / elapsed
        scoreable_ms += elapsed
        weighted_p += elapsed * p
        weighted_y += elapsed * y
    reasons = []
    if duration < small_ms:
        reasons.append("huang_warmup")
    if undefined_ms:
        reasons.append("huang_degenerate_pitch_range")
    if observable_ms / duration < config.min_observable_ratio:
        reasons.append("insufficient_observation")
    if scoreable_ms / duration < config.min_scoreable_ratio:
        reasons.append("insufficient_scoreable_time")
    if not scoreable_ms:
        reasons.append("no_scoreable_time")
    # 유효한 0~1 지표의 가중 평균에 생기는 부동소수점 경계 오차만 제한한다.
    p = _clip(weighted_p / scoreable_ms) if scoreable_ms else None
    y = _clip(weighted_y / scoreable_ms) if scoreable_ms else None
    # 일부 퇴화 구간은 비율 gate로 처리하며 전체 관측이 실패한 것처럼 만들지 않는다.
    blocked = any(reason != "huang_degenerate_pitch_range" for reason in reasons)
    return dict(score=round(100 * huang_concentration(p, y), 1) if not blocked else None,
                observable_duration_ms=observable_ms, scoreable_duration_ms=scoreable_ms,
                reason_codes=reasons, head_up_rate=p, front_facing_rate=y)


def _number(value, name):
    """참/거짓 값과 결측·무한대를 제외하고 계산 가능한 숫자만 받습니다."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}: 유한한 숫자가 필요합니다")
    return float(value)


def _clip(value, low=0.0, high=1.0):
    """계산값을 지정된 범위 안으로 제한합니다."""
    return max(low, min(high, value))


def _validate_angles(angles):
    """직접 함수를 호출할 때도 잘못된 각도가 점수로 바뀌지 않게 검사합니다."""
    if not isinstance(angles, HeadAngles):
        raise ValueError("HeadAngles가 필요합니다")
    for axis in ("pitch", "yaw", "roll"):
        if not -180 <= _number(getattr(angles, axis), axis) <= 180:
            raise ValueError("상대 각도는 -180~180도여야 합니다")


def classify_orientation(angles, config=None, *, previous_state=None):
    """한 시점의 자세 후보를 반환합니다. 지속적인 비참여 판정은 아닙니다."""
    config = config or ModelConfig()
    if angles is None:
        return "not_observable"
    _validate_angles(angles)
    # 아래 보기를 먼저 분리하므로 아래+옆 보기 역시 즉시 감점하지 않습니다.
    if angles.pitch >= config.down_pitch_degrees:
        return "looking_down"
    # 시작/종료 각도를 분리해 경계 부근의 흔들림으로 아래 보기 시간이 끊기지 않게 한다.
    if (previous_state == "looking_down"
            and angles.pitch > config.down_exit_pitch_degrees):
        return "looking_down"
    if abs(angles.yaw) <= config.front_yaw_degrees and abs(angles.pitch) <= config.front_pitch_degrees:
        return "front"
    if abs(angles.yaw) >= config.away_yaw_degrees or angles.pitch <= -config.away_up_degrees:
        return "away"
    return "ambiguous"


def orientation_severity(angles, config=None):
    """좌우·위쪽 이탈 강도를 0~1로 반환하고 아래 보기·결측은 None으로 둡니다."""
    config = config or ModelConfig()
    state = classify_orientation(angles, config)
    if state in {"looking_down", "not_observable"}:
        return None
    yaw = _clip((abs(angles.yaw) - config.away_yaw_degrees)
                / (config.full_away_yaw_degrees - config.away_yaw_degrees))
    up = _clip((-angles.pitch - config.away_up_degrees)
               / (config.full_away_up_degrees - config.away_up_degrees))
    return max(yaw, up)


def behavior_score(observation: Observation, *, front_seconds=0.0, away_seconds=0.0,
                   looking_down_seconds=0.0, down_start_score=None,
                   previous_stable_score=None, previous_orientation_state=None,
                   config=None):
    """관측과 연속 유지 시간을 받아 0~100점 또는 None을 반환합니다.

    시간은 호출자가 실제 경과 시간으로 계산해야 합니다. 이 함수는 상태를
    기억하지 않으며, 복귀 확인·10초 집계·2초 출력은 아직 수행하지 않습니다.
    아래 보기는 5초 유예 후 10초 동안 시작 점수에서 최대 20점을 감점합니다.
    """
    config = config or ModelConfig()
    front_seconds = _number(front_seconds, "front_seconds")
    away_seconds = _number(away_seconds, "away_seconds")
    looking_down_seconds = _number(looking_down_seconds, "looking_down_seconds")
    if min(front_seconds, away_seconds, looking_down_seconds) < 0:
        raise ValueError("유지 시간은 0 이상이어야 합니다")
    if previous_stable_score is not None:
        previous_stable_score = _number(previous_stable_score, "previous_stable_score")
        if not 0 <= previous_stable_score <= 100:
            raise ValueError("이전 점수는 0~100이어야 합니다")
    if down_start_score is not None:
        down_start_score = _number(down_start_score, "down_start_score")
        if not 0 <= down_start_score <= 100:
            raise ValueError("아래 보기 시작 점수는 0~100이어야 합니다")
    state = classify_orientation(
        observation.angles, config, previous_state=previous_orientation_state,
    )
    if state in {"not_observable", "ambiguous"}:
        return None
    if state == "looking_down":
        if down_start_score is None:
            return None
        decay = _clip(
            (looking_down_seconds - config.down_grace_seconds) / config.down_decay_seconds
        )
        return _clip(down_start_score - config.down_penalty * decay, 0, 100)
    if state == "front":
        return _clip(config.front_base_score + config.front_bonus
                     * _clip(front_seconds / config.front_full_seconds), 0, 100)
    # 짧은 이탈은 새 감점을 만들지 않습니다. 이전 안정 점수가 없으면 보류합니다.
    if away_seconds <= config.away_grace_seconds:
        return previous_stable_score
    duration = _clip((away_seconds - config.away_grace_seconds)
                     / (config.full_away_seconds - config.away_grace_seconds))
    return _clip(config.away_base_score
                 - config.angle_penalty * orientation_severity(observation.angles, config)
                 - config.duration_penalty * duration, 0, 100)
