"""실험용 행동 점수입니다. 시간 버퍼와 상태 지속 판정은 후속 단계에서 연결합니다."""

import math

from .config import ModelConfig
from .input_adapter import HeadAngles, Observation


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
