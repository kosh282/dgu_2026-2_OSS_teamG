"""Gaussian이 사용하는 각도 검증과 방향 상태 분류."""

import math

from .config import ModelConfig
from .input_adapter import HeadAngles


def _number(value, name):
    """참/거짓 값과 결측·무한대를 제외하고 계산 가능한 숫자만 받습니다."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}: 유한한 숫자가 필요합니다")
    return float(value)


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
