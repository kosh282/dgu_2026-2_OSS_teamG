"""모든 기본 수치는 검증 전 실험 후보이며 집중도를 보장하지 않습니다."""

from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class ModelConfig:
    """각도는 도, 시간은 초입니다. 변경한 설정은 별도 버전으로 기록하세요."""

    config_version: str = "experimental-0.4"
    feature_version: str = "experimental-0.8"
    window_seconds: float = 10.0
    output_interval_seconds: float = 2.0
    front_yaw_degrees: float = 15.0
    yaw_neutral_deadband_degrees: float = 2.0
    yaw_small_angle_scale: float = 0.75
    yaw_full_scale_start_degrees: float = 15.0
    front_pitch_degrees: float = 10.0
    away_yaw_degrees: float = 30.0
    full_away_yaw_degrees: float = 60.0
    away_up_degrees: float = 20.0
    full_away_up_degrees: float = 40.0
    down_pitch_degrees: float = 15.0
    down_exit_pitch_degrees: float = 10.0
    down_grace_seconds: float = 5.0
    down_decay_seconds: float = 10.0
    down_penalty: float = 20.0
    away_grace_seconds: float = 3.0
    full_away_seconds: float = 10.0
    recovery_seconds: float = 2.0
    front_full_seconds: float = 5.0
    max_observation_hold_seconds: float = 0.3
    min_observable_ratio: float = 0.7
    min_scoreable_ratio: float = 0.7
    front_base_score: float = 90.0
    front_bonus: float = 10.0
    away_base_score: float = 70.0
    angle_penalty: float = 30.0
    duration_penalty: float = 40.0

    def __post_init__(self):
        """잘못된 설정으로 계산이 실패하지 않도록 자료형과 값의 순서를 확인합니다."""
        # Python은 참/거짓 값도 숫자로 취급하지만, 실험 설정에서는 이를 거부합니다.
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in {"config_version", "feature_version"}:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{item.name}: 비어 있지 않은 문자열이 필요합니다")
            elif (isinstance(value, bool) or not isinstance(value, (int, float))
                  or not math.isfinite(value) or value < 0):
                raise ValueError(f"{item.name}: 유한한 0 이상의 숫자가 필요합니다")
        for name in ("window_seconds", "output_interval_seconds", "recovery_seconds",
                     "front_full_seconds", "max_observation_hold_seconds"):
            if getattr(self, name) == 0:
                raise ValueError(f"{name}: 0보다 커야 합니다")
        if not (self.front_yaw_degrees < self.away_yaw_degrees
                < self.full_away_yaw_degrees <= 180):
            raise ValueError("좌우 각도는 정면 < 이탈 < 최대 이탈 <= 180 이어야 합니다")
        if not self.yaw_neutral_deadband_degrees < self.yaw_full_scale_start_degrees <= self.away_yaw_degrees:
            raise ValueError("yaw 보정 구간은 데드밴드보다 크고 이탈 임계값 이하여야 합니다")
        if not 0 < self.yaw_small_angle_scale <= 1:
            raise ValueError("yaw 민감도 계수는 0보다 크고 1 이하여야 합니다")
        if not (self.front_pitch_degrees < self.away_up_degrees
                < self.full_away_up_degrees <= 180):
            raise ValueError("위쪽 각도는 정면 < 이탈 < 최대 이탈 <= 180 이어야 합니다")
        if not self.front_pitch_degrees < self.down_pitch_degrees <= 180:
            raise ValueError("아래 보기 각도는 정면 범위 밖이어야 합니다")
        if not 0 <= self.down_exit_pitch_degrees < self.down_pitch_degrees:
            raise ValueError("아래 보기 종료 각도는 시작 각도보다 작아야 합니다")
        if self.down_decay_seconds == 0:
            raise ValueError("down_decay_seconds: 0보다 커야 합니다")
        if self.down_penalty > 100:
            raise ValueError("down_penalty는 100 이하여야 합니다")
        if self.full_away_seconds <= self.away_grace_seconds:
            raise ValueError("최대 이탈 시간은 유예 시간보다 커야 합니다")
        if self.output_interval_seconds > self.window_seconds:
            raise ValueError("출력 간격은 집계 구간 이하여야 합니다")
        if max(self.min_observable_ratio, self.min_scoreable_ratio) > 1:
            raise ValueError("관측률은 0~1이어야 합니다")
        if self.front_base_score + self.front_bonus > 100 or self.away_base_score > 100:
            raise ValueError("기준 행동 점수는 0~100 범위여야 합니다")
