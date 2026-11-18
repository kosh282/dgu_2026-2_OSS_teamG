"""Gaussian용 얼굴 방향·연속 관측 시간을 관리하는 공통 상태."""

import math

from .config import ModelConfig
from .input_adapter import parse_feature_window
from .scoring import classify_orientation


DEFAULT_DISPLAY_SCORE_HOLD_SECONDS = 2.0


class HeadOrientationState:
    """단일 얼굴 데모의 자세 유지 시간을 관리한다.

    이 실행기는 세션 이벤트를 만들거나 결과를 저장하지 않는다. 프레임별
    결과는 `latest`에만 보관해 개발자 화면에서 확인할 수 있게 한다.
    """

    def __init__(self, config=None, display_score_hold_seconds=DEFAULT_DISPLAY_SCORE_HOLD_SECONDS):
        if (isinstance(display_score_hold_seconds, bool)
                or not isinstance(display_score_hold_seconds, (int, float))
                or not math.isfinite(display_score_hold_seconds)
                or display_score_hold_seconds <= 0):
            raise ValueError("display_score_hold_seconds는 0보다 큰 유한한 숫자여야 합니다")
        self.config = config or ModelConfig.gaussian()
        self.model_id = 'gaussian-head-direction'
        self.model_version = 'angle-time-0.4'
        self.display_score_hold_seconds = float(display_score_hold_seconds)
        self._last_display_score = None
        self._last_display_score_ms = None
        self.latest = self._empty_result("waiting_for_observation")
        self._last_valid_ms = None
        self._previous_state = None
        self._front_seconds = 0.0
        self._away_seconds = 0.0
        self._down_seconds = 0.0
        self._down_start_score = None
        self._previous_stable_score = None
        self._last_track_key = None

    def reset(self):
        """새 데모 세션을 위해 자세·유지 시간 상태를 비운다."""
        self._reset_time_state()
        self._last_display_score = None
        self._last_display_score_ms = None
        self.latest = self._empty_result("waiting_for_observation")

    def _display_fields(self, timestamp_ms, current_score=None):
        """최근 점수를 화면에서만 잠시 유지한다. 점수 계산 상태는 바꾸지 않는다."""
        if current_score is not None and timestamp_ms is not None:
            self._last_display_score = round(current_score, 1)
            self._last_display_score_ms = timestamp_ms
        if self._last_display_score is None or self._last_display_score_ms is None:
            return {
                "display_score": None,
                "display_state": "unavailable",
                "display_age_ms": None,
            }
        if timestamp_ms is None or timestamp_ms < self._last_display_score_ms:
            return {
                "display_score": None,
                "display_state": "unavailable",
                "display_age_ms": None,
            }
        age_ms = timestamp_ms - self._last_display_score_ms
        if age_ms <= self.display_score_hold_seconds * 1000:
            return {
                "display_score": self._last_display_score,
                "display_state": "current" if current_score is not None else "held",
                "display_age_ms": age_ms,
            }
        return {
            "display_score": None,
            "display_state": "expired",
            "display_age_ms": age_ms,
        }

    def _empty_result(self, reason):
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "config_version": self.config.config_version,
            "status": "waiting",
            "orientation_state": "not_observable",
            "score": None,
            "reason_codes": [reason],
            "angles": None,
            "front_seconds": 0.0,
            "away_seconds": 0.0,
            "looking_down_seconds": 0.0,
            "previous_stable_score": None,
            "thresholds": self._thresholds(),
        }

    def _thresholds(self):
        """결과 로그와 개발자 화면에 실제 적용한 판정 기준을 함께 남긴다."""
        return {
            "front_pitch_degrees": self.config.front_pitch_degrees,
            "down_pitch_degrees": self.config.down_pitch_degrees,
            "away_up_degrees": self.config.away_up_degrees,
            "away_yaw_degrees": self.config.away_yaw_degrees,
            "down_grace_seconds": self.config.down_grace_seconds,
            "down_decay_seconds": self.config.down_decay_seconds,
            "down_penalty": self.config.down_penalty,
            "down_exit_pitch_degrees": self.config.down_exit_pitch_degrees,
            "yaw_neutral_deadband_degrees": self.config.yaw_neutral_deadband_degrees,
            "yaw_small_angle_scale": self.config.yaw_small_angle_scale,
            "yaw_full_scale_start_degrees": self.config.yaw_full_scale_start_degrees,
        }

    def _reset_time_state(self, *, clear_score=True):
        self._last_valid_ms = None
        self._previous_state = None
        self._front_seconds = 0.0
        self._away_seconds = 0.0
        self._down_seconds = 0.0
        self._down_start_score = None
        self._last_track_key = None
        if clear_score:
            self._previous_stable_score = None

    def __call__(self, feature_window):
        """VisionPipeline.model_consumer 콜백으로 호출한다."""
        try:
            observation = parse_feature_window(feature_window, self.config)
        except (TypeError, ValueError):
            self._reset_time_state()
            self.latest = self._empty_result("invalid_feature_window")
            self.latest["status"] = "input_error"
            timestamp_hint = (
                feature_window.get("start_ms")
                if isinstance(feature_window, dict)
                and isinstance(feature_window.get("start_ms"), int)
                and not isinstance(feature_window.get("start_ms"), bool)
                and feature_window.get("start_ms") >= 0
                else None
            )
            self.latest.update(self._display_fields(timestamp_hint))
            return self.latest

        timestamp_ms = observation.timestamp_ms
        if not observation.observable:
            # 짧은 검출 공백은 기존 상태를 잠시 보존하되 새 점수로 표시하지 않는다.
            if (self._last_valid_ms is None
                    or timestamp_ms - self._last_valid_ms
                    > self.config.max_observation_hold_seconds * 1000):
                self._reset_time_state()
            self.latest = {
                "model_id": self.model_id,
                "model_version": self.model_version,
                "config_version": self.config.config_version,
                "status": "deferred",
                "orientation_state": "not_observable",
                "score": None,
                "reason_codes": list(observation.reason_codes),
                "angles": None,
                "front_seconds": self._front_seconds,
                "away_seconds": self._away_seconds,
                "looking_down_seconds": self._down_seconds,
                "previous_stable_score": self._previous_stable_score,
                "thresholds": self._thresholds(),
            }
            self.latest.update(self._display_fields(timestamp_ms))
            return self.latest

        same_track = observation.transient_track_key == self._last_track_key
        delta_from_previous = (
            timestamp_ms - self._last_valid_ms if self._last_valid_ms is not None else None
        )
        classification_previous_state = (
            self._previous_state
            if same_track and delta_from_previous is not None
            and 0 < delta_from_previous <= self.config.max_observation_hold_seconds * 1000
            else None
        )
        state = classify_orientation(
            observation.angles, self.config, previous_state=classification_previous_state,
        )
        delta_seconds = 0.0
        continuous = False
        reset_reason = None
        if self._last_valid_ms is not None:
            previous_valid_ms = self._last_valid_ms
            if observation.transient_track_key != self._last_track_key:
                self._reset_time_state()
                reset_reason = "tracking_key_changed"
            delta_ms = timestamp_ms - previous_valid_ms
            if delta_ms <= 0:
                self._reset_time_state()
                reset_reason = "non_monotonic_timestamp"
            elif delta_ms <= self.config.max_observation_hold_seconds * 1000:
                if reset_reason is None:
                    delta_seconds = delta_ms / 1000.0
                    continuous = True
            else:
                self._reset_time_state()
                reset_reason = "observation_gap"

        if state == "front":
            self._front_seconds = (
                self._front_seconds + delta_seconds
                if continuous and self._previous_state == "front" else 0.0
            )
            self._away_seconds = 0.0
            self._down_seconds = 0.0
            self._down_start_score = None
        elif state == "away":
            self._away_seconds = (
                self._away_seconds + delta_seconds
                if continuous and self._previous_state == "away" else 0.0
            )
            self._front_seconds = 0.0
            self._down_seconds = 0.0
            self._down_start_score = None
        elif state == "looking_down":
            if continuous and self._previous_state == "looking_down":
                self._down_seconds += delta_seconds
            else:
                self._down_seconds = 0.0
                # 아래 보기가 시작된 직전의 안정 점수를 감점 기준으로 고정한다.
                self._down_start_score = self._previous_stable_score
            self._front_seconds = 0.0
            self._away_seconds = 0.0
        else:
            # 불명확·아래 보기에서는 자세 유지 시간을 세지 않는다.
            self._front_seconds = 0.0
            self._away_seconds = 0.0
            self._down_seconds = 0.0
            self._down_start_score = None

        # 점수는 하위 Gaussian 실행기가 계산한다. 여기서는 관측 상태만 갱신한다.
        score = None
        reasons = [reset_reason] if reset_reason else []

        self._last_valid_ms = timestamp_ms
        self._previous_state = state
        self._last_track_key = observation.transient_track_key
        self.latest = {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "config_version": self.config.config_version,
            "status": "scored" if score is not None else "deferred",
            "orientation_state": state,
            "score": round(score, 1) if score is not None else None,
            "reason_codes": reasons,
            "angles": {
                "pitch": observation.angles.pitch,
                "yaw": observation.angles.yaw,
                "roll": observation.angles.roll,
            },
            "source_angles": ({
                "pitch": observation.source_angles.pitch,
                "yaw": observation.source_angles.yaw,
                "roll": observation.source_angles.roll,
            } if observation.source_angles is not None else None),
            "front_seconds": round(self._front_seconds, 2),
            "away_seconds": round(self._away_seconds, 2),
            "looking_down_seconds": round(self._down_seconds, 2),
            "previous_stable_score": (
                round(self._previous_stable_score, 1)
                if self._previous_stable_score is not None else None
            ),
            "thresholds": self._thresholds(),
        }
        self.latest.update(self._display_fields(timestamp_ms, current_score=self.latest["score"]))
        return self.latest
