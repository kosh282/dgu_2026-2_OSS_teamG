"""FeatureWindow를 받아 실시간 데모용 행동 점수를 계산하는 상태 실행기."""

import math
from collections import deque

from .config import ModelConfig
from .input_adapter import parse_feature_window
from .scoring import behavior_score, classify_orientation, huang_window_score


MODEL_ID = "head-orientation-rules"
MODEL_VERSION = "experimental-0.5"
DEFAULT_DISPLAY_SCORE_HOLD_SECONDS = 2.0


class EngagementRuntime:
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
        self.config = config or ModelConfig()
        if self.config.scoring_method == "gaussian" and type(self) is EngagementRuntime:
            raise ValueError("Gaussian 설정은 engagement_model.gaussian.GaussianRuntime으로 실행하세요")
        self.model_id = MODEL_ID
        self.model_version = ("huang-online-0.2" if self.config.scoring_method == "huang"
                              else MODEL_VERSION)
        self._huang_segments = deque()
        self._huang_sample = None
        self._huang_start_ms = None
        self._huang_awaiting_front = False
        self._huang_recovery_ms = None
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
            "away_grace_seconds": self.config.away_grace_seconds,
            "down_grace_seconds": self.config.down_grace_seconds,
            "down_decay_seconds": self.config.down_decay_seconds,
            "down_penalty": self.config.down_penalty,
            "down_exit_pitch_degrees": self.config.down_exit_pitch_degrees,
            "yaw_neutral_deadband_degrees": self.config.yaw_neutral_deadband_degrees,
            "yaw_small_angle_scale": self.config.yaw_small_angle_scale,
            "yaw_full_scale_start_degrees": self.config.yaw_full_scale_start_degrees,
            **({"huang_small_seconds": self.config.huang_small_seconds,
                "huang_window_seconds": self.config.window_seconds,
                "huang_amplitude_degrees": self.config.huang_amplitude_degrees,
                "huang_max_yaw_degrees": self.config.huang_max_yaw_degrees}
               if self.config.scoring_method == "huang" else {}),
        }

    def _reset_time_state(self, *, clear_score=True):
        self._huang_segments.clear()
        self._huang_sample = None
        self._huang_start_ms = None
        self._huang_awaiting_front = False
        self._huang_recovery_ms = None
        self._last_valid_ms = None
        self._previous_state = None
        self._front_seconds = 0.0
        self._away_seconds = 0.0
        self._down_seconds = 0.0
        self._down_start_score = None
        self._last_track_key = None
        if clear_score:
            self._previous_stable_score = None

    def _apply_huang(self, result, timestamp_ms, observation=None):
        """기존 상태 관리를 공유하고 Huang 구간 점수만 계산한다."""
        if self.config.scoring_method != "huang":
            return
        primitive = None
        state = result["orientation_state"]
        if observation is not None and observation.observable:
            if state == "looking_down":
                self._huang_awaiting_front = True
                self._huang_recovery_ms = None
            elif state == "front" and self._huang_awaiting_front:
                # 실시간 복귀 점수는 새 관측에서 재개한다. publisher의 전체 구간은 보존한다.
                self._huang_segments.clear()
                self._huang_sample = None
                self._huang_start_ms = timestamp_ms
                self._huang_recovery_ms = timestamp_ms
                self._huang_awaiting_front = False
        if observation is not None and observation.observable and result["orientation_state"] != "looking_down":
            angles = observation.source_angles or observation.angles
            primitive = (-angles.pitch, angles.yaw)
        result["huang_observation"] = primitive
        if self._huang_start_ms is None:
            self._huang_start_ms = timestamp_ms
        if self._huang_sample is not None:
            previous_ms, observable, previous_primitive = self._huang_sample
            stop = min(timestamp_ms, previous_ms + round(self.config.max_observation_hold_seconds * 1000))
            if stop > previous_ms:
                self._huang_segments.append((previous_ms, stop, observable, None, previous_primitive))
        self._huang_sample = (timestamp_ms, result["angles"] is not None, primitive)
        start = max(self._huang_start_ms, timestamp_ms - round(self.config.window_seconds * 1000))
        while self._huang_segments and self._huang_segments[0][1] <= start:
            self._huang_segments.popleft()
        # 실시간 표시는 완료된 작은 구간까지만 평가한다. 미완료 꼬리를 결측으로 세면
        # 첫 점수 재개 후 3~6초 사이에 70% gate가 반복해서 꺼지는 문제가 생긴다.
        small_ms = round(self.config.huang_small_seconds * 1000)
        completed_end = timestamp_ms - (timestamp_ms - start) % small_ms
        summary = (huang_window_score(self._huang_segments, start, completed_end, self.config)
                   if completed_end > start else dict(score=None, reason_codes=["huang_warmup"]))
        reasons = list(result["reason_codes"])
        if (self._huang_recovery_ms is not None
                and timestamp_ms - self._huang_recovery_ms < self.config.huang_small_seconds * 1000):
            reasons.append("huang_recovery_warmup")
        if observation is None or not observation.observable:
            summary["score"] = None
        elif result["orientation_state"] == "looking_down":
            summary["score"] = None
            reasons.append("downward_activity_uncertain")
        result["score"] = summary["score"]
        result["status"] = "scored" if result["score"] is not None else "deferred"
        result["reason_codes"] = list(dict.fromkeys(reasons + summary["reason_codes"]))
        if result["score"] is not None:
            self._previous_stable_score = result["score"]
        result["previous_stable_score"] = self._previous_stable_score

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
            self._apply_huang(self.latest, timestamp_ms, observation)
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

        score = behavior_score(
            observation,
            front_seconds=self._front_seconds,
            away_seconds=self._away_seconds,
            looking_down_seconds=self._down_seconds,
            down_start_score=self._down_start_score,
            previous_stable_score=self._previous_stable_score,
            previous_orientation_state=classification_previous_state,
            config=self.config,
        ) if self.config.scoring_method == "legacy" else None
        reasons = [reset_reason] if reset_reason else []
        if score is None and self.config.scoring_method == "legacy":
            if state == "looking_down":
                reasons.append("looking_down_deferred")
            elif state == "ambiguous":
                reasons.append("orientation_ambiguous")
            elif state == "away" and self._previous_stable_score is None:
                reasons.append("away_grace_without_previous_score")
        elif score is not None:
            self._previous_stable_score = score
            if state == "looking_down":
                reasons.append(
                    "looking_down_score_decay" if self._down_seconds > self.config.down_grace_seconds
                    else "looking_down_grace"
                )

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
        self._apply_huang(self.latest, timestamp_ms, observation)
        self.latest.update(self._display_fields(timestamp_ms, current_score=self.latest["score"]))
        return self.latest


class ComparisonRuntime:
    """동일 특징을 독립 실행기에 전달한다. 요약은 프레임 기준이며 성능 지표가 아니다."""

    def __init__(self, primary="huang", *, include_gaussian=False, gaussian_config=None):
        if primary not in {"legacy", "huang", "gaussian"}:
            raise ValueError("primary는 legacy, huang 또는 gaussian이어야 합니다")
        self.runtimes = {"legacy": EngagementRuntime(),
                         "huang": EngagementRuntime(ModelConfig.huang())}
        if include_gaussian or primary == "gaussian":
            from .gaussian import GaussianRuntime
            self.runtimes["gaussian"] = GaussianRuntime(gaussian_config)
        self.primary = primary
        self.counts = {name: dict(frames=0, deferred=0, score_sum=0.0, scored=0)
                       for name in self.runtimes}

    def __getattr__(self, name):
        return getattr(self.runtimes[self.primary], name)

    def __call__(self, feature_window):
        for name, runtime in self.runtimes.items():
            result = runtime(feature_window)
            stats = self.counts[name]
            stats["frames"] += 1
            if result["score"] is None:
                stats["deferred"] += 1
            else:
                stats["score_sum"] += result["score"]
                stats["scored"] += 1
        return self.latest

    def summary(self):
        return {name: dict(frames=stats["frames"],
                           mean_score=round(stats["score_sum"] / stats["scored"], 2) if stats["scored"] else None,
                           deferred_ratio=round(stats["deferred"] / stats["frames"], 4) if stats["frames"] else None)
                for name, stats in self.counts.items()}
