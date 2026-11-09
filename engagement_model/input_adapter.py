"""VisionPipeline의 프레임별 사전을 단일 학생 관측으로 변환합니다."""

from dataclasses import dataclass
import math
from uuid import UUID

from .config import ModelConfig


@dataclass(frozen=True)
class HeadAngles:
    """정면 기준 상대 각도입니다. 현재 환경에서 아래쪽 pitch는 양수입니다."""

    pitch: float
    yaw: float
    roll: float


@dataclass(frozen=True)
class Observation:
    """관측 불가는 angles=None으로 표현하며 0도와 구분합니다."""

    session_id: str
    timestamp_ms: int
    feature_version: str
    angles: HeadAngles | None
    reason_codes: tuple[str, ...] = ()
    transient_track_key: int | None = None
    source_angles: HeadAngles | None = None

    @property
    def observable(self):
        """점수 계산에 필요한 상대 각도가 준비되었는지 알려줍니다."""
        return self.angles is not None


def soften_angle(value, deadband, scale, full_scale_start):
    """중립 부근만 완화하고 큰 각도에서는 기울기 1을 유지하는 홀함수."""
    magnitude = abs(value)
    if magnitude <= deadband:
        return 0.0
    softened = (min(magnitude, full_scale_start) - deadband) * scale
    softened += max(0.0, magnitude - full_scale_start)
    return math.copysign(softened, value)


def parse_feature_window(data, config=None):
    """형식 오류는 ValueError, 정상적인 미검출·보정 실패는 관측 불가로 반환합니다.

    원본 사전을 수정하지 않습니다. 프레임 사이의 시간 역전·신원 변경 검사는
    여러 관측을 기억하는 후속 상태 관리 모듈에서 수행해야 합니다.
    """
    config = config or ModelConfig()
    if not isinstance(data, dict):
        raise ValueError("FeatureWindow는 사전이어야 합니다")
    sid = data.get("session_id")
    try:
        UUID(sid)
    except (ValueError, TypeError, AttributeError):
        raise ValueError("session_id는 UUID 문자열이어야 합니다") from None
    for name in ("start_ms", "end_ms"):
        value = data.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name}는 0 이상의 정수여야 합니다")
    if data["start_ms"] != data["end_ms"]:
        raise ValueError("현재 어댑터는 시점별 FeatureWindow만 지원합니다")
    if data.get("feature_version") != config.feature_version:
        raise ValueError("지원하지 않는 feature_version입니다")
    features = data.get("features")
    quality = data.get("quality")
    if not isinstance(features, dict) or not isinstance(quality, dict):
        raise ValueError("features와 quality 사전이 필요합니다")
    if quality.get("status") not in {"valid", "partial", "not_observable"}:
        raise ValueError("알 수 없는 최상위 품질 상태입니다")
    faces = features.get("faces")
    if not isinstance(faces, list):
        raise ValueError("features.faces는 목록이어야 합니다")

    reasons = []
    if quality["status"] == "not_observable":
        reasons.append("insufficient_observation")
    if not faces:
        reasons.append("face_not_found")
    elif len(faces) != 1:
        reasons.append("multiple_faces_unsupported")
    if features.get("head_calibration_status") != "ready":
        reasons.append("baseline_unavailable")

    angles = None
    source_angles = None
    track_key = None
    if len(faces) == 1:
        face = faces[0]
        if not isinstance(face, dict):
            raise ValueError("얼굴 항목은 사전이어야 합니다")
        face_quality = face.get("quality")
        # 이 모델은 머리 각도만 사용한다. 눈/표정 누락으로 전체 얼굴 품질이 partial이어도
        # 유효한 상대 각도가 있으면 쓸 수 있지만, 얼굴 자체를 관측할 수 없으면 제외한다.
        if (not isinstance(face_quality, dict)
                or face_quality.get("status") not in {"valid", "partial"}):
            reasons.append("landmark_low_quality")
        relative = face.get("head_angles_relative_degrees")
        if not isinstance(relative, dict) or any(
            isinstance(relative.get(axis), bool)
            or not isinstance(relative.get(axis), (int, float))
            or not math.isfinite(relative[axis]) or not -180 <= relative[axis] <= 180
            for axis in ("pitch", "yaw", "roll")
        ):
            reasons.append("relative_angles_unavailable")
        elif not reasons:
            source_angles = HeadAngles(**{
                axis: float(relative[axis]) for axis in ("pitch", "yaw", "roll")
            })
            model_yaw = soften_angle(source_angles.yaw, config.yaw_neutral_deadband_degrees,
                                     config.yaw_small_angle_scale, config.yaw_full_scale_start_degrees)
            # 중립 부근만 완만히 압축한다. 큰 좌우 회전은 거의 그대로 유지한다.
            angles = HeadAngles(source_angles.pitch, model_yaw, source_angles.roll)
        track_key = face.get("transient_track_key")
        if track_key is not None and (isinstance(track_key, bool)
                or not isinstance(track_key, int) or track_key < 0):
            raise ValueError("임시 추적 키는 0 이상의 정수 또는 None이어야 합니다")
    return Observation(sid, data["start_ms"], config.feature_version,
                       angles, tuple(dict.fromkeys(reasons)), track_key, source_angles)
