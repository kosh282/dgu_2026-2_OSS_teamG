"""단일 얼굴의 정면 기준을 메모리에서 시험하는 실험적 보정."""

import math


AXES = ("pitch", "yaw", "roll")


def angle_difference(angle, baseline):
    """±180도 경계를 넘더라도 가장 짧은 부호 있는 차이를 반환한다."""
    return (angle - baseline + 180.0) % 360.0 - 180.0


class HeadCalibration:
    """명시적 시작 후 품질이 유효한 얼굴 하나를 3초 관찰한다."""

    def __init__(self, duration_ms=3000, min_samples=10):
        self.duration_ms = duration_ms
        self.min_samples = min_samples
        self.status = "waiting"
        self.samples = []
        self.baseline = None
        self.last_reason_code = None
        self._first_sample_ms = None
        self._completed_elapsed_ms = None

    def start(self):
        """사용자가 정면을 보고 움직임을 멈춘 뒤 보정을 시작한다."""
        self.status = "collecting"
        self.samples = []
        self.baseline = None
        self.last_reason_code = None
        self._first_sample_ms = None
        self._completed_elapsed_ms = None

    def invalidate(self, reason_code="calibration_invalidated"):
        self.status = "waiting"
        self.samples = []
        self.baseline = None
        self.last_reason_code = reason_code
        self._first_sample_ms = None
        self._completed_elapsed_ms = None

    def diagnostics(self, capture_ms=None):
        """화면·터미널에서 보정 진행률과 마지막 입력 문제를 확인할 자료."""
        if self.status == "ready":
            elapsed_ms = self._completed_elapsed_ms or self.duration_ms
            progress = 1.0
        elif self._first_sample_ms is not None and self.samples:
            last_ms = self.samples[-1][0] if capture_ms is None else capture_ms
            elapsed_ms = max(0, last_ms - self._first_sample_ms)
            sample_progress = min(1.0, len(self.samples) / self.min_samples)
            time_progress = min(1.0, elapsed_ms / self.duration_ms)
            progress = min(sample_progress, time_progress)
        else:
            elapsed_ms = 0
            progress = 0.0
        return {
            "status": self.status,
            "sample_count": len(self.samples),
            "required_samples": self.min_samples,
            "elapsed_ms": elapsed_ms,
            "required_duration_ms": self.duration_ms,
            "progress": progress,
            "last_reason_code": self.last_reason_code,
        }

    def _input_reason(self, faces):
        if not faces:
            return "no_face_detected"
        if len(faces) != 1:
            return "multiple_faces_detected"
        quality = faces[0].get("quality")
        if (not isinstance(quality, dict)
                or quality.get("status") not in {"valid", "partial"}):
            return "face_quality_not_observable"
        angles = (faces[0].get("head_angles_smoothed_degrees")
                  or faces[0].get("head_angles_degrees"))
        if not isinstance(angles, dict) or any(
            isinstance(angles.get(axis), bool)
            or not isinstance(angles.get(axis), (int, float))
            or not math.isfinite(angles[axis])
            or not -180 <= angles[axis] <= 180
            for axis in AXES
        ):
            return "head_angles_unavailable"
        return None

    def update(self, faces, capture_ms):
        for face in faces:
            face["head_angles_relative_degrees"] = None
            face["head_angles_relative_raw_degrees"] = None

        reason = self._input_reason(faces)
        if reason:
            if self.status == "ready":
                self.invalidate(reason)
            elif self.status == "collecting":
                self.samples = []
                self._first_sample_ms = None
                self.last_reason_code = reason
            return self.status

        if self.status == "waiting":
            return self.status

        face = faces[0]
        # 스무딩 신호가 있으면 보정 기준과 이후 모델 입력에 사용한다.
        angles = face.get("head_angles_smoothed_degrees") or face["head_angles_degrees"]
        if self.status == "collecting":
            reset_reason = None
            if self.samples:
                previous_ms = self.samples[-1][0]
                if capture_ms <= previous_ms:
                    reset_reason = "calibration_timestamp_not_increasing"
                elif capture_ms - previous_ms > 1000:
                    reset_reason = "calibration_frame_gap"
                if reset_reason:
                    self.samples = []
                    self._first_sample_ms = None
                    self.last_reason_code = reset_reason

            if not self.samples:
                self._first_sample_ms = capture_ms
            self.samples.append((capture_ms, dict(angles)))
            if reset_reason:
                self.last_reason_code = reset_reason

            elapsed_ms = capture_ms - self._first_sample_ms
            if len(self.samples) >= self.min_samples and elapsed_ms >= self.duration_ms:
                baseline = {}
                for axis in AXES:
                    radians = [math.radians(sample[1][axis]) for sample in self.samples]
                    x = sum(math.cos(value) for value in radians) / len(radians)
                    y = sum(math.sin(value) for value in radians) / len(radians)
                    if math.hypot(x, y) < 0.98:
                        self.samples = []
                        self._first_sample_ms = None
                        self.last_reason_code = "head_pose_unstable"
                        return self.status
                    baseline[axis] = math.degrees(math.atan2(y, x))
                self.baseline = baseline
                self._completed_elapsed_ms = elapsed_ms
                self.samples = []
                self.status = "ready"
                self.last_reason_code = None

        if self.status == "ready":
            # 보정 기준은 현재 한 얼굴에 대한 임시 기준이며 장기 신원 확인이 아니다.
            face["head_angles_relative_degrees"] = {
                axis: angle_difference(angles[axis], self.baseline[axis])
                for axis in AXES
            }
            # 비교·진단을 위해 원본도 같은 스무딩 기준점에서의 상대각으로 별도 유지한다.
            raw_angles = face.get("head_angles_degrees") or angles
            face["head_angle_model_input_source"] = (
                "smoothed" if face.get("head_angles_smoothed_degrees") else "raw_fallback"
            )
            face["head_angles_relative_raw_degrees"] = {
                axis: angle_difference(raw_angles[axis], self.baseline[axis])
                for axis in AXES
            }
        return self.status
