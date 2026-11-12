"""웹캠 Gaussian 전용 상대 회전 보정. 영상 추정·공통 FeatureWindow를 변경하지 않는다."""

from copy import deepcopy
import math
import numpy as np

from .gaussian import GaussianRuntime

AXES = ('pitch', 'yaw', 'roll')


def euler_rotation(angles):
    """OpenCV Euler 출력의 Rz(roll) Ry(yaw) Rx(pitch)를 복원한다."""
    if not isinstance(angles, dict) or any(
            isinstance(angles.get(axis), bool) or not isinstance(angles.get(axis), (int, float))
            or not math.isfinite(angles[axis]) or not -180 <= angles[axis] <= 180 for axis in AXES):
        raise ValueError('유효한 Pitch/Yaw/Roll이 필요합니다')
    p, y, r = (math.radians(angles[axis]) for axis in AXES)
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


def rotation_euler(matrix):
    """상대 회전을 정준 Euler 표현으로 변환한다. Yaw는 ±90도, 특이점에서 Roll=0이다."""
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(matrix), 1, atol=1e-7)):
        raise ValueError('유효한 회전행렬이 필요합니다')
    cosine = math.hypot(matrix[0, 0], matrix[1, 0])
    yaw = math.atan2(-matrix[2, 0], cosine)
    if cosine > 1e-8:
        pitch, roll = math.atan2(matrix[2, 1], matrix[2, 2]), math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        pitch, roll = math.atan2(-matrix[1, 2], matrix[1, 1]), 0.0
    return dict(zip(AXES, map(math.degrees, (pitch, yaw, roll))))


def relative_rotation_angles(current, reference):
    return rotation_euler(relative_rotation_matrix(euler_rotation(current), reference))


def calibration_frame(reference):
    """기준의 Yaw/Roll을 따른 축. Pitch 증가=아래라는 기존 국소 계약을 보존한다.

    R0=Rz(r0)Ry(y0)Rx(p0)에서 F=Rz(r0)Ry(y0)이다.
    카메라 축의 상대 회전을 F로 좌표 변환한다. Euler 동치 표현은 먼저
    행렬에서 정준화하므로 수집 당시 표현 분기에 의존하지 않는다.
    """
    origin = rotation_euler(reference)
    return euler_rotation({**origin, 'pitch': 0.0})


def relative_rotation_matrix(current, reference):
    frame = calibration_frame(reference)
    return frame.T @ current @ np.asarray(reference).T @ frame


class RotationCalibration:
    """기존 3초/10표본·안정성 기준을 행렬 평균에 적용하는 임시 단일 얼굴 보정."""

    def __init__(self, duration_ms=3000, min_samples=10):
        self.duration_ms, self.min_samples = duration_ms, min_samples
        self.invalidate()

    def invalidate(self, reason=None):
        self.status, self.last_reason_code = 'waiting', reason
        self.baseline = self.reference = self.relative = self.relative_matrix = None
        self.observation_ms = None
        self.samples = []
        self.diagnostic_samples = []

    def start(self):
        self.invalidate()
        self.status = 'collecting'

    def update(self, faces, timestamp):
        self.observation_ms = timestamp
        self.relative = self.relative_matrix = None
        if len(faces) > 1:
            self.invalidate('multiple_faces_detected')
            return
        reason = 'no_face_detected' if not faces else None
        angles = faces[0].get('head_angles_degrees') if faces else None
        if faces and (faces[0].get('quality') or {}).get('status') not in ('valid', 'partial'):
            reason = 'face_quality_not_observable'
        try:
            rotation = euler_rotation(angles)
        except ValueError:
            reason = reason or 'head_angles_unavailable'
        if reason:
            self.last_reason_code = reason
            if self.status == 'collecting':
                self.samples.clear()
                self.diagnostic_samples.clear()
            return
        self.last_reason_code = None
        if self.status == 'waiting':
            return
        if self.status == 'collecting':
            if self.samples and not 0 < timestamp-self.samples[-1][0] <= 1000:
                self.samples.clear()
            self.samples.append((timestamp, dict(angles)))
            self.diagnostic_samples = deepcopy(self.samples)
            if len(self.samples) < self.min_samples or timestamp-self.samples[0][0] < self.duration_ms:
                return
            mean = np.mean([euler_rotation(a) for _, a in self.samples], axis=0)
            u, singular, vt = np.linalg.svd(mean)
            # 기존 보정의 0.98 응집도 기준을 회전행렬 평균에 적용한다. 최적값 주장이 아니다.
            if min(singular) < .98:
                self.samples.clear()
                self.diagnostic_samples.clear()
                self.last_reason_code = 'head_pose_unstable'
                return
            self.reference = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
            self.baseline = rotation_euler(self.reference)
            self.samples.clear()
            self.status = 'ready'
        if self.status == 'ready':
            try:
                rotation_euler(self.reference)
            except (ValueError, TypeError):
                self.invalidate('rotation_baseline_invalid')
                return
            self.relative_matrix = relative_rotation_matrix(rotation, self.reference)
            self.relative = rotation_euler(self.relative_matrix)

    def diagnostics(self, capture_ms=None):
        elapsed = self.samples[-1][0]-self.samples[0][0] if self.samples else 0
        return dict(status=self.status, sample_count=len(self.samples), required_samples=self.min_samples,
            elapsed_ms=elapsed, required_duration_ms=self.duration_ms,
            progress=1.0 if self.status == 'ready' else min(1, len(self.samples)/self.min_samples, elapsed/self.duration_ms),
            last_reason_code=self.last_reason_code)

    def prepare_feature(self, feature):
        result = deepcopy(feature)
        result['features']['head_calibration_status'] = self.status
        result['features']['head_calibration_diagnostics'] = self.diagnostics(feature['start_ms'])
        for face in result['features']['faces']:
            current = self.relative if feature['start_ms'] == self.observation_ms else None
            face['head_angles_relative_degrees'] = deepcopy(current)
            face['head_angles_relative_raw_degrees'] = deepcopy(current)
            face['head_angle_model_input_source'] = 'rotation_matrix_raw'
        return result


class RotationGaussianRuntime(GaussianRuntime):
    """기존 Gaussian 식에 행렬 보정된 입력만 전달한다. 다른 런타임에는 관여하지 않는다."""

    def __init__(self, calibration, config=None):
        self.rotation_calibration = calibration
        self.input_feature = None
        super().__init__(config)

    def __call__(self, feature):
        self.input_feature = self.rotation_calibration.prepare_feature(feature)
        result = super().__call__(self.input_feature)
        result['calibration_method'] = 'rotation_matrix'
        result['relative_rotation_matrix'] = (self.rotation_calibration.relative_matrix.tolist()
            if self.rotation_calibration.relative_matrix is not None
            and feature['start_ms'] == self.rotation_calibration.observation_ms else None)
        return result
