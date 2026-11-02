"""단일 얼굴의 정면 기준을 메모리에서 시험하는 실험적 보정."""
import math

AXES = ('pitch', 'yaw', 'roll')


def angle_difference(angle, baseline):
    """±180도 경계를 넘더라도 가장 짧은 부호 있는 차이를 반환한다."""
    return (angle - baseline + 180.0) % 360.0 - 180.0


class HeadCalibration:
    """명시적 시작 후 단일 얼굴을 3초 관찰한다. 개인 식별은 하지 않는다."""

    def __init__(self, duration_ms=3000, min_samples=10):
        self.duration_ms = duration_ms
        self.min_samples = min_samples
        self.status = 'waiting'
        self.samples = []
        self.baseline = None

    def start(self):
        self.status = 'collecting'
        self.samples = []
        self.baseline = None

    def invalidate(self):
        self.status = 'waiting'
        self.samples = []
        self.baseline = None

    def update(self, faces, capture_ms):
        for face in faces:
            face['head_angles_relative_degrees'] = None

        # 복수 얼굴, 가림 또는 검출 실패 시 기존 기준을 다른 얼굴에 적용하지 않는다.
        angles = faces[0].get('head_angles_degrees') if len(faces) == 1 else None
        valid = isinstance(angles, dict) and all(
            isinstance(angles.get(axis), (int, float)) and math.isfinite(angles[axis])
            for axis in AXES
        )
        if not valid:
            if self.status == 'ready':
                self.invalidate()
            elif self.status == 'collecting':
                self.samples = []
            return self.status
        if self.status == 'collecting':
            # 긴 프레임 공백은 연속된 정면 자세라고 가정하지 않는다.
            if self.samples and (capture_ms <= self.samples[-1][0] or capture_ms - self.samples[-1][0] > 1000):
                self.samples = []
            self.samples.append((capture_ms, dict(angles)))
            if (len(self.samples) >= self.min_samples
                    and capture_ms - self.samples[0][0] >= self.duration_ms):
                baseline = {}
                for axis in AXES:
                    radians = [math.radians(sample[1][axis]) for sample in self.samples]
                    x = sum(math.cos(value) for value in radians) / len(radians)
                    y = sum(math.sin(value) for value in radians) / len(radians)
                    if math.hypot(x, y) < 0.98:
                        self.samples = []
                        return self.status
                    baseline[axis] = math.degrees(math.atan2(y, x))
                self.baseline = baseline
                self.samples = []
                self.status = 'ready'
        if self.status == 'ready':
            # 캘리브레이션 기준은 영상 속 한 얼굴에 대한 임시 기준이며 지속적 신원 확인이 아니다.
            faces[0]['head_angles_relative_degrees'] = {
                axis: angle_difference(angles[axis], self.baseline[axis]) for axis in AXES
            }
        return self.status
