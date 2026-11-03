"""MediaPipe 얼굴 특징을 시간축에서 보강하는 교체 가능한 후처리기.

이 모듈은 검출기와 분리되어 있어 모델이나 MediaPipe 결과 형식이 바뀌어도
입력 얼굴 특징을 이 인터페이스에 맞추면 후처리 부분을 계속 재사용할 수 있다.

알고리즘 참고 및 일부 MIT 코드 적응:
ATemova/face-mesh의 1-Euro pose smoothing, EAR blink tracking, centroid tracker.
저작권 및 라이선스 전문은 저장소 루트 THIRD_PARTY_NOTICES.md 참고.
"""
from collections import deque
from dataclasses import dataclass
import math


AXES = ('pitch', 'yaw', 'roll')
POSTPROCESSING_VERSION = 'face-postprocess-0.1'


def wrap_degrees(angle):
    """각도 차이를 [-180, 180) 범위로 접어 경계 통과 점프를 피한다."""
    return (angle + 180.0) % 360.0 - 180.0


class OneEuroFilter:
    """단일 각도 신호에 적용하는 시간 기반 1-Euro 저역 필터."""

    def __init__(self, min_cutoff=1.0, beta=0.05, derivative_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.value = None
        self.raw_value = None
        self.filtered_derivative = 0.0
        self.last_time = None

    @staticmethod
    def _alpha(cutoff, delta_seconds):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / delta_seconds)

    @staticmethod
    def _low_pass(previous, value, alpha):
        return value if previous is None else alpha * value + (1.0 - alpha) * previous

    def update(self, value, timestamp_seconds):
        if self.last_time is None:
            self.value = self.raw_value = value
            self.last_time = timestamp_seconds
            return value

        delta = timestamp_seconds - self.last_time
        if delta <= 0:
            return self.value
        # 프레임이 장시간 끊긴 뒤에는 과거 상태를 새 신호에 이어 붙이지 않는다.
        if delta > 1.0:
            self.value = self.raw_value = value
            self.filtered_derivative = 0.0
            self.last_time = timestamp_seconds
            return value

        frequency = 1.0 / delta
        derivative = (value - self.raw_value) * frequency
        derivative_alpha = self._alpha(self.derivative_cutoff, delta)
        self.filtered_derivative = self._low_pass(
            self.filtered_derivative, derivative, derivative_alpha)
        cutoff = self.min_cutoff + self.beta * abs(self.filtered_derivative)
        alpha = self._alpha(cutoff, delta)
        self.value = self._low_pass(self.value, value, alpha)
        self.raw_value = value
        self.last_time = timestamp_seconds
        return self.value


class AngularPoseSmoother:
    """pitch/yaw/roll을 각도 경계를 고려해 각각 스무딩한다."""

    def __init__(self, min_cutoff=1.0, beta=0.05):
        self.filters = {axis: OneEuroFilter(min_cutoff, beta) for axis in AXES}
        self.previous_raw = {}
        self.unwrapped = {}
        self.previous_smoothed = None
        self.previous_time = None

    def update(self, angles, timestamp_ms):
        timestamp = timestamp_ms / 1000.0
        unwrapped_angles = {}
        for axis in AXES:
            raw = float(angles[axis])
            if axis not in self.previous_raw:
                self.unwrapped[axis] = raw
            else:
                self.unwrapped[axis] += wrap_degrees(raw - self.previous_raw[axis])
            self.previous_raw[axis] = raw
            unwrapped_angles[axis] = self.filters[axis].update(self.unwrapped[axis], timestamp)

        smoothed = {axis: wrap_degrees(unwrapped_angles[axis]) for axis in AXES}
        movement = None
        if self.previous_smoothed is not None and self.previous_time is not None:
            interval = timestamp_ms - self.previous_time
            if 0 < interval <= 1000:
                movement = {
                    'delta_degrees': {
                        axis: wrap_degrees(smoothed[axis] - self.previous_smoothed[axis])
                        for axis in AXES
                    },
                    'interval_ms': interval,
                }
        self.previous_smoothed = smoothed
        self.previous_time = timestamp_ms
        return smoothed, movement


class _BlinkCounter:
    """EAR 기준 아래 상태가 연속 관측된 뒤 눈이 다시 열릴 때 한 번 센다."""

    def __init__(self, threshold, min_closed_frames):
        self.threshold = threshold
        self.min_closed_frames = min_closed_frames
        self.count = 0
        self.below_frames = 0
        self.closed = False

    def update(self, ear):
        if ear is None or not math.isfinite(ear):
            # 관측 공백은 눈 감김으로 채우지 않는다.
            self.below_frames = 0
            self.closed = False
            return False
        if ear < self.threshold:
            self.below_frames += 1
            if self.below_frames >= self.min_closed_frames:
                self.closed = True
        else:
            blink_completed = self.closed
            if blink_completed:
                self.count += 1
            self.below_frames = 0
            self.closed = False
            return blink_completed
        return False


class BlinkMetrics:
    """양쪽 눈 EAR에서 깜빡임 수와 최근 60초 빈도를 계산한다.

    기본 임계치는 참고 라이브러리 값이며 참가자 타일에 맞춘 검증값은 아니다.
    수업 화면과 타일 크기에 맞춰 PostprocessingConfig에서 조정할 수 있다.
    """

    def __init__(self, threshold=0.21, min_closed_frames=2, rate_window_ms=60_000):
        self.left = _BlinkCounter(threshold, min_closed_frames)
        self.right = _BlinkCounter(threshold, min_closed_frames)
        self.both = _BlinkCounter(threshold, min_closed_frames)
        self.rate_window_ms = rate_window_ms
        self.events = deque()

    def update(self, left_ear, right_ear, timestamp_ms, mean_ear=None):
        left_event = self.left.update(left_ear)
        right_event = self.right.update(right_ear)
        if mean_ear is None and left_ear is not None and right_ear is not None:
            mean_ear = (left_ear + right_ear) / 2.0
        both_event = self.both.update(mean_ear)
        if both_event:
            self.events.append(timestamp_ms)

        cutoff = timestamp_ms - self.rate_window_ms
        while self.events and self.events[0] < cutoff:
            self.events.popleft()
        return {
            'closed': self.both.closed,
            'left_closed': self.left.closed,
            'right_closed': self.right.closed,
            'left_count': self.left.count,
            'right_count': self.right.count,
            'total_count': self.both.count,
            'count_window_ms': self.rate_window_ms,
            'count_in_window': len(self.events),
            'blinks_per_minute': round(len(self.events) * 60_000 / self.rate_window_ms, 1),
            'event_completed': bool(both_event or left_event or right_event),
        }


@dataclass
class PostprocessingConfig:
    """추후 특징/모델 교체를 위해 튜닝 가능한 후처리 설정."""

    pose_min_cutoff: float = 1.0
    pose_beta: float = 0.05
    eye_closed_threshold: float = 0.21
    blink_min_closed_frames: int = 2
    track_max_distance: float = 0.12
    track_max_gap_ms: int = 1000
    blink_rate_window_ms: int = 60_000


@dataclass
class _FaceState:
    center: tuple[float, float]
    last_seen_ms: int
    pose: AngularPoseSmoother
    blink: BlinkMetrics


class FacePostProcessor:
    """한 프레임의 얼굴 후보에만 시간 후처리를 적용한다.

    얼굴 중심 간 최근접 연결은 내부 임시 키로 상태를 잇기 위한 실험 방법이다.
    키는 반환 특징에만 잠시 붙으며, 최종 세션/중간 JSON에는 직렬화하지 않는다.
    장기 신원, 참가자 ID 또는 검증된 사람 추적을 제공하지 않는다.
    """

    def __init__(self, config=None):
        self.config = config or PostprocessingConfig()
        self._states = {}
        self._next_key = 0

    def _assign_keys(self, faces, centers, timestamp_ms):
        centers = [tuple(center) for center in centers]
        if len(centers) != len(faces):
            centers = [
                ((face.get('head_position') or {}).get('x', 0.5),
                 (face.get('head_position') or {}).get('y', 0.5))
                for face in faces
            ]
        candidates = []
        for face_index, center in enumerate(centers):
            for key, state in self._states.items():
                if timestamp_ms - state.last_seen_ms > self.config.track_max_gap_ms:
                    continue
                distance = math.hypot(center[0] - state.center[0], center[1] - state.center[1])
                if distance <= self.config.track_max_distance:
                    candidates.append((distance, face_index, key))
        candidates.sort()
        assignments = {}
        used_keys = set()
        for _, face_index, key in candidates:
            if face_index not in assignments and key not in used_keys:
                assignments[face_index] = key
                used_keys.add(key)
        for face_index in range(len(faces)):
            if face_index not in assignments:
                assignments[face_index] = self._next_key
                self._next_key += 1
        for key, state in list(self._states.items()):
            if timestamp_ms - state.last_seen_ms > self.config.track_max_gap_ms:
                del self._states[key]
        return [assignments[index] for index in range(len(faces))], centers

    def process(self, faces, centers, timestamp_ms):
        """입력 얼굴 객체를 보강하고 같은 리스트를 반환한다."""
        keys, centers = self._assign_keys(faces, centers, timestamp_ms)
        for face, center, key in zip(faces, centers, keys):
            state = self._states.get(key)
            if state is None:
                state = _FaceState(
                    center=center,
                    last_seen_ms=timestamp_ms,
                    pose=AngularPoseSmoother(
                        self.config.pose_min_cutoff, self.config.pose_beta),
                    blink=BlinkMetrics(
                        self.config.eye_closed_threshold,
                        self.config.blink_min_closed_frames,
                        self.config.blink_rate_window_ms),
                )
                self._states[key] = state
            state.center = center
            state.last_seen_ms = timestamp_ms

            angles = face.get('head_angles_degrees')
            if angles:
                smoothed, movement = state.pose.update(angles, timestamp_ms)
            else:
                smoothed, movement = None, None
            blink = state.blink.update(
                face.get('eye_aspect_ratio_left'),
                face.get('eye_aspect_ratio_right'),
                timestamp_ms,
                mean_ear=face.get('eye_openness'),
            )
            # 임시 키는 오직 런타임 모델 콜백/개발자 화면에서의 상태 연결용이다.
            face.update({
                'transient_track_key': key,
                'head_angles_smoothed_degrees': smoothed,
                'head_angle_change_candidate': movement,
                'blink_metrics': blink,
                'postprocessing_version': POSTPROCESSING_VERSION,
                'tracking_quality': 'position_only',
            })
        return faces

    def reset(self):
        """수업이 끝나거나 새 스트림으로 전환될 때 시간 상태를 버린다."""
        self._states.clear()
        self._next_key = 0
