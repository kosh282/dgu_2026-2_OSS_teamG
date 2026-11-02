"""MediaPipe 상체 위치 측정. 개인 식별과 얼굴 연결은 수행하지 않는다."""
from pathlib import Path
import math

DEFAULT_POSE_MODEL = Path(__file__).resolve().parent / 'models' / 'pose_landmarker_lite.task'


def shoulder_features(points):
    """양쪽 어깨의 관측 조건을 확인하고 화면 상대 중심 위치를 반환한다."""
    left, right = points[11], points[12]
    shoulders = (left, right)
    visibility_values = [float(point.visibility) for point in shoulders]
    visibility = min(visibility_values) if all(map(math.isfinite, visibility_values)) else None
    in_frame = all(
        math.isfinite(point.x) and math.isfinite(point.y)
        and 0 <= point.x <= 1 and 0 <= point.y <= 1
        for point in shoulders
    )
    # 어느 한 어깨라도 범위 밖이거나 가시성이 부족하면 0 대신 null을 전달한다.
    observable = visibility is not None and visibility >= 0.5 and in_frame
    position = None
    if observable:
        position = {'x': float((left.x + right.x) / 2),
                    'y': float((left.y + right.y) / 2)}
    return {
        'shoulder_center': position,
        'shoulder_visibility': visibility,
        'quality': {'status': 'valid' if observable else 'not_observable',
                    'reason_codes': [] if observable else ['insufficient_observation']},
    }


class PoseAnalyzer:
    """전체 창에서 상체 후보를 찾고, 프레임마다 독립적인 위치를 반환한다."""

    def __init__(self, model_path=DEFAULT_POSE_MODEL, max_poses=20, show_landmarks=False):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=max_poses,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.last_timestamp = -1
        # 시각화 테스트에서만 원본 좌표를 메모리에 유지한다.
        self.show_landmarks = show_landmarks
        self.last_landmarks = []

    def analyze(self, frame):
        import cv2
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        timestamp = max(frame.capture_ms, self.last_timestamp + 1)
        self.last_timestamp = timestamp
        result = self.detector.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), timestamp)
        self.last_landmarks = (
            [[(float(p.x), float(p.y), float(p.visibility)) for p in points]
             for points in result.pose_landmarks]
            if self.show_landmarks else []
        )
        return [shoulder_features(points) for points in result.pose_landmarks]

    def close(self):
        self.detector.close()
