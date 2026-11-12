"""팀원 얼굴 분석을 수업 창의 여러 얼굴에 적용하는 공통 분석기."""
from pathlib import Path
import math

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EXPRESSION_KEYS = {'mouthSmileLeft', 'mouthSmileRight', 'browInnerUp', 'jawOpen'}
# 실행 위치가 달라도 영상 담당 폴더의 같은 모델을 읽는다.
DEFAULT_MODEL = Path(__file__).resolve().parent / 'models' / 'face_landmarker.task'


def eye_ratio(landmarks, indices, width, height):
    """팀원 코드의 눈 위아래 거리 / 좌우 거리 계산."""
    import numpy as np
    # 정규화 좌표를 실제 픽셀로 변환해 기존 눈 개방 비율 계산을 유지한다.
    points = np.array([
        [landmarks[i].x * width, landmarks[i].y * height]
        for i in indices
    ])
    horizontal = np.linalg.norm(points[0] - points[3])
    if horizontal <= 1e-9:
        return None
    vertical = np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4])
    return float(vertical / (2 * horizontal))


def points_in_frame(landmarks, indices):
    """필요한 기준점이 화면 안에 있고 유한한지 확인한다. 정확도 검증은 아니다."""
    return all(
        math.isfinite(landmarks[index].x) and math.isfinite(landmarks[index].y)
        and 0 <= landmarks[index].x <= 1 and 0 <= landmarks[index].y <= 1
        for index in indices
    )


def periodic_angles(angles):
    """각도 경계에서 값이 끊기지 않도록 각 축의 sin/cos를 함께 전달한다."""
    if angles is None or not all(math.isfinite(angles[axis]) for axis in ('pitch', 'yaw', 'roll')):
        return None
    return {
        axis: {'sin': math.sin(math.radians(angles[axis])),
               'cos': math.cos(math.radians(angles[axis]))}
        for axis in ('pitch', 'yaw', 'roll')
    }



class FaceAnalyzer:
    """프레임 단위 다중 얼굴 분석. 순서는 개인 식별이나 추적 정보가 아니다."""
    def __init__(self, model_path=DEFAULT_MODEL, max_faces=20, show_landmarks=False, pose_source='face-transform'):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        if pose_source != 'face-transform':
            raise ValueError('지원하지 않는 얼굴 자세 추정 방식입니다')
        self.pose_source = pose_source
        self.mp = mp
        # Windows의 한글 경로에서도 읽도록 Python에서 모델 바이트를 전달한다.
        base_options = python.BaseOptions(model_asset_buffer=Path(model_path).read_bytes())
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=pose_source == 'face-transform',
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.last_timestamp = -1
        # 시험 화면에서만 원본 좌표를 볼 수 있도록 보관한다. 기본 분석은 보관하지 않는다.
        self.show_landmarks = show_landmarks
        self.last_landmarks = []
        # 얼굴 배열 순서는 프레임마다 달라질 수 있으므로, 후처리기가 임시 위치 추적에 쓸 중심점도 함께 보관한다.
        self.last_face_centers = []
        self.last_face_boxes = []

    def analyze(self, frame):
        import cv2
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        # MediaPipe는 증가하는 시각을 요구한다. 공통 캡처 시각은 별도로 유지한다.
        timestamp = max(frame.capture_ms, self.last_timestamp + 1)
        self.last_timestamp = timestamp
        result = self.detector.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), timestamp)
        self.last_landmarks = (
            [[(float(p.x), float(p.y), float(p.z)) for p in points]
             for points in result.face_landmarks]
            if self.show_landmarks else []
        )
        self.last_face_centers = [
            (float(sum(point.x for point in points) / len(points)),
             float(sum(point.y for point in points) / len(points)))
            for points in result.face_landmarks if points
        ]
        # 재식별 표시용 현재 프레임 경계. 특징 사전/JSON에는 넣지 않는다.
        self.last_face_boxes = [
            [max(0., min(p.x for p in points)), max(0., min(p.y for p in points)),
             min(1., max(p.x for p in points)), min(1., max(p.y for p in points))]
            for points in result.face_landmarks
        ]
        # 원본은 첫 얼굴 하나만 사용했지만 수업 창에서는 검출된 모든 얼굴을 처리한다.
        # 배열의 순서는 매 프레임 바뀔 수 있으므로 참가자 ID로 사용하면 안 된다.
        faces = []
        for index, points in enumerate(result.face_landmarks):
            eye_observable = points_in_frame(points, LEFT_EYE + RIGHT_EYE)
            pose_observable = points_in_frame(points, (1, 152, 33, 263, 61, 291))
            left = eye_ratio(points, LEFT_EYE, frame.width, frame.height) if eye_observable else None
            right = eye_ratio(points, RIGHT_EYE, frame.width, frame.height) if eye_observable else None
            openness = (left + right) / 2 if left is not None and right is not None else None
            if self.pose_source == 'face-transform':
                # 같은 검출의 행렬을 사용한다. 행렬이 없으면 PnP로 대체하지 않는다.
                from engagement_model.face_transform import transform_angles
                import numpy as np
                try:
                    angles = transform_angles(getattr(result, 'facial_transformation_matrixes', [])[index]) if pose_observable else None
                except (ValueError, IndexError, TypeError, np.linalg.LinAlgError):
                    angles = None
            orientation = periodic_angles(angles)
            if orientation is None:
                angles = None
            # 얼굴 경계 밖의 점을 포함한 형태 점수는 관측 불가로 둔다.
            expression_observable = points_in_frame(points, (1, 13, 14, 33, 61, 105, 133, 263, 291, 334, 362))
            expression = {
                category.category_name: float(category.score)
                for category in (result.face_blendshapes[index] if result.face_blendshapes else [])
                if category.category_name in EXPRESSION_KEYS
                and math.isfinite(category.score)
            } if expression_observable else {}
            if expression.keys() != EXPRESSION_KEYS:
                expression = {}
            reasons = []
            if angles is None and self.pose_source == 'face-transform':
                reasons.append('pose_unreliable')
            if openness is None or angles is None or not expression:
                reasons.append('insufficient_observation')
            faces.append({
                'eye_openness': openness,
                # 좌우 EAR를 따로 보존해 후처리 설정을 바꾸거나 다른 깜빡임 판정기를 꽂을 수 있게 한다.
                'eye_aspect_ratio_left': left,
                'eye_aspect_ratio_right': right,
                'head_angles_degrees': angles,
                'head_orientation_periodic': orientation,
                # 코의 화면 내 상대 위치. 프레임 간 이동량 계산에만 사용한다.
                'head_position': ({'x': float(points[1].x), 'y': float(points[1].y)}
                                  if points_in_frame(points, (1,)) else None),
                # 감정 분류가 아닌 MediaPipe의 얼굴 근육 형태 점수다.
                'expression_blendshapes': expression if expression else None,
                'quality': {'status': 'partial' if reasons else 'valid', 'reason_codes': reasons},
            })
        return faces

    def close(self):
        self.detector.close()
