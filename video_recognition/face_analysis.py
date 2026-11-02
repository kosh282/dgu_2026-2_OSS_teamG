"""팀원 얼굴 분석을 수업 창의 여러 얼굴에 적용하는 공통 분석기."""
from pathlib import Path

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
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


def head_angles(landmarks, width, height):
    """기존 solvePnP 머리 방향 추정. 보정 전 실험값이며 참여도 판정이 아니다."""
    import cv2
    import numpy as np
    # 코·턱·양쪽 눈과 입의 2D 위치를 기존 얼굴 모형과 대응시킨다.
    image_points = np.array(
        [[landmarks[i].x * width, landmarks[i].y * height]
         for i in (1, 152, 33, 263, 61, 291)],
        dtype=np.float64,
    )
    model_points = np.array(
        [[0, 0, 0], [0, -330, -65], [-225, 170, -135],
         [225, 170, -135], [-150, -150, -125], [150, -150, -125]],
        dtype=np.float64,
    )
    camera = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float64)
    ok, rotation, _ = cv2.solvePnP(model_points, image_points, camera, np.zeros((4, 1)))
    if not ok:
        return None
    matrix, _ = cv2.Rodrigues(rotation)
    angles = cv2.RQDecomp3x3(matrix)[0]
    return {key: float(value) for key, value in zip(('pitch', 'yaw', 'roll'), angles)}


class FaceAnalyzer:
    """프레임 단위 다중 얼굴 분석. 순서는 개인 식별이나 추적 정보가 아니다."""
    def __init__(self, model_path=DEFAULT_MODEL, max_faces=20, show_landmarks=False):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.mp = mp
        # Windows의 한글 경로에서도 읽도록 Python에서 모델 바이트를 전달한다.
        base_options = python.BaseOptions(model_asset_buffer=Path(model_path).read_bytes())
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.last_timestamp = -1
        # 시험 화면에서만 원본 좌표를 볼 수 있도록 보관한다. 기본 분석은 보관하지 않는다.
        self.show_landmarks = show_landmarks
        self.last_landmarks = []

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
        # 원본은 첫 얼굴 하나만 사용했지만 수업 창에서는 검출된 모든 얼굴을 처리한다.
        # 배열의 순서는 매 프레임 바뀔 수 있으므로 참가자 ID로 사용하면 안 된다.
        faces = []
        for points in result.face_landmarks:
            left = eye_ratio(points, LEFT_EYE, frame.width, frame.height)
            right = eye_ratio(points, RIGHT_EYE, frame.width, frame.height)
            openness = (left + right) / 2 if left is not None and right is not None else None
            faces.append({
                'eye_openness': openness,
                'head_angles_degrees': head_angles(points, frame.width, frame.height),
            })
        return faces

    def close(self):
        self.detector.close()
