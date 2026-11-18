"""팀원 얼굴 분석을 수업 창의 여러 얼굴에 적용하는 공통 분석기."""
from pathlib import Path
import math
from types import SimpleNamespace
from .layout_regions import detect_layout_regions

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
EXPRESSION_KEYS = {'mouthSmileLeft', 'mouthSmileRight', 'browInnerUp', 'jawOpen'}
# 실행 위치가 달라도 영상 담당 폴더의 같은 모델을 읽는다.
DEFAULT_MODEL = Path(__file__).resolve().parent / 'models' / 'face_landmarker.task'
DEFAULT_FACE_DETECTION_CONFIDENCE = 0.3
DEFAULT_FACE_PRESENCE_CONFIDENCE = 0.3
# 전체 프레임 재검출 주기. 참가자 배치가 변하면 이 주기를 기다리지 않는다.
DEFAULT_FULL_FRAME_INTERVAL_MS = 4000


def _overlapping_regions(width, height):
    """화면 배치와 무관하게 전체 화면을 겹치는 네 영역으로 나눈다."""
    tile_width = max(1, min(width, math.ceil(width * 0.65)))
    tile_height = max(1, min(height, math.ceil(height * 0.65)))
    return [(x, y, x + tile_width, y + tile_height)
            for y in (0, height - tile_height) for x in (0, width - tile_width)]


def _box_overlap(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0


def _same_face_box(a, b):
    """강한 겹침 또는 크기·중심이 거의 같은 경계만 중복 후보로 본다."""
    overlap = _box_overlap(a, b)
    if overlap >= 0.7:
        return True
    if overlap < 0.55:
        return False
    wa, ha = a[2] - a[0], a[3] - a[1]
    wb, hb = b[2] - b[0], b[3] - b[1]
    area_a, area_b = wa * ha, wb * hb
    if min(area_a, area_b) <= 0 or min(area_a, area_b) / max(area_a, area_b) < 0.75:
        return False
    center_a = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    center_b = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return math.dist(center_a, center_b) <= 0.15 * min(math.hypot(wa, ha), math.hypot(wb, hb))


def _layout_changed(previous, current):
    """순서는 무시하고 영역 크기·위치가 크게 달라진 경우에만 재검출한다."""
    if len(previous) != len(current):
        return True
    unmatched = list(current)
    for a in previous:
        def iou(b):
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            union = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - intersection
            return intersection / union if union > 0 else 0
        if not unmatched:
            return True
        best = max(range(len(unmatched)), key=lambda index: iou(unmatched[index]))
        if iou(unmatched[best]) < 0.8:
            return True
        unmatched.pop(best)
    return False


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
    def __init__(self, model_path=DEFAULT_MODEL, max_faces=20, show_landmarks=False, pose_source='face-transform',
                 full_frame_interval_ms=DEFAULT_FULL_FRAME_INTERVAL_MS):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        if pose_source != 'face-transform':
            raise ValueError('지원하지 않는 얼굴 자세 추정 방식입니다')
        if isinstance(full_frame_interval_ms, bool) or not isinstance(full_frame_interval_ms, int) or full_frame_interval_ms <= 0:
            raise ValueError('전체 프레임 분석 간격은 양의 밀리초 정수여야 합니다')
        self.pose_source = pose_source
        self.mp = mp
        # Windows의 한글 경로에서도 읽도록 Python에서 모델 바이트를 전달한다.
        base_options = python.BaseOptions(model_asset_buffer=Path(model_path).read_bytes())
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=DEFAULT_FACE_DETECTION_CONFIDENCE,
            min_face_presence_confidence=DEFAULT_FACE_PRESENCE_CONFIDENCE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=pose_source == 'face-transform',
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        tile_options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=DEFAULT_FACE_DETECTION_CONFIDENCE,
            min_face_presence_confidence=DEFAULT_FACE_PRESENCE_CONFIDENCE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        try:
            self.tile_detector = vision.FaceLandmarker.create_from_options(tile_options)
        except Exception:
            self.detector.close()
            raise
        self.max_faces = max_faces
        self.last_timestamp = -1
        self.full_frame_interval_ms = full_frame_interval_ms
        self._last_full_frame_ms = None
        self._layout_regions = []
        self._layout_timestamp = None
        self._layout_frame_size = None
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
        frame_size = (frame.width, frame.height)
        size_changed = self._layout_frame_size != frame_size
        layout_changed = False
        if (self._layout_frame_size != frame_size or self._layout_timestamp is None
                or timestamp - self._layout_timestamp >= 1000):
            new_regions = detect_layout_regions(frame.image)
            layout_changed = _layout_changed(self._layout_regions, new_regions)
            self._layout_regions = new_regions
            self._layout_timestamp = timestamp
            self._layout_frame_size = frame_size
        # 전체 화면 결과는 현재 프레임에서만 사용한다. 재검출 사이에는 부분 영역만 사용한다.
        result = None
        if (self._last_full_frame_ms is None or size_changed or layout_changed
                or timestamp - self._last_full_frame_ms >= self.full_frame_interval_ms):
            result = self.detector.detect_for_video(
                self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), timestamp)
            self._last_full_frame_ms = timestamp
        # 전체 화면과 각 부분의 결과를 공통 좌표로 변환한 뒤 겹치는 얼굴을 한 번만 쓴다.
        import numpy as np
        candidates = []

        def add_results(detected, region):
            x0, y0, x1, y1 = region
            rw, rh = x1 - x0, y1 - y0
            for index, local in enumerate(detected.face_landmarks):
                if not local:
                    continue
                xs, ys = [p.x for p in local], [p.y for p in local]
                # 부분 영역 경계에 잘린 얼굴은 자세·눈 값을 신뢰하기 어렵다.
                if ((x0 > 0 and min(xs) <= 0.01) or (x1 < frame.width and max(xs) >= 0.99)
                        or (y0 > 0 and min(ys) <= 0.01) or (y1 < frame.height and max(ys) >= 0.99)):
                    continue
                points = [SimpleNamespace(x=(x0 + p.x * rw) / frame.width,
                                          y=(y0 + p.y * rh) / frame.height,
                                          z=p.z * rw / frame.width) for p in local]
                box = (min(p.x for p in points), min(p.y for p in points),
                       max(p.x for p in points), max(p.y for p in points))
                blendshapes = detected.face_blendshapes or []
                shapes = blendshapes[index] if index < len(blendshapes) else []
                matrices = detected.facial_transformation_matrixes
                matrix = matrices[index] if matrices is not None and index < len(matrices) else None
                # 영역 안에서 더 크게 관측된 얼굴을 우선한다.
                priority = (max(xs) - min(xs)) * (max(ys) - min(ys))
                candidates.append((priority, box, points, shapes, matrix))

        if result is not None:
            add_results(result, (0, 0, frame.width, frame.height))
        regions = _overlapping_regions(frame.width, frame.height) + self._layout_regions
        for x0, y0, x1, y1 in regions:
            crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
            tile = self.tile_detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=crop))
            add_results(tile, (x0, y0, x1, y1))
        selected = []
        for candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
            if not any(_same_face_box(candidate[1], item[1]) for item in selected):
                selected.append(candidate)
                if len(selected) >= self.max_faces:
                    break
        selected.sort(key=lambda item: (item[1][1], item[1][0]))
        all_points = [item[2] for item in selected]
        self.last_landmarks = (
            [[(float(p.x), float(p.y), float(p.z)) for p in points] for points in all_points]
            if self.show_landmarks else []
        )
        self.last_face_centers = [
            (float(sum(point.x for point in points) / len(points)),
             float(sum(point.y for point in points) / len(points)))
            for points in all_points if points
        ]
        # 재식별 표시용 현재 프레임 경계. 특징 사전/JSON에는 넣지 않는다.
        self.last_face_boxes = [
            [max(0., min(p.x for p in points)), max(0., min(p.y for p in points)),
             min(1., max(p.x for p in points)), min(1., max(p.y for p in points))]
            for points in all_points
        ]
        # 원본은 첫 얼굴 하나만 사용했지만 수업 창에서는 검출된 모든 얼굴을 처리한다.
        # 배열의 순서는 매 프레임 바뀔 수 있으므로 참가자 ID로 사용하면 안 된다.
        faces = []
        for _, _, points, shapes, matrix in selected:
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
                    angles = transform_angles(matrix) if pose_observable and matrix is not None else None
                except (ValueError, IndexError, TypeError, np.linalg.LinAlgError):
                    angles = None
            orientation = periodic_angles(angles)
            if orientation is None:
                angles = None
            # 얼굴 경계 밖의 점을 포함한 형태 점수는 관측 불가로 둔다.
            expression_observable = points_in_frame(points, (1, 13, 14, 33, 61, 105, 133, 263, 291, 334, 362))
            expression = {
                category.category_name: float(category.score)
                for category in shapes
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
        try:
            self.detector.close()
        finally:
            self.tile_detector.close()
