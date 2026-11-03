"""분석 입력 프레임 미리보기와 기준점 시각화."""
from functools import lru_cache

import cv2
import numpy as np


# 어깨와 팔, 몸통의 대표 연결선. 얼굴과 상체의 순서는 서로 관련이 없다.
POSE_EDGES = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (11, 23), (12, 24), (23, 24))


@lru_cache(maxsize=1)
def _face_mesh_connections():
    """MediaPipe Tasks의 연결 정보를 가져온다(얼굴 감지는 다시 실행하지 않음)."""
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

    def edges(connections):
        return tuple((item.start, item.end) for item in connections)

    return {
        'contours': edges(FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS),
        'left_iris': edges(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS),
        'right_iris': edges(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS),
        'tessellation': edges(FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION),
    }


class LandmarkOverlaySmoother:
    """개발자 화면에만 적용하는 얼굴별 1-Euro 랜드마크 스무더."""

    def __init__(self, min_cutoff=1.0, beta=0.05, derivative_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.states = {}

    @staticmethod
    def _alpha(cutoff, frequency):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * frequency)

    def smooth(self, landmarks_by_face, face_features, timestamp_ms):
        """정규화 랜드마크를 임시 얼굴 키별로 스무딩한 새 배열로 반환한다."""
        if len(landmarks_by_face) != len(face_features):
            return landmarks_by_face
        output, seen = [], set()
        for index, (landmarks, face) in enumerate(zip(landmarks_by_face, face_features)):
            key = face.get('transient_track_key', index)
            seen.add(key)
            coords = np.asarray(landmarks, dtype=np.float32)
            state = self.states.get(key)
            delta_ms = timestamp_ms - state['timestamp_ms'] if state else 0
            if (state is None or state['smoothed'].shape != coords.shape
                    or delta_ms <= 0 or delta_ms > 1000):
                smoothed = coords.copy()
                derivative = np.zeros_like(coords)
            else:
                frequency = 1000.0 / delta_ms
                raw_derivative = (coords - state['smoothed']) * frequency
                derivative_alpha = self._alpha(self.derivative_cutoff, frequency)
                derivative = (derivative_alpha * raw_derivative
                              + (1.0 - derivative_alpha) * state['derivative'])
                cutoff = self.min_cutoff + self.beta * np.abs(derivative)
                alpha = self._alpha(cutoff, frequency)
                smoothed = alpha * coords + (1.0 - alpha) * state['smoothed']
            self.states[key] = {
                'smoothed': smoothed,
                'derivative': derivative,
                'timestamp_ms': timestamp_ms,
            }
            output.append(smoothed.tolist())
        for key, state in list(self.states.items()):
            if key not in seen and timestamp_ms - state['timestamp_ms'] > 1000:
                del self.states[key]
        return output

    def reset(self):
        self.states.clear()


def draw_landmarks(frame, face_landmarks, pose_landmarks, *,
                   face_mesh=False, tessellation=False, show_points=True):
    """원본 프레임은 그대로 두고, 화면 안의 얼굴·상체 기준점을 복사본에 그린다."""
    overlay = frame.copy()
    height, width = overlay.shape[:2]

    def pixel(point):
        x, y = point[:2]
        if not np.isfinite(x) or not np.isfinite(y) or not (0 <= x < 1 and 0 <= y < 1):
            return None
        return (int(x * width), int(y * height))

    connections = _face_mesh_connections() if face_mesh and face_landmarks else None
    for face in face_landmarks:
        coordinates = [pixel(point) for point in face]
        if connections:
            if tessellation:
                # 촘촘한 삼각망은 얇고 은은하게 그려 영상과 랜드마크를 가리지 않게 한다.
                mesh_layer = overlay.copy()
                for start, end in connections['tessellation']:
                    if start < len(coordinates) and end < len(coordinates):
                        a, b = coordinates[start], coordinates[end]
                        if a is not None and b is not None:
                            cv2.line(mesh_layer, a, b, (100, 100, 70), 1, cv2.LINE_AA)
                cv2.addWeighted(mesh_layer, 0.28, overlay, 0.72, 0, dst=overlay)
            for start, end in connections['contours']:
                if start < len(coordinates) and end < len(coordinates):
                    a, b = coordinates[start], coordinates[end]
                    if a is not None and b is not None:
                        cv2.line(overlay, a, b, (40, 230, 100), 1, cv2.LINE_AA)
            for group in ('left_iris', 'right_iris'):
                for start, end in connections[group]:
                    if start < len(coordinates) and end < len(coordinates):
                        a, b = coordinates[start], coordinates[end]
                        if a is not None and b is not None:
                            cv2.line(overlay, a, b, (0, 230, 255), 2, cv2.LINE_AA)
        if show_points:
            for coordinate in coordinates:
                if coordinate is not None:
                    cv2.circle(overlay, coordinate, 1, (0, 255, 0), -1)

    for pose in pose_landmarks:
        # 화면 밖에 있거나 가시성이 낮은 점은 화면 테두리로 억지 보정하지 않는다.
        coordinates = {
            index: pixel(point)
            for index, point in enumerate(pose)
            if len(point) >= 3 and point[2] >= 0.5
        }
        for start, end in POSE_EDGES:
            if coordinates.get(start) is not None and coordinates.get(end) is not None:
                cv2.line(overlay, coordinates[start], coordinates[end], (0, 200, 255), 2)
        for index in range(11, 25):
            if coordinates.get(index) is not None:
                cv2.circle(overlay, coordinates[index], 3, (0, 200, 255), -1)
    return overlay


def draw_face_postprocessing(frame, faces):
    """개발자 화면에만 임시 얼굴 키와 스무딩/깜빡임 신호를 겹쳐 그린다."""
    overlay = frame.copy()
    height, width = overlay.shape[:2]
    for index, face in enumerate(faces):
        position = face.get('head_position') or {}
        if 'x' not in position or 'y' not in position:
            continue
        x = min(max(int(position['x'] * width), 4), width - 4)
        y = min(max(int(position['y'] * height), 20), height - 4)
        key = face.get('transient_track_key', index)
        angles = face.get('head_angles_smoothed_degrees') or {}
        blink = face.get('blink_metrics') or {}
        pose_text = ('smooth P/Y/R: ' + ', '.join(
            f'{angles.get(axis, 0):.0f}' for axis in ('pitch', 'yaw', 'roll'))
            if angles else 'smooth pose: unavailable')
        blink_text = f"T{key} | blink {blink.get('total_count', 0)} / 60s"
        for line_number, label in enumerate((blink_text, pose_text)):
            text_y = max(18, y - 34 + line_number * 18)
            cv2.putText(overlay, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(overlay, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return overlay


class PreviewCapture:
    """실제 분석기가 받을 동일한 캡처 프레임을 미리보기용으로 보관한다."""

    def __init__(self, capture):
        self.capture = capture
        self.last_frame = None

    def read(self):
        frame = self.capture.read()
        # 원본을 수정하지 않고 메모리에만 보관한다. 별도 재캡처나 파일 저장은 하지 않는다.
        self.last_frame = frame.copy()
        return frame

    def close(self):
        if hasattr(self.capture, 'close'):
            self.capture.close()


