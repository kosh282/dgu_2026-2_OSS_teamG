"""분석 입력 프레임 미리보기와 기준점 시각화."""
import cv2
import numpy as np


# 어깨와 팔, 몸통의 대표 연결선. 얼굴과 상체의 순서는 서로 관련이 없다.
POSE_EDGES = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (11, 23), (12, 24), (23, 24))


def draw_landmarks(frame, face_landmarks, pose_landmarks):
    """원본 프레임은 그대로 두고, 화면 안의 얼굴·상체 기준점을 복사본에 그린다."""
    overlay = frame.copy()
    height, width = overlay.shape[:2]

    def pixel(point):
        x, y = point[:2]
        if not np.isfinite(x) or not np.isfinite(y) or not (0 <= x < 1 and 0 <= y < 1):
            return None
        return (int(x * width), int(y * height))

    for face in face_landmarks:
        for point in face:
            coordinate = pixel(point)
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


