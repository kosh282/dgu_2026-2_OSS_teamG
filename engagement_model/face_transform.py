"""MediaPipe 얼굴 변환행렬에서 카메라 기준 회전 각도를 계산한다."""
import numpy as np
from .rotation_calibration import rotation_euler


def transform_angles(matrix):
    """canonical→runtime 행렬을 기존 카메라 축에 맞춘다. 아래 Pitch는 양수.

    MediaPipe: x 오른쪽, y 위, 카메라는 -z를 본다.
    기존 PnP 카메라: x 오른쪽, y 아래, z 앞쪽. A=diag(1,-1,-1).
    스케일은 SVD로 분리하며 반사·퇴화·심한 비강체 성분은 거부한다.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError('face_transform_invalid')
    linear = matrix[:3, :3]
    u, singular, vt = np.linalg.svd(linear)
    if singular[-1] <= 1e-8 or np.linalg.det(linear) <= 0 or singular[0]/singular[-1] > 1.1:
        raise ValueError('face_transform_not_rigid')
    rotation = np.diag([1., -1., -1.]) @ (u @ vt)
    return rotation_euler(rotation)
