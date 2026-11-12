"""독립 웹캠 전용 PnP 검증. 공통 영상 분석기의 눈·표정·검출 기능을 재사용한다."""
from dataclasses import dataclass
from copy import deepcopy
import math
import cv2
import numpy as np
from video_recognition.face_analysis import periodic_angles


@dataclass(frozen=True)
class PoseValidationConfig:
    # 정답 데이터로 최적화하지 않은 실험용 품질 기준이다.
    max_reprojection_diagonal_ratio: float = .02
    max_rotation_speed_degrees: float = 720.
    min_jump_degrees: float = 45.
    continuity_gap_ms: int = 300


class ValidatedPose:
    def __init__(self, config=None):
        self.config = config or PoseValidationConfig()
        self.previous = None
        self.previous_ms = None

    def estimate(self, points, width, height, timestamp_ms):
        # 공통 head_angles()와 동일한 6점·얼굴 모형·카메라 근사를 사용한다.
        model = np.array([[0, 0, 0], [0, -330, -65], [-225, 170, -135],
                          [225, 170, -135], [-150, -150, -125], [150, -150, -125]], float)
        image = np.array([[points[i][0]*width, points[i][1]*height]
                          for i in (1, 152, 33, 263, 61, 291)], float)
        info = dict(status='pose_unreliable', reason=None)
        if not np.isfinite(image).all() or (image < 0).any() or (image > [width, height]).any():
            info['reason'] = 'pose_landmarks_invalid'
            return None, info
        camera = np.array([[width, 0, width/2], [0, width, height/2], [0, 0, 1]], float)
        try:
            old_ok, old_r, old_t = cv2.solvePnP(model, image, camera, np.zeros(4))
            if old_ok:
                old_rotation = cv2.Rodrigues(old_r)[0]
                old_projected = cv2.projectPoints(model, old_r, old_t, camera, np.zeros(4))[0].reshape(-1, 2)
                info.update(original_min_depth=float(((old_rotation @ model.T).T + old_t.reshape(3))[:,2].min()),
                            original_rmse_pixels=float(np.sqrt(np.mean(np.sum((old_projected-image)**2, axis=1)))))
            ok, rvec, tvec = cv2.solvePnP(model, image, camera, np.zeros(4), flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                info['reason'] = 'pose_solver_failed'
                return None, info
            # SQPnP만 무조건 선택하지 않는다. 같은 점들의 픽셀 오차를 직접 최소화한
            # 해와 기존 iterative 해를 함께 비교하되 음의 깊이는 후보에서 제외한다.
            candidates = [('sqpnp', rvec, tvec)]
            if old_ok:
                candidates.append(('iterative', old_r, old_t))
            for name, seed_r, seed_t in list(candidates):
                seed_rotation = cv2.Rodrigues(seed_r)[0]
                if ((seed_rotation @ model.T).T + seed_t.reshape(3))[:, 2].min() <= 0:
                    continue
                try:
                    refined_r, refined_t = cv2.solvePnPRefineLM(
                        model, image, camera, np.zeros(4), seed_r.copy(), seed_t.copy())
                    candidates.append((name + '_refined', refined_r, refined_t))
                except cv2.error:
                    pass
            ranked = []
            for name, candidate_r, candidate_t in candidates:
                candidate_rotation = cv2.Rodrigues(candidate_r)[0]
                if not np.isfinite(candidate_rotation).all() or not np.isfinite(candidate_t).all():
                    continue
                candidate_depth = ((candidate_rotation @ model.T).T + candidate_t.reshape(3))[:, 2]
                if candidate_depth.min() <= 0:
                    continue
                candidate_projected = cv2.projectPoints(model, candidate_r, candidate_t, camera, np.zeros(4))[0].reshape(-1, 2)
                error = float(np.sqrt(np.mean(np.sum((candidate_projected-image)**2, axis=1))))
                if math.isfinite(error):
                    ranked.append((error, name, candidate_r, candidate_t))
            if not ranked:
                info['reason'] = 'pose_nonpositive_depth'
                return None, info
            error, selected, rvec, tvec = min(ranked, key=lambda row: row[0])
            info.update(selected_solver=selected,
                        candidate_rmse_pixels={name: error for error, name, _, _ in ranked})
            rotation = cv2.Rodrigues(rvec)[0]
            depth = ((rotation @ model.T).T + tvec.reshape(3))[:, 2]
            projected = cv2.projectPoints(model, rvec, tvec, camera, np.zeros(4))[0].reshape(-1, 2)
            rmse = float(np.sqrt(np.mean(np.sum((projected-image)**2, axis=1))))
            info.update(min_depth=float(depth.min()), reprojection_rmse_pixels=rmse,
                        rvec=rvec.reshape(3).tolist(), tvec=tvec.reshape(3).tolist(),
                        rotation=rotation.tolist(), jump_degrees=None)
            if not np.isfinite(rotation).all() or not np.isfinite(tvec).all() or depth.min() <= 0:
                info['reason'] = 'pose_nonpositive_depth'
                return None, info
            if not math.isfinite(rmse) or rmse > self.config.max_reprojection_diagonal_ratio*math.hypot(width, height):
                info['reason'] = 'pose_reprojection_error'
                return None, info
            if self.previous is not None and 0 < timestamp_ms-self.previous_ms <= self.config.continuity_gap_ms:
                jump = math.degrees(math.acos(float(np.clip((np.trace(rotation @ self.previous.T)-1)/2, -1, 1))))
                info['jump_degrees'] = jump
                limit = max(self.config.min_jump_degrees,
                            self.config.max_rotation_speed_degrees*(timestamp_ms-self.previous_ms)/1000)
                if jump > limit:
                    info['reason'] = 'pose_rotation_discontinuity'
                    return None, info
            self.previous, self.previous_ms = rotation, timestamp_ms
            angles = dict(zip(('pitch', 'yaw', 'roll'), map(float, cv2.RQDecomp3x3(rotation)[0])))
            info.update(status='valid', reason=None)
            return angles, info
        except cv2.error:
            info['reason'] = 'pose_solver_failed'
            return None, info


class ValidatedFaceAnalyzer:
    """기존 분석기를 감싸 단일 얼굴 자세만 검증한 값으로 교체한다."""
    def __init__(self, delegate, *, apply_validated=True):
        self.delegate = delegate
        self.apply_validated = apply_validated
        self.original_faces = []
        self.validated_faces = []
        self.estimator = ValidatedPose()
        self.diagnostics = None

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def analyze(self, frame):
        faces = self.delegate.analyze(frame)
        self.original_faces = deepcopy(faces)
        self.diagnostics = None
        if len(faces) != 1 or len(self.delegate.last_landmarks) != 1:
            self.estimator = ValidatedPose(self.estimator.config)
            self.validated_faces = deepcopy(faces)
            return faces
        face = faces[0]
        original = face.get('head_angles_degrees')
        angles, info = self.estimator.estimate(self.delegate.last_landmarks[0], frame.width, frame.height, frame.capture_ms)
        # 회전·이동 벡터는 실행 메모리 전용이며 FeatureWindow/저장 JSON에 추가하지 않는다.
        self.diagnostics = dict(original=original, validated=angles, **info)
        face['head_angles_degrees'] = angles
        face['head_orientation_periodic'] = periodic_angles(angles)
        if angles is None:
            face['quality']['status'] = 'partial'
            face['quality']['reason_codes'] = list(dict.fromkeys(face['quality']['reason_codes'] + ['pose_unreliable']))
        self.validated_faces = deepcopy(faces)
        return faces if self.apply_validated else deepcopy(self.original_faces)
