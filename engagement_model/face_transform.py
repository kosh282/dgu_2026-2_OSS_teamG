"""동일 MediaPipe 검출의 얼굴 변환행렬과 PnP를 비교하는 독립 웹캠 실험."""
from collections import deque
from copy import deepcopy
import math
from pathlib import Path
import numpy as np
from video_recognition.face_analysis import DEFAULT_MODEL, periodic_angles
from video_recognition.face_postprocessing import FacePostProcessor
from .gaussian import GaussianRuntime
from .rotation_calibration import rotation_euler
from .pose_validation import ValidatedFaceAnalyzer


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


class _MatrixDetector:
    def __init__(self, detector):
        self.detector = detector
        self.matrices = []

    def detect_for_video(self, *args, **kwargs):
        result = self.detector.detect_for_video(*args, **kwargs)
        self.matrices = result.facial_transformation_matrixes
        return result

    def close(self):
        self.detector.close()


class FaceTransformAnalyzer:
    """기존 검출을 한 번만 수행하며 주 경로의 얼굴 특징은 변경하지 않는다."""
    def __init__(self, delegate, *, validated_pnp=False):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=Path(DEFAULT_MODEL).read_bytes()),
            running_mode=vision.RunningMode.VIDEO, num_faces=20,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True)
        replacement = vision.FaceLandmarker.create_from_options(options)
        delegate.detector.close()
        self.detector = _MatrixDetector(replacement)
        delegate.detector = self.detector
        self.pnp = ValidatedFaceAnalyzer(delegate) if validated_pnp else delegate
        self.transform_faces = []
        self.transform_reason = None
        self.previous_landmarks = None
        self.landmark_motion = None

    def __getattr__(self, name):
        return getattr(self.pnp, name)

    def analyze(self, frame):
        faces = self.pnp.analyze(frame)
        self.transform_faces = deepcopy(faces)
        self.transform_reason = None
        self.landmark_motion = None
        landmarks = getattr(self.pnp, 'last_landmarks', [])
        if len(landmarks) == 1:
            points = np.asarray(landmarks[0], float)[:, :2]
            if self.previous_landmarks is not None and self.previous_landmarks.shape == points.shape:
                old = self.previous_landmarks
                centered = points-points.mean(axis=0)
                old_centered = old-old.mean(axis=0)
                scale = np.sqrt(np.mean(np.sum(centered**2,axis=1)))
                old_scale = np.sqrt(np.mean(np.sum(old_centered**2,axis=1)))
                self.landmark_motion = dict(center_shift=float(np.linalg.norm(points.mean(axis=0)-old.mean(axis=0))),
                    shape_rms=float(np.sqrt(np.mean(np.sum((centered/max(scale,1e-8)-old_centered/max(old_scale,1e-8))**2,axis=1)))))
            self.previous_landmarks = points.copy()
        else:
            self.previous_landmarks = None
        for index, face in enumerate(self.transform_faces):
            try:
                angles = transform_angles(self.detector.matrices[index])
            except (ValueError, IndexError, TypeError, np.linalg.LinAlgError):
                angles = None
                self.transform_reason = 'face_transform_unavailable'
            face['head_angles_degrees'] = angles
            face['head_orientation_periodic'] = periodic_angles(angles)
            # PnP 실패는 독립적인 행렬 측정의 실패를 의미하지 않는다.
            reasons = [r for r in face['quality'].get('reason_codes', []) if r != 'pose_unreliable']
            if angles is None:
                reasons.append('pose_unreliable')
            face['quality'] = dict(status='partial' if reasons else 'valid', reason_codes=reasons)
        return faces


class FaceTransformComparison:
    def __init__(self, calibration, config):
        self.calibration = calibration
        self.processor = FacePostProcessor()
        self.runtime = GaussianRuntime(config)
        self.rows = deque(maxlen=10000)
        self.input_feature = None

    def start(self):
        self.calibration.start()
        self.runtime.reset()
        self.rows.clear()

    def update(self, feature, analyzer, pnp_result, stage='manual', phase='hold', pnp_baseline=None):
        copied = deepcopy(feature)
        faces = deepcopy(analyzer.transform_faces)
        self.processor.process(faces, analyzer.last_face_centers, feature['start_ms'])
        self.calibration.update(faces, feature['start_ms'])
        copied['features']['faces'] = faces
        copied['features']['head_calibration_status'] = self.calibration.status
        copied['features']['head_calibration_diagnostics'] = self.calibration.diagnostics(feature['start_ms'])
        self.runtime(copied)
        self.runtime.latest['angle_estimator'] = 'mediapipe_face_transform'
        self.input_feature = copied
        pnp_faces = feature['features']['faces']
        row = dict(time_ms=feature['start_ms'], stage=stage, phase=phase, channels={},
                   landmark_motion=deepcopy(getattr(analyzer, 'landmark_motion', None)))
        for name, raw_faces, result in (('pnp', pnp_faces, pnp_result),
                                         ('face_transform', faces, self.runtime.latest)):
            row['channels'][name] = dict(raw=raw_faces[0].get('head_angles_degrees') if len(raw_faces)==1 else None,
                smoothed=raw_faces[0].get('head_angles_smoothed_degrees') if len(raw_faces)==1 else None,
                baseline=deepcopy(pnp_baseline if name=='pnp' else self.calibration.baseline),
                relative=deepcopy(result.get('source_angles')), state=result.get('orientation_state'),
                current=result.get('current_direction_score'), score=result.get('score'),
                downward_status=result.get('downward_status'), reasons=list(result.get('reason_codes', [])),
                away_seconds=result.get('away_seconds'))
        self.rows.append(row)
        return row

    def summary(self):
        result = {}
        for stage in sorted({r['stage'] for r in self.rows}):
            rows = [r for r in self.rows if r['stage']==stage and r['phase']=='hold']
            result[stage] = {}
            for name in ('pnp', 'face_transform'):
                samples = [r['channels'][name] for r in rows]
                valid = [r['current'] for r in samples if r['current'] is not None]
                angles = {}
                for channel in ('raw', 'smoothed', 'relative'):
                    angles[channel] = {}
                    for axis in ('pitch', 'yaw'):
                        values = [r[channel][axis] for r in samples if r[channel] is not None]
                        if values:
                            mean = math.degrees(math.atan2(sum(math.sin(math.radians(v)) for v in values),sum(math.cos(math.radians(v)) for v in values)))
                            std = math.sqrt(sum(((v-mean+180)%360-180)**2 for v in values)/len(values))
                            centered = [mean+(v-mean+180)%360-180 for v in values]
                            low, high = min(centered), max(centered)
                        else:
                            mean = std = low = high = None
                        angles[channel][axis] = dict(mean=mean, std=std, count=len(values),
                            min=low, max=high, span=high-low if values else None)
                baselines = {tuple(sorted(r['baseline'].items())) for r in samples if r['baseline'] is not None}
                result[stage][name] = dict(frames=len(samples), null_current=len(samples)-len(valid),
                    mean_current=sum(valid)/len(valid) if valid else None,
                    baseline_changes=max(0,len(baselines)-1),
                    looking_down=sum(r['state']=='looking_down' for r in samples), angles=angles)
        return result

    def recovery_summary(self):
        """정기 로그 사이에 일어난 하향과 복귀도 전체 메모리 관측으로 검증한다."""
        output = {}
        for name in ('pnp', 'face_transform'):
            pending = False
            down = violations = 0
            recoveries = []
            for row in self.rows:
                sample = row['channels'][name]
                if sample['state'] == 'looking_down':
                    pending = True
                    down += 1
                    violations += sample['current'] is not None
                elif pending and sample['state'] == 'front':
                    recoveries.append(dict(time_ms=row['time_ms'], current=sample['current'],
                                           relative=sample['relative']))
                    pending = False
            output[name] = dict(looking_down_frames=down, nonnull_during_down=violations,
                                first_front_recoveries=recoveries, awaiting_front_return=pending)
        return output
