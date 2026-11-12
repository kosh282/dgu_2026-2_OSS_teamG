"""기존 영상 파이프라인을 재사용하는 독립 모델 비교 실행기. 기본은 저장하지 않으며 --yaw-log는 집계만 저장한다.

수동 보정·좌우 자동 진단:
    python -B -m engagement_model.webcam_demo --yaw-diagnostic --single-model
정면 준비 후 CALIBRATE 버튼/C/Space를 한 번 누른다. 이후 정면→본인 왼쪽→정면→
본인 오른쪽→정면을 자동 안내한다. 각 구간은 기본 이동 2초, 유지 3초이며 유지 표본만
요약한다. --diagnostic-move-seconds / --diagnostic-hold-seconds로 안내 시간을 조정한다.
기본 수치 기록은 메모리에만 있고 완료/중단 시 터미널 요약을 출력한다. Q/Esc로 종료하며,
새 수동 보정은 이전 진단을 종료하고 메모리를 새로 시작한다. 자동 보정 옵션과 병용하지 않는다.
"""

import argparse
from collections import Counter, deque
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import textwrap
import time
from uuid import uuid4

import cv2
import numpy as np

from integration import IntegrationService
from video_recognition.face_analysis import FaceAnalyzer
from video_recognition.pose_analysis import PoseAnalyzer
from video_recognition.integrated_analysis import VisionPipeline
from video_recognition.head_calibration import HeadCalibration, AXES
from video_recognition.preview import PreviewCapture, LandmarkOverlaySmoother, draw_landmarks
from video_recognition.developer_view import render_video_frame
from video_recognition.webcam_test_mac import WebcamCapture

from .config import ModelConfig
from .runtime import EngagementRuntime, ComparisonRuntime
from .gaussian import GaussianRuntime, gaussian_angle_score, gaussian_score_angles
from .input_adapter import HeadAngles
from .rotation_calibration import RotationCalibration, RotationGaussianRuntime, rotation_euler
from .pose_validation import ValidatedFaceAnalyzer
from .face_transform import FaceTransformAnalyzer, FaceTransformComparison


WINDOW = 'Engagement comparison - C/Space calibrate | V overlay | Q quit'
CALIBRATE_BUTTON = (16, 430, 290, 474)


class CalibrationControls:
    """마우스 콜백은 요청만 전달하고 실제 보정 시작은 분석 루프에서 수행한다."""

    def __init__(self, requested=False):
        self.requested = requested
        self.last_key_request = None

    def mouse(self, event, x, y, flags, userdata):
        left, top, right, bottom = CALIBRATE_BUTTON
        if event == cv2.EVENT_LBUTTONDOWN and left <= x <= right and top <= y <= bottom:
            self.requested = True

    def consume(self, key):
        key_request = key in (ord('c'), ord('C'), ord(' '))
        if key_request:
            now = time.monotonic()
            repeated = self.last_key_request is not None and now - self.last_key_request < 0.6
            self.last_key_request = now
            key_request = not repeated
        requested = self.requested or key_request
        self.requested = False
        return requested

REASON_HELP = {
    'baseline_unavailable': 'Press C, face forward and hold still for 3 seconds.',
    'relative_angles_unavailable': 'Relative angles need a valid calibration.',
    'huang_warmup': 'Waiting for a complete 3-second interval.',
    'gaussian_warmup': 'Waiting for elapsed observation time; Gaussian uses a 10-second time kernel.',
    'huang_recovery_warmup': 'Front-facing resumed; collecting a fresh 3-second interval.',
    'downward_activity_uncertain': 'Writing/reading cannot be identified; downward activity is deferred.',
    'huang_degenerate_pitch_range': 'Pitch normalization denominator is zero; affected intervals deferred.',
    'insufficient_scoreable_time': 'Too little scoreable time in the rolling window (minimum 70%).',
    'face_not_found': 'No face detected. Waiting for return; a completed calibration is retained.',
    'multiple_faces_unsupported': 'Only one face is supported.',
}


class WebcamHeadCalibration(HeadCalibration):
    """단일 사용자 데모에서 미검출만으로 완료된 보정을 지우지 않는다.

    신원 재식별은 하지 않는다. 사용자가 바뀌거나 카메라를 옮기면 C로 재보정한다.
    단일 얼굴의 관측 실패는 기준점 손상이 아니다. 현재 상대각만 비운다.
    다중 얼굴과 수집 중 검출 공백은 기존 보정기의 정책을 따른다.
    """

    def start(self):
        super().start()
        self.diagnostic_samples = []

    def update(self, faces, capture_ms):
        # 완료 시 상위 구현이 지우는 실제 채택 표본만 숫자로 복사한다. 보정 계산은 그대로다.
        collecting = self.status == 'collecting'
        previous = list(self.samples) if collecting else []
        status = self._update(faces, capture_ms)
        if collecting:
            if status == 'ready':
                angles = faces[0].get('head_angles_smoothed_degrees') or faces[0]['head_angles_degrees']
                self.diagnostic_samples = deepcopy(previous + [(capture_ms, dict(angles))])
            else:
                self.diagnostic_samples = deepcopy(self.samples)
        return status

    def _update(self, faces, capture_ms):
        if self.status == 'ready':
            valid_baseline = isinstance(self.baseline, dict) and all(
                not isinstance(self.baseline.get(axis), bool)
                and isinstance(self.baseline.get(axis), (int, float))
                and math.isfinite(self.baseline[axis])
                and -180 <= self.baseline[axis] <= 180
                for axis in AXES
            )
            if not valid_baseline:
                self.invalidate('calibration_baseline_invalid')
            else:
                reason = self._input_reason(faces)
                if reason and len(faces) <= 1:
                    for face in faces:
                        face['head_angles_relative_degrees'] = None
                        face['head_angles_relative_raw_degrees'] = None
                    self.last_reason_code = reason
                    return self.status
                self.last_reason_code = None
        return super().update(faces, capture_ms)


def diagnostic_angles(value):
    """허용한 축 숫자만 복사한다. 미관측을 0도로 채우지 않는다."""
    return {axis: (float(value[axis]) if isinstance(value, dict)
                   and not isinstance(value.get(axis), bool)
                   and isinstance(value.get(axis), (int, float))
                   and math.isfinite(value[axis]) else None) for axis in AXES}


def diagnostic_angle_stats(values):
    """±180도 경계를 고려한 표본 평균·편차. 유지 지시가 실제 안정성을 보장하지는 않는다."""
    values = [v for v in values if v is not None]
    if not values:
        return dict(count=0, mean=None, std=None, min=None, max=None)
    x = sum(math.cos(math.radians(v)) for v in values) / len(values)
    y = sum(math.sin(math.radians(v)) for v in values) / len(values)
    mean = math.degrees(math.atan2(y, x)) if math.hypot(x, y) > 1e-6 else None
    std = (math.sqrt(sum(((v-mean+180) % 360-180)**2 for v in values) / len(values))
           if mean is not None else None)
    return dict(count=len(values), mean=mean, std=std, min=min(values), max=max(values))


class DualWebcamCalibration(WebcamHeadCalibration):
    """기존 모델용 축별 보정을 유지하면서 Gaussian용 행렬 보정을 병렬 갱신한다."""

    def __init__(self, rotation):
        super().__init__()
        self.rotation = rotation

    def start(self):
        super().start()
        self.rotation.start()

    def update(self, faces, capture_ms):
        self.rotation.update(faces, capture_ms)
        return super().update(faces, capture_ms)


class PoseInputDiagnostics:
    """단일 얼굴의 입력 흔들림과 기존 PnP 재투영 오차를 진단 전용으로 계산한다."""

    def __init__(self):
        self.previous = None

    def reset(self):
        self.previous = None

    def observe(self, landmarks, frame):
        empty = {'reprojection_error_px': None, 'reprojection_error_face_ratio': None,
                 'landmark_shape_change_ratio': None, 'face_width_px': None}
        if frame is None or len(landmarks) != 1 or len(landmarks[0]) <= 291:
            self.reset()
            return empty
        height, width = frame.shape[:2]
        points = np.asarray([[landmarks[0][i][0] * width, landmarks[0][i][1] * height]
                             for i in (1, 152, 33, 263, 61, 291)], dtype=np.float64)
        if not np.isfinite(points).all() or np.any(points < 0) or np.any(points[:, 0] > width) or np.any(points[:, 1] > height):
            self.reset()
            return empty
        # 양 눈 사이 거리는 얼굴 크기의 근사치다. 좌표 원본은 파일에 저장하지 않는다.
        scale = float(np.linalg.norm(points[2] - points[3]))
        if scale < 1e-6:
            self.reset()
            return empty
        shape = (points - points.mean(axis=0)) / scale
        change = float(np.linalg.norm(shape - self.previous, axis=1).mean()) if self.previous is not None else None
        self.previous = shape
        model = np.array([[0, 0, 0], [0, -330, -65], [-225, 170, -135],
                          [225, 170, -135], [-150, -150, -125], [150, -150, -125]], dtype=np.float64)
        camera = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float64)
        try:
            ok, rotation, translation = cv2.solvePnP(model, points, camera, np.zeros((4, 1)))
            if not ok:
                return {**empty, 'landmark_shape_change_ratio': change, 'face_width_px': scale}
            projected, _ = cv2.projectPoints(model, rotation, translation, camera, np.zeros((4, 1)))
            error = float(np.linalg.norm(projected.reshape(-1, 2) - points, axis=1).mean())
            if not math.isfinite(error):
                error = None
        except cv2.error:
            error = None
        return {'reprojection_error_px': error,
                'reprojection_error_face_ratio': error / scale if error is not None else None,
                'landmark_shape_change_ratio': change, 'face_width_px': scale}


POSE_DIAGNOSTIC_FIELDS = ('reprojection_error_px', 'reprojection_error_face_ratio',
                          'landmark_shape_change_ratio', 'face_width_px')


def scalar_stats(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return {'count': 0, 'mean': None, 'max': None}
    return {'count': len(values), 'mean': sum(values) / len(values), 'max': max(values)}


class GuidedYawDiagnostic:
    """수동 보정 후 구간을 자동 안내한다. 숫자만 메모리에 두고 계약에는 전달하지 않는다."""

    STAGES = ('front_initial', 'left', 'front_after_left', 'right', 'front_after_right')
    LABELS = ('FACE FORWARD', 'TURN TO YOUR LEFT', 'RETURN TO FRONT',
              'TURN TO YOUR RIGHT', 'RETURN TO FRONT')

    def __init__(self, move_seconds=2, hold_seconds=3, report_sink=None, *, include_down=False, front_stability=False):
        if front_stability:
            self.STAGES = self.STAGES[:1] + ('front_small_motion', 'front_return') + self.STAGES[1:]
            self.LABELS = self.LABELS[:1] + ('FACE CAMERA: SMALL BODY SHIFT ONLY', 'FACE FORWARD, STAY STILL') + self.LABELS[1:]
        if include_down:
            self.STAGES = self.STAGES + ('down', 'front_after_down')
            self.LABELS = self.LABELS + ('LOOK DOWN: RELATIVE PITCH > 15 DEG', 'RETURN TO FRONT')

        if any(not math.isfinite(v) or v <= 0 or round(v*1000) < 1
               for v in (move_seconds, hold_seconds)):
            raise ValueError('진단 이동/유지 시간은 유한한 양수이며 최소 1ms여야 합니다')
        self.move_ms, self.hold_ms = round(move_seconds*1000), round(hold_seconds*1000)
        self.state = 'waiting'
        self.rows = deque(maxlen=10000)
        self.started_ms = None
        self.baseline = None
        self.calibration_samples = []
        self.last_track = None
        self.truncated = False
        self.reported = False
        self.report = None
        self.model_info = None
        self.calibration_range = None
        self.report_sink = report_sink

    def calibration_started(self):
        if self.state == 'running':
            self.finish('사용자 재보정으로 중단')
        self.rows.clear()
        self.calibration_samples = []
        self.started_ms = None
        self.baseline = None
        self.last_track = None
        self.truncated = self.reported = False
        self.report = None
        self.model_info = None
        self.calibration_range = None
        self.state = 'collecting'

    def phase(self, timestamp):
        if self.started_ms is None:
            return None
        elapsed = max(0, timestamp-self.started_ms)
        span = self.move_ms+self.hold_ms
        index, within = divmod(elapsed, span)
        if index >= len(self.STAGES):
            return None
        moving = within < self.move_ms
        remaining = (self.move_ms if moving else span)-within
        return index, 'move' if moving else 'hold', remaining

    def tick(self, timestamp):
        if self.state == 'running' and self.phase(timestamp) is None:
            self.finish('완료')

    def update(self, feature, model, calibration, pose_diagnostics=None):
        timestamp = feature['start_ms']
        if self.state == 'collecting':
            self.calibration_samples = deepcopy(getattr(calibration, 'diagnostic_samples', []))
        if self.state == 'collecting' and calibration.status == 'ready':
            self.baseline = diagnostic_angles(calibration.baseline)
            self.calibration_range = deepcopy(getattr(calibration, 'observed_range', None))
            self.model_info = dict(model_id=model.model_id, model_version=model.model_version,
                config_version=model.config.config_version, angle_mode=model.config.gaussian_angle_mode,
                calibration_method=model.latest.get('calibration_method', 'axiswise_euler'),
                angle_estimator=model.latest.get('angle_estimator', 'pnp'),
                sigma_pitch=model.config.gaussian_sigma_pitch_degrees,
                sigma_yaw=model.config.gaussian_sigma_yaw_degrees)
            self.calibration_samples = deepcopy(getattr(calibration, 'diagnostic_samples', []))
            self.started_ms = timestamp
            self.state = 'running'
            print('좌우 진단 시작: 정면 → 본인 왼쪽 → 정면 → 본인 오른쪽 → 정면. 비슷한 크기로 돌려주세요.', flush=True)
        self.tick(timestamp)
        if self.state != 'running':
            return
        if calibration.status != 'ready' or diagnostic_angles(calibration.baseline) != self.baseline:
            self.finish('보정 무효화 또는 기준 변경으로 중단; 다시 수동 보정 필요')
            return
        index, phase, _ = self.phase(timestamp)
        features = feature['features']
        faces = features.get('faces', [])
        face = faces[0] if len(faces) == 1 else {}
        track = face.get('transient_track_key')
        changed = self.last_track is not None and track is not None and track != self.last_track
        if track is not None:
            self.last_track = track
        primitive = model.latest.get('gaussian_observation')
        used = dict(pitch=primitive[0], yaw=primitive[1], roll=None) if primitive is not None else None
        quality = face.get('quality') or {}
        top_quality = feature.get('quality') or {}
        row = dict(timestamp_ms=timestamp, stage=self.STAGES[index], phase=phase,
            raw=diagnostic_angles(face.get('head_angles_degrees')),
            smoothed=diagnostic_angles(face.get('head_angles_smoothed_degrees')),
            relative=diagnostic_angles(face.get('head_angles_relative_degrees')),
            used=diagnostic_angles(used), baseline=dict(self.baseline),
            face_count=len(faces), tracking_quality=face.get('tracking_quality'),
            track_available=track is not None, track_changed=changed,
            quality_status=quality.get('status'), quality_reasons=list(quality.get('reason_codes', [])),
            window_quality=top_quality.get('status'),
            angle_source=face.get('head_angle_model_input_source'),
            pose_diagnostics=pose_diagnostics or {},
            current_score=model.latest.get('current_direction_score'),
            orientation=model.latest.get('orientation_state'), downward_status=model.latest.get('downward_status'))
        # 단계 간 산술 일치만 진단한다. 관측의 물리적 정확성을 판정하지 않는다.
        matrix = model.latest.get('relative_rotation_matrix')
        matrix_angles = rotation_euler(matrix) if matrix is not None else None
        source = row['smoothed'] if row['angle_source'] != 'raw_fallback' else row['raw']
        row['relative_formula_error'] = {axis: (
            row['relative'][axis]-matrix_angles[axis] if matrix_angles is not None else
            (row['relative'][axis] - ((source[axis]-self.baseline[axis]+180) % 360-180)+180) % 360-180
            if all(v is not None for v in (source[axis], self.baseline[axis], row['relative'][axis])) else None)
            for axis in AXES}
        row['used_formula_error'] = dict(pitch=None, yaw=None)
        if primitive is not None and all(row['relative'][axis] is not None for axis in AXES):
            expected = gaussian_score_angles(HeadAngles(**row['relative']), model.config)
            row['used_formula_error'] = {axis: row['used'][axis]-getattr(expected, axis) for axis in ('pitch', 'yaw')}
        if len(self.rows) == self.rows.maxlen:
            self.truncated = True
        self.rows.append(row)

    def screen_lines(self, timestamp):
        if self.state == 'waiting':
            return ['YAW TEST: Face forward, then click CALIBRATE or press C/Space.',
                    'Keep camera fixed. Manual calibration required.']
        if self.state == 'collecting':
            return ['YAW TEST: CALIBRATING - keep facing forward and stay still.']
        if self.state == 'running':
            phase = self.phase(timestamp)
            if phase is None:
                return ['YAW TEST: finishing...']
            index, part, remaining = phase
            return [f'YAW TEST {index+1}/{len(self.STAGES)}: {self.LABELS[index]}',
                    f'{"MOVE / SETTLE" if part == "move" else "HOLD STILL"}: {math.ceil(remaining/1000)}s',
                    'Match left/right turn size; follow instructions, do not recalibrate.']
        return ['YAW TEST FINISHED - summary printed in terminal.',
                'C/Space or CALIBRATE starts a new trial. Q/Esc quits.']

    def summary(self, reason):
        stages = {}
        for name in self.STAGES:
            rows = [r for r in self.rows if r['stage'] == name]
            hold = [r for r in rows if r['phase'] == 'hold']
            valid = [r for r in hold if r['face_count'] == 1 and r['quality_status'] in ('valid', 'partial')
                     and r['window_quality'] in ('valid', 'partial')]
            scores = [r['current_score'] for r in valid if r['current_score'] is not None]
            stages[name] = dict(move_count=sum(r['phase']=='move' for r in rows), hold_count=len(hold),
                valid_hold_count=len(valid), track_changes=sum(r['track_changed'] for r in rows),
                orientation_counts={state: sum(r.get('orientation') == state for r in hold)
                                    for state in sorted({r.get('orientation') for r in hold if r.get('orientation') is not None})},
                null_current_count=sum(r['current_score'] is None for r in hold),
                score_mean=sum(scores)/len(scores) if scores else None,
                score_valid_ratio=len(scores)/len(hold) if hold else None,
                channels={channel: {axis: diagnostic_angle_stats([r[channel][axis] for r in valid])
                                    for axis in AXES} for channel in ('raw', 'smoothed', 'relative', 'used')})
        pairs = {}
        for channel in ('raw', 'smoothed', 'relative', 'used'):
            means = [stages[s]['channels'][channel]['yaw']['mean'] for s in ('front_initial', 'left', 'right')]
            front, left, right = means
            dl = (left-front+180) % 360-180 if left is not None and front is not None else None
            dr = (right-front+180) % 360-180 if right is not None and front is not None else None
            pairs[channel] = dict(front_mean=front, left_mean=left, right_mean=right,
                left_deviation_from_front=dl, right_deviation_from_front=dr,
                absolute_deviation_difference=abs(abs(dl)-abs(dr)) if dl is not None and dr is not None else None,
                absolute_mean_difference=abs(abs(left)-abs(right)) if left is not None and right is not None else None,
                opposite_sides=dl*dr < 0 if dl is not None and dr is not None else None)
        errors = {}
        for key in ('relative_formula_error', 'used_formula_error'):
            values = [abs(v) for row in self.rows for v in row[key].values() if v is not None]
            errors[key+'_max'] = max(values) if values else None
        front = stages['front_initial']['channels']['smoothed']['yaw']['mean']
        baseline_yaw = self.baseline.get('yaw') if self.baseline else None
        errors['baseline_minus_initial_front_yaw'] = ((baseline_yaw-front+180) % 360-180
            if baseline_yaw is not None and front is not None else None)
        if self.model_info and self.model_info.get('calibration_method') == 'rotation_matrix':
            errors['baseline_minus_initial_front_yaw'] = None
        left_score, right_score = stages['left']['score_mean'], stages['right']['score_mean']
        returns = {}
        for name in (stage for stage in self.STAGES if stage.startswith('front')):
            values = [r['relative']['yaw'] for r in self.rows if r['stage'] == name and r['phase'] == 'hold'
                      and r['face_count'] == 1 and r['quality_status'] in ('valid', 'partial')
                      and r['window_quality'] in ('valid', 'partial') and r['relative']['yaw'] is not None]
            returns[name] = {**stages[name]['channels']['relative']['yaw'],
                'mean_absolute_error_from_zero': sum(abs(v) for v in values)/len(values) if values else None}
        return dict(reason=reason, completed=reason=='완료', truncated=self.truncated,
            all_stages_have_valid_hold_observations=all(s['valid_hold_count'] > 0 for s in stages.values()),
            timing=dict(move_seconds=self.move_ms/1000, hold_seconds=self.hold_ms/1000),
            model=self.model_info,
            baseline=self.baseline, calibration_range=self.calibration_range,
            calibration_count=len(self.calibration_samples),
            calibration_distribution={axis: diagnostic_angle_stats([sample[1].get(axis) for sample in self.calibration_samples])
                                      for axis in AXES}, stages=stages, yaw_comparison=pairs,
            front_return_yaw=returns,
            score_mean_difference=abs(left_score-right_score) if left_score is not None and right_score is not None else None,
            consistency=errors)

    def finish(self, reason):
        if self.reported:
            return
        self.reported = True
        self.state = 'done'
        self.report = self.summary(reason)
        if self.report_sink is not None:
            try:
                path = self.report_sink(self.report, self.rows, self.started_ms)
                print(f'좌우 진단 요약 저장: {path}', flush=True)
            except (OSError, TypeError, ValueError) as exc:
                print(f'좌우 진단 로그 저장 실패: {type(exc).__name__}', flush=True)
        print('\n=== 좌우 Yaw 진단 요약 ===', flush=True)
        for stage, stats in self.report['stages'].items():
            print(f"{stage}: hold={stats['hold_count']}, valid={stats['valid_hold_count']}, "
                  f"score_mean={stats['score_mean']}, score_valid_ratio={stats['score_valid_ratio']}, "
                  f"track_changes={stats['track_changes']}", flush=True)
            for channel, axes in stats['channels'].items():
                text = ' / '.join(f"{axis}:mean={axes[axis]['mean']},std={axes[axis]['std']},n={axes[axis]['count']}" for axis in AXES)
                print(f'  {channel} {text}', flush=True)
        print(json.dumps({key: value for key, value in self.report.items() if key != 'stages'},
                         ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        print('해석: raw/smoothed는 최초 정면 대비 편차로 좌우를 비교하세요. 0도 자체가 정면이라는 보장은 없습니다.\n'
              'relative_formula_error는 보정 산술, used_formula_error는 완화 산술의 일치도입니다.\n'
              'baseline_minus_initial_front_yaw와 front_return_yaw로 정면 재현성을 확인하세요.\n'
              'hold는 유지 안내 구간입니다. 표준편차·표본 수·결측·추적 변경도 함께 보세요.\n'
              '진짜 좌우 회전량은 측정하지 않았으므로 원본 비대칭을 solvePnP 오류로 확정할 수 없습니다.\n'
              '재투영 오차와 기준점 변화는 --yaw-log 사용 시 초별 로그에서 확인합니다.\n'
              'used.roll=null: Gaussian은 Roll을 점수에 사용하지 않습니다.', flush=True)


def save_yaw_diagnostic(report, rows, started_ms, directory=None):
    """개별 프레임이나 식별정보 대신 단계별·1초 단위 각도 통계만 선택 저장한다."""
    directory = Path(directory) if directory else Path(__file__).resolve().parent / 'diagnostic_logs'
    directory.mkdir(parents=True, exist_ok=True)
    buckets = {}
    for row in rows:
        second = max(0, (row['timestamp_ms'] - started_ms) // 1000) if started_ms is not None else 0
        key = (second, row['stage'], row['phase'])
        buckets.setdefault(key, []).append(row)
    timeline = []
    for (second, stage, phase), samples in sorted(buckets.items()):
        valid = [r for r in samples if r['face_count'] == 1 and r['quality_status'] in ('valid', 'partial')
                 and r['window_quality'] in ('valid', 'partial')]
        timeline.append({'elapsed_second': second, 'stage': stage, 'phase': phase,
            'sample_count': len(samples), 'valid_count': len(valid),
            'track_changes': sum(bool(r['track_changed']) for r in samples),
            'yaw': {channel: diagnostic_angle_stats([r[channel]['yaw'] for r in valid])
                    for channel in ('raw', 'smoothed', 'relative', 'used')},
            'pose_input': {key: scalar_stats([r.get('pose_diagnostics', {}).get(key) for r in valid])
                           for key in POSE_DIAGNOSTIC_FIELDS}})
    payload = {'format_version': 2, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'summary': report, 'timeline': timeline}
    path = directory / f'yaw-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}.json'
    with path.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return path


def model_lines(name, result):
    """실제 score만 표시하며 화면용 과거 점수를 현재 점수로 대체하지 않는다."""
    score = result.get('score')
    value = f'{score:.1f}/100' if score is not None else 'DEFERRED (no current score)'
    label = '10s aggregate' if 'current_direction_score' in result else 'score'
    lines = [f'{name} {label}: {value}',
             f"state={result.get('orientation_state')} | status={result.get('status')}",
             f"front={result.get('front_seconds', 0):.1f}s  away={result.get('away_seconds', 0):.1f}s  down={result.get('looking_down_seconds', 0):.1f}s"]
    reasons = result.get('reason_codes', [])
    lines.append('Reasons: ' + (', '.join(reasons) or 'none'))
    lines.extend(REASON_HELP[reason] for reason in reasons if reason in REASON_HELP)
    if 'current_direction_score' in result:
        current = result['current_direction_score']
        lines.insert(0, f'{name} current direction: {current:.1f}/100' if current is not None
                     else f'{name} current direction: unavailable')
    angles = result.get('source_angles') or result.get('angles')
    if angles:
        lines.append(f"Relative pitch/yaw: {angles['pitch']:.2f}/{angles['yaw']:.2f} deg")
    return lines


class GaussianDiagnostics:
    """실제 입력과 축별 기여를 메모리에서만 진단한다. 산정·JSON은 변경하지 않는다."""

    def __init__(self):
        self.previous = None
        self.drops = deque(maxlen=100)
        self.lines = []
        self.events = deque(maxlen=100)
        self.last_track = None
        self.previous_snapshot = None

    def reset(self):
        self.previous = None
        self.drops.clear()
        self.lines = []
        self.events.clear()
        self.last_track = None
        self.previous_snapshot = None

    def _show_recent_event(self, timestamp):
        while self.events and self.events[0]['timestamp_ms'] < timestamp - 8000:
            self.events.popleft()
        if self.events:
            event = max(self.events, key=lambda item: item['drop'])
            self.lines.append(f"HISTORICAL DROP (8s hold): t={event['timestamp_ms']}ms drop={event['drop']:.1f}; driver={event['driver']}")
            for label in ('before', 'after'):
                snapshot = event[label]
                self.lines.append(f"  {label}: track={snapshot['track']} baseline={snapshot['baseline']} current={snapshot['current']}")
                self.lines.append(f"  raw={snapshot['raw']} smooth={snapshot['smoothed']}")
                self.lines.append(f"  relative={snapshot['relative']} USED={snapshot['used']} terms={snapshot['terms']}")

    def update(self, feature, model, *, baseline=None):
        result, config = model.latest, model.config
        timestamp = feature['start_ms']
        features = feature['features']
        faces = features.get('faces', [])
        face = faces[0] if len(faces) == 1 else {}
        track = face.get('transient_track_key')
        track_changed = self.last_track is not None and track is not None and track != self.last_track
        if track is not None:
            self.last_track = track
        self.lines = [f"Gaussian actual input | t={timestamp}ms | calibration={features.get('head_calibration_status')}",
                      f'Sigma pitch/yaw: {config.gaussian_sigma_pitch_degrees:g}/{config.gaussian_sigma_yaw_degrees:g} deg (unchanged)']
        self.lines.append(f'Baseline P/Y: {baseline.get("pitch")}/{baseline.get("yaw")} deg' if baseline else 'Baseline P/Y: unavailable')
        self.lines.append(f'Track key: {track}; changed={track_changed}')
        for label, key in [('Raw', 'head_angles_degrees'),
                           ('Smoothed', 'head_angles_smoothed_degrees'),
                           ('Relative', 'head_angles_relative_degrees')]:
            angles = face.get(key)
            self.lines.append(f'{label} P/Y: ' + (
                f"{angles['pitch']:.2f}/{angles['yaw']:.2f} deg" if angles else 'unavailable'))
        primitive = result.get('gaussian_observation')
        if primitive is None:
            self.previous = None
            self.drops.clear()
            self.lines.append('Gaussian input/current/candidates: unavailable (not scored; no stale values)')
            self.previous_snapshot = None
            self._show_recent_event(timestamp)
            return
        pitch, yaw = primitive
        p_term = (pitch / config.gaussian_sigma_pitch_degrees) ** 2
        y_term = (yaw / config.gaussian_sigma_yaw_degrees) ** 2
        total = p_term + y_term
        snapshot = deepcopy(dict(track=track, baseline=baseline,
            current=result['current_direction_score'], used=primitive, terms=(p_term, y_term),
            raw=face.get('head_angles_degrees'), smoothed=face.get('head_angles_smoothed_degrees'),
            relative=face.get('head_angles_relative_degrees')))
        dominant = 'equal' if p_term == y_term else ('Pitch' if p_term > y_term else 'Yaw')
        self.lines.extend([
            f'USED P/Y: {pitch:.2f}/{yaw:.2f} deg (angle mode={config.gaussian_angle_mode})',
            f"Current: {result['current_direction_score']:.1f}/100 | 10s: {result.get('score')}",
            f'Exponent terms (P/sigmaP)^2={p_term:.3f}; (Y/sigmaY)^2={y_term:.3f}',
            f'Current attenuation: {dominant}; P/Y share=' +
            (f'{100*p_term/total:.1f}/{100*y_term/total:.1f}%' if total else '0/0% (neutral)'),
        ])
        candidates = [gaussian_angle_score(HeadAngles(pitch, yaw, 0),
                      replace(config, gaussian_sigma_yaw_degrees=sigma)) for sigma in (20, 30, 40)]
        self.lines.append('SAME INPUT sigmaYaw 20/30/40: ' + '/'.join(f'{s:.2f}' for s in candidates))
        previous = self.previous
        if (previous is not None and track == previous[1]
                and 0 < timestamp - previous[0] <= config.max_observation_hold_seconds * 1000):
            _, _, old_p, old_y, old_pt, old_yt, old_score = previous
            dp = (pitch - old_p + 180) % 360 - 180
            dy = (yaw - old_y + 180) % 360 - 180
            dpt, dyt = p_term - old_pt, y_term - old_yt
            drop = old_score - result['current_direction_score']
            driver = 'equal' if dpt == dyt else ('Pitch' if dpt > dyt else 'Yaw')
            self.lines.append(f'Delta P/Y={dp:+.2f}/{dy:+.2f} deg in {timestamp-previous[0]}ms; score={-drop:+.1f}')
            if drop > 0:
                self.drops.append((timestamp, drop, driver, dpt, dyt))
                self.events.append(dict(timestamp_ms=timestamp, drop=drop, driver=driver,
                                        before=self.previous_snapshot, after=snapshot))
        else:
            self.drops.clear()
            self.lines.append('Delta unavailable: first valid input, gap or track change')
        while self.drops and self.drops[0][0] < timestamp - 5000:
            self.drops.popleft()
        if self.drops:
            at, drop, driver, dpt, dyt = max(self.drops, key=lambda item: item[1])
            self.lines.append(f'Largest drop in last 5s: {drop:.1f} at {at}ms; driver={driver}; delta terms P/Y={dpt:+.3f}/{dyt:+.3f}')
        self.previous = (timestamp, track, pitch, yaw, p_term, y_term, result['current_direction_score'])
        self.previous_snapshot = snapshot
        self._show_recent_event(timestamp)


class UncertaintyDiagnostics:
    """행동의 불명확함과 현재/집계 점수 보류 원인을 분리하는 프레임 집계."""
    def __init__(self, config):
        # 20도가 실제로 더 넓고 이탈 경계 미만일 때만 비교한다.
        # 사용자 이탈 임계값을 바꾸거나 다른 후보를 몰래 선택하지 않는다.
        self.candidate = None
        self.candidate_disabled_reason = None
        if config.front_yaw_degrees >= 20:
            self.candidate_disabled_reason = 'candidate_not_wider_than_baseline'
        elif config.away_yaw_degrees <= 20:
            self.candidate_disabled_reason = 'candidate_reaches_away_boundary'
        else:
            self.candidate = GaussianRuntime(replace(config, front_yaw_degrees=20))
        self.counts = {name: Counter() for name in ('baseline', 'candidate_front_yaw_20')}

    def reset(self):
        if self.candidate is not None:
            self.candidate.reset()
        for counts in self.counts.values():
            counts.clear()

    def update(self, feature, baseline):
        results = [('baseline', baseline)]
        if self.candidate is not None:
            self.candidate(feature)
            results.append(('candidate_front_yaw_20', self.candidate.latest))
        for name, result in results:
            counts = self.counts[name]
            counts['frames'] += 1
            counts['state:' + result['orientation_state']] += 1
            reasons = result.get('reason_codes', [])
            for reason in set(reasons):
                counts['reason:' + reason] += 1
            current = result.get('current_direction_score')
            counts['current_null'] += current is None
            counts['aggregate_null'] += result.get('score') is None
            if current is not None:
                counts['current_sum'] += current
                counts['current_valid'] += 1
            if current is None:
                if 'pose_unreliable' in reasons:
                    cause = 'pose_unreliable'
                elif 'face_not_found' in reasons:
                    cause = 'face_not_found'
                elif 'multiple_faces_unsupported' in reasons:
                    cause = 'multiple_faces'
                elif 'baseline_unavailable' in reasons:
                    cause = 'calibration_unavailable'
                elif result.get('downward_status'):
                    cause = 'downward_uncertain'
                else:
                    cause = 'other_observation_failure'
                counts['null_cause:' + cause] += 1

    def summary(self):
        results = {}
        for name, counts in self.counts.items():
            frames = counts['frames']
            results[name] = dict(frames=frames,
                counts={key: value for key, value in counts.items() if key != 'current_sum'},
                ratios={key: value/frames for key, value in counts.items()
                        if key not in ('frames', 'current_sum', 'current_valid') and frames},
                mean_valid_current_score=counts['current_sum']/counts['current_valid'] if counts['current_valid'] else None)
            if name == 'candidate_front_yaw_20':
                results[name].update(enabled=self.candidate is not None,
                                     disabled_reason=self.candidate_disabled_reason)
        return results


class ScoreTrace:
    """점수와 보류 상태를 구분하는 제한된 메모리 기록. 파일에는 저장하지 않는다."""

    def __init__(self):
        self.records = deque(maxlen=6000)

    def update(self, feature, models, *, baselines=None):
        faces = feature.get('features', {}).get('faces', [])
        face = faces[0] if len(faces) == 1 else {}
        record = {'time_ms': feature['start_ms'], 'models': {},
                  'raw': deepcopy(face.get('head_angles_degrees')),
                  'smoothed': deepcopy(face.get('head_angles_smoothed_degrees')),
                  'baselines': deepcopy(baselines),
                  'quality': deepcopy(face.get('quality'))}
        for name in ('gaussian', 'gaussian_axiswise'):
            if name not in models:
                continue
            latest = models[name].latest
            record['models'][name] = deepcopy({key: latest.get(key) for key in (
                'current_direction_score', 'score', 'source_angles',
                'gaussian_observation', 'orientation_state', 'downward_status',
                'status', 'reason_codes')})
        self.records.append(record)
        return record

    def summary(self):
        result = {}
        for name in ('gaussian', 'gaussian_axiswise'):
            samples = [row['models'][name] for row in self.records if name in row['models']]
            if not samples:
                continue
            fields = {}
            for key in ('current_direction_score', 'score'):
                values = [row[key] for row in samples if row[key] is not None]
                fields[key] = {'min': min(values) if values else None,
                               'max': max(values) if values else None,
                               'null_ratio': 1 - len(values) / len(samples)}
            fields['current_null_history_available'] = sum(
                row['current_direction_score'] is None and row['score'] is not None
                for row in samples)
            fields['downward_states'] = {state: sum(row['downward_status'] == state for row in samples)
                                        for state in {row['downward_status'] for row in samples}}
            result[name] = {'retained_frames': len(samples), **fields}
        return result


    def disagreements(self):
        """정기 출력 사이의 분류 불일치도 종료 시 원본 수치와 함께 확인한다."""
        rows = [row for row in self.records if 'gaussian_axiswise' in row['models']
                and row['models']['gaussian']['orientation_state'] !=
                    row['models']['gaussian_axiswise']['orientation_state']]
        def roll(row):
            angles = row['models']['gaussian']['source_angles'] or {}
            return abs(angles.get('roll', 0))
        return {'count': len(rows), 'examples': sorted(rows, key=roll, reverse=True)[:3]}


def panel_lines(feature, models, *, primary='gaussian', debug=False, diagnostics=None):
    """일반 화면은 현재 점수만, 집계/입력/사유/모델 비교는 디버그에 표시한다."""
    result = models[primary].latest
    current = result.get('current_direction_score') if primary == 'gaussian' else result.get('score')
    label = 'Current direction' if primary == 'gaussian' else 'Model score'
    value = f'{current:.1f}/100' if current is not None else 'UNAVAILABLE'
    calibration = feature['features'].get('head_calibration_status', 'waiting')
    lines = [f'Model: {primary}', f'{label}: {value}',
             f"Behavior: {result.get('orientation_state', 'not_observable')}",
             f'Calibration: {calibration}',
             'Click video window, C/Space: calibrate | D: debug | Q/Esc: quit']
    if result.get('angle_estimator'):
        lines[0] += f" | {result['angle_estimator']}"
    if primary == 'gaussian':
        aggregate = result.get('score')
        angles = result.get('source_angles')
        lines[2:2] = [
            f'10s score: {aggregate:.1f}/100' if aggregate is not None else '10s score: UNAVAILABLE',
            f"Relative P/Y: {angles['pitch']:.1f}/{angles['yaw']:.1f} deg" if angles else 'Relative P/Y: UNAVAILABLE',
            f'Sigma P/Y: {models[primary].config.gaussian_sigma_pitch_degrees:g}/{models[primary].config.gaussian_sigma_yaw_degrees:g} deg']
        lines.append('Current = latest observation; 10s = history (not current).')
        if current is None:
            lines.append('Current: DEFERRED (not zero, not previous score)')
            if result.get('downward_status'):
                lines[1] = 'Current direction: UNAVAILABLE (downward activity uncertain)'
        if result.get('calibration_method'):
            lines.append(f"Calibration method: {result['calibration_method']} (raw rotation)")
        else:
            lines[0] += ' | calibration: Euler'
        if 'gaussian_axiswise' in models:
            old = models['gaussian_axiswise'].latest
            reference = old.get('current_direction_score')
            lines.append(f'Old Euler current: {reference:.1f}/100' if reference is not None else 'Old Euler current: UNAVAILABLE')
            lines.append(f"Old Euler state: {old.get('orientation_state')} | relative P/Y/R: {old.get('source_angles')}")
        if 'gaussian_face_transform' in models:
            other = models['gaussian_face_transform'].latest
            current_other = other.get('current_direction_score')
            lines.append(f"Face transform current: {current_other:.1f}/100" if current_other is not None else 'Face transform current: UNAVAILABLE')
            lines.append(f"Face transform: {other.get('orientation_state')} | P/Y: {other.get('source_angles')}")
        if 'gaussian_pnp' in models:
            other = models['gaussian_pnp'].latest
            value = other.get('current_direction_score')
            lines.append(f'PnP reference: {value:.1f}/100' if value is not None else 'PnP reference: UNAVAILABLE')
    if result.get('downward_status'):
        lines.append(f"Downward: {result['downward_status']} (activity uncertain)")
    calibration_diagnostics = feature['features'].get('head_calibration_diagnostics', {})
    reason = calibration_diagnostics.get('last_reason_code')
    if calibration == 'collecting':
        lines.append(f"Calibration progress: {100 * calibration_diagnostics.get('progress', 0):.0f}% | hold face forward")
        if reason:
            lines.append(f'Calibration input issue: {reason}')
    if calibration == 'ready' and not feature['features'].get('faces'):
        lines.append('No face detected; calibration retained. Waiting for return.')
    elif calibration == 'waiting':
        lines.append(f'Recalibration required: {reason or "baseline_unavailable"}. Press C.')
    elif calibration == 'ready' and reason:
        lines.append(f'Observation unavailable: {reason}. Calibration retained.')
    if debug:
        lines.extend(['', 'DEBUG: score means temporal aggregate; current direction is separate.'])
        if diagnostics is not None:
            lines.extend(diagnostics.lines)
        for name, model in models.items():
            lines.extend(model_lines(name, model.latest))
            if model.config.scoring_method == 'gaussian':
                sampling = model.sampling_diagnostics()
                lines.append(f'Sigma P/Y: {model.config.gaussian_sigma_pitch_degrees}/{model.config.gaussian_sigma_yaw_degrees} deg (experimental)')
                lines.append(f"Sample intervals: {sampling['min_interval_ms']}..{sampling['max_interval_ms']} ms; regular={sampling['regular']}")
            lines.append('')
    return lines


def render_tracking(frame, feature, landmarks, *, show_overlay=True):
    """현재 프레임의 검출만 표시하며 과거 얼굴 위치를 남기지 않는다."""
    faces = feature['features'].get('faces', [])
    current_landmarks = landmarks if faces else []
    preview = (draw_landmarks(frame, current_landmarks, [])
               if show_overlay else frame.copy())
    height, width = preview.shape[:2]
    if not faces:
        label, color = 'FACE NOT DETECTED - waiting for return', (0, 180, 255)
    elif len(faces) > 1:
        label, color = 'MULTIPLE FACES - scoring unavailable', (0, 180, 255)
    elif faces[0].get('head_angles_relative_degrees') is None:
        label, color = 'FACE DETECTED - waiting for valid calibrated pose', (0, 180, 255)
    else:
        label, color = 'FACE DETECTED - pose available', (40, 230, 100)
    if show_overlay:
        for points in current_landmarks:
            valid = [(p[0], p[1]) for p in points
                     if len(p) >= 2 and math.isfinite(p[0]) and math.isfinite(p[1])
                     and 0 <= p[0] < 1 and 0 <= p[1] < 1]
            if valid:
                xs, ys = zip(*valid)
                cv2.rectangle(preview, (int(min(xs) * width), int(min(ys) * height)),
                              (int(max(xs) * width), int(max(ys) * height)), color, 2)
    cv2.putText(preview, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                .5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(preview, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                .5, color, 1, cv2.LINE_AA)
    return preview


def _score_label(value):
    return f'{value:.1f}' if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else '--'


def _debug_pages(feature, models, primary, diagnostics, face):
    """내용 길이가 바뀌어도 캔버스 크기는 유지하고 상세 정보만 쪽을 넘긴다."""
    lines = panel_lines(feature, models, primary=primary, debug=True, diagnostics=diagnostics)
    gaze = vars(face).get('gaze')
    if gaze is not None:
        lines.extend(gaze.lines())
        policy = getattr(models[primary], 'downward_policy', None)
        if policy is not None:
            lines.append(policy.diagnostic)
    wrapped = [part for line in lines for part in (textwrap.wrap(line, 44, break_long_words=True) or [''])]
    return [wrapped[i:i+36] for i in range(0, len(wrapped), 36)] or [[]]


def render_panel(frame, feature, models, *, overlay_smoother, face, pose, show_overlay,
                 primary='gaussian', debug=False, diagnostics=None, guided=None, debug_page=0,
                 live=False, calibration_range=None):
    """주요 정보의 위치는 고정하고 상세 정보는 페이지별로 표시한다."""
    preview = (render_video_frame(frame, feature, face.last_landmarks, pose.last_landmarks,
                                  overlay_smoother, show_overlay=show_overlay)
               if debug else render_tracking(frame, feature, face.last_landmarks,
                                              show_overlay=show_overlay))
    height, width = preview.shape[:2]
    scale = min(640 / width, 480 / height)
    preview = cv2.resize(preview, (max(1, round(width * scale)), max(1, round(height * scale))))
    canvas = np.zeros((860, 1400, 3), dtype=np.uint8)
    canvas[:preview.shape[0], :preview.shape[1]] = preview
    result = models[primary].latest
    current = result.get('current_direction_score') if primary == 'gaussian' else result.get('score')
    aggregate = result.get('score') if primary == 'gaussian' else None
    angles = result.get('source_angles') or {}
    calibration = feature['features'].get('head_calibration_status', 'waiting')
    cal_info = feature['features'].get('head_calibration_diagnostics') or {}
    gaze = vars(face).get('gaze')
    gaze_data = gaze.latest if gaze is not None else {}
    def label(text, x, y, scale=.55, color=(190, 196, 205), thickness=1):
        cv2.putText(canvas, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale * 1.2, color, max(2, thickness), cv2.LINE_AA)
    # 고정 좌표: 모델 점수와 카메라 기준 시선은 서로 다른 의미의 실험값이다.
    group = bool(getattr(models[primary], 'display_group', False))
    label(f'MODEL: {primary.upper()}  |  ' + ('GROUP | ' if group else '') + 'EXPERIMENTAL', 660, 30)
    label(('GROUP CURRENT' if group else 'CURRENT DIRECTION') if primary == 'gaussian' else 'MODEL SCORE', 660, 65)
    label('10s AGGREGATE' if primary == 'gaussian' else 'AGGREGATE: N/A', 1030, 65)
    label(_score_label(current), 660, 125, 1.65, (245, 245, 245), 3)
    label(_score_label(aggregate), 1030, 125, 1.65, (245, 245, 245), 3)
    if result.get('angle_estimator'):
        label('POSE: ' + result['angle_estimator'], 1030, 345, .40)
    if 'gaussian_pnp' in models:
        label('PnP ref: ' + _score_label(models['gaussian_pnp'].latest.get('current_direction_score')), 10, 545, .48)
    elif 'gaussian_face_transform' in models:
        label('MP ref: ' + _score_label(models['gaussian_face_transform'].latest.get('current_direction_score')), 10, 545, .48)
    label('/ 100', 810, 125, .6)
    label('/ 100', 1180, 125, .6)
    label('RELATIVE HEAD PITCH / YAW', 660, 165)
    label(f"{_score_label(angles.get('pitch'))} / {_score_label(angles.get('yaw'))} deg", 660, 200, .85, (245, 245, 245), 2)
    label('GAZE CANDIDATE (NOT SCREEN GAZE)', 660, 245)
    label(str(gaze_data.get('state', 'OFF')).upper()[:35], 660, 280, .8, (170, 220, 255), 2)
    label(f"Gaze Y/P: {_score_label(gaze_data.get('yaw'))} / {_score_label(gaze_data.get('pitch'))} deg", 660, 310, .55)
    label(f"Calibration: {calibration}  {100 * cal_info.get('progress', 0):.0f}%", 660, 345, .6)
    if calibration_range:
        yaw_range = calibration_range.get('axes', {}).get('yaw')
        range_text = (f"Yaw range P10-P90: {yaw_range['low']:+.1f} to {yaw_range['high']:+.1f} deg"
                      if yaw_range else 'Yaw range: insufficient samples')
        label(f"{range_text} ({calibration_range['status']}; diagnostic only)", 10, 540, .48)
    label(f"State: {result.get('orientation_state', 'not_observable')} | Down: {result.get('downward_status') or '--'}"[:80], 660, 375, .52)
    reason = cal_info.get('last_reason_code') or gaze_data.get('reason') or ', '.join(result.get('reason_codes') or [])
    label(('Reason: ' + str(reason or '--'))[:82], 660, 403, .48)
    if debug:
        pages = _debug_pages(feature, models, primary, diagnostics, face)
        page = debug_page % len(pages)
        label(f'DEBUG  {page + 1}/{len(pages)}  |  [ / ] change page', 660, 448, .55, (120, 210, 255), 1)
        for index, line in enumerate(pages[page]):
            column, row = divmod(index, 18)
            label(line, 660 + column*365, 471 + row*21, .40, (220, 224, 230))
    else:
        label('D: debug  |  V: overlay  |  Q / Esc: quit', 660, 448, .55)
    if live:
        label('C: recalibrate  V: overlay  [ / ]: page  Q: hide', 10, 510, .48)
    else:
        left, top, right, bottom = CALIBRATE_BUTTON
        cv2.rectangle(canvas, (left, top), (right, bottom), (80, 70, 20), -1)
        cv2.putText(canvas, 'CLICK TO CALIBRATE', (left + 12, top + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
    if guided is not None:
        # 고정 크기 안내를 영상 영역에 그려 상세 디버그 행 수와 관계없이 읽을 수 있게 한다.
        cv2.rectangle(canvas, (0, 345), (640, 425), (20, 20, 20), -1)
        for index, line in enumerate(guided.screen_lines(feature['start_ms'])):
            cv2.putText(canvas, line, (10, 366+index*23), cv2.FONT_HERSHEY_SIMPLEX,
                        .47, (60, 240, 255), 1, cv2.LINE_AA)
    return canvas


def pose_display_view(feature, models, calibration, comparison, pose_source):
    """표시·진단의 주 경로만 선택한다. 비교 기준 PnP 런타임은 그대로 둔다."""
    if pose_source == 'face-transform':
        view = {key: value for key, value in models.items() if key != 'gaussian_face_transform'}
        view['gaussian_pnp'] = models['gaussian']
        view['gaussian'] = comparison.runtime
        return comparison.input_feature, view, comparison.calibration
    return feature, models, calibration


def run(camera_index=0, duration=0.0, compare=True, *, primary='gaussian', gaussian_config=None,
        debug=False, calibrate=False, yaw_diagnostic=False, diagnostic_move_seconds=2,
        diagnostic_hold_seconds=3, gaze=False, yaw_log=False,
        calibration_method='euler', compare_calibration=False, validated_pose=False,
        diagnostic_down=False, compare_face_transform=False, front_stability=False, pose_source='pnp'):
    """카메라·분석기·독립 세션을 소유하며 오류/종료 시 자원을 반환한다."""
    if not math.isfinite(duration) or duration < 0:
        raise ValueError('duration은 유한한 0 이상의 초여야 합니다.')
    if primary not in {'legacy', 'huang', 'gaussian'}:
        raise ValueError('primary는 legacy, huang 또는 gaussian이어야 합니다')
    if pose_source not in ('pnp', 'face-transform'):
        raise ValueError('pose_source는 pnp 또는 face-transform이어야 합니다.')
    if calibration_method not in ('matrix', 'euler'):
        raise ValueError('calibration_method는 matrix 또는 euler여야 합니다.')
    if compare_calibration and (calibration_method != 'matrix' or primary != 'gaussian'):
        raise ValueError('보정 비교는 --model gaussian --calibration-method matrix로 실행하세요.')
    compare_face_transform = compare_face_transform or pose_source == 'face-transform'
    if yaw_diagnostic and (primary != 'gaussian' or calibrate):
        raise ValueError('좌우 진단은 Gaussian 모델에서 수동 보정으로 실행하세요 (--calibrate 제외)')
    if diagnostic_down and not yaw_diagnostic:
        raise ValueError('--diagnostic-down은 --yaw-diagnostic과 함께 사용하세요.')
    if front_stability and not (yaw_diagnostic and compare_face_transform):
        raise ValueError('--front-stability는 --yaw-diagnostic --compare-face-transform과 함께 사용하세요.')
    if compare_face_transform and (primary != 'gaussian' or calibration_method != 'euler'):
        raise ValueError('얼굴 변환행렬 비교는 --model gaussian --calibration-method euler로 실행하세요.')
    if gaze and (compare_face_transform or validated_pose or calibration_method != 'euler'):
        raise ValueError('gaze cannot be combined with experimental pose sources; run each path separately')
    if gaze and primary != 'gaussian':
        raise ValueError('--gaze는 Gaussian 독립 데모에서 사용하세요 (--model gaussian).')
    if yaw_diagnostic and (primary != 'gaussian' or calibrate):
        raise ValueError('좌우 진단은 Gaussian 모델에서 수동 보정으로 실행하세요 (--calibrate 제외)')
    if yaw_log and not yaw_diagnostic:
        raise ValueError('--yaw-log는 --yaw-diagnostic과 함께 사용하세요')
    guided = GuidedYawDiagnostic(diagnostic_move_seconds, diagnostic_hold_seconds,
                                 report_sink=save_yaw_diagnostic if yaw_log else None,
                                 include_down=diagnostic_down, front_stability=front_stability) if yaw_diagnostic else None
    if compare:
        runtime = ComparisonRuntime(primary, include_gaussian=True, gaussian_config=gaussian_config)
    elif primary == 'gaussian':
        runtime = GaussianRuntime(gaussian_config)
    else:
        runtime = EngagementRuntime(ModelConfig.huang() if primary == 'huang' else ModelConfig())
    models = runtime.runtimes if compare else {primary: runtime}
    rotation = RotationCalibration() if calibration_method == 'matrix' and primary == 'gaussian' else None
    reference = models.get('gaussian') if compare_calibration and rotation is not None else None
    if rotation is not None:
        upgraded = RotationGaussianRuntime(rotation, models['gaussian'].config)
        if compare:
            runtime.runtimes['gaussian'] = upgraded
        else:
            runtime = upgraded
        models = {**models, 'gaussian': upgraded}
        if reference is not None:
            models['gaussian_axiswise'] = reference

    def consume(feature):
        runtime(feature)
        if reference is not None:
            reference(feature)

    diagnostics = GaussianDiagnostics() if 'gaussian' in models else None
    uncertainty = UncertaintyDiagnostics(models['gaussian'].config) if 'gaussian' in models and not gaze else None
    score_trace = ScoreTrace() if reference is not None else None
    pose_records = deque(maxlen=6000)
    transform_comparison = None
    gaze_estimator = None
    if gaze:
        from video_recognition.gaze_estimation import GazeEstimator, GazeDownPolicy
        gaze_estimator = GazeEstimator()
        if 'gaussian' in models:
            gaussian = models['gaussian']
            gaussian.downward_policy = GazeDownPolicy(gaze_estimator)
            gaussian.model_version += '+mobilegaze-s0-0.1'
            gaussian.config = replace(gaussian.config,
                config_version=gaussian.config.config_version+'+mobilegaze-s0-0.1')
    frames = 0
    try:
        with ExitStack() as resources:
            if gaze_estimator is not None:
                resources.callback(gaze_estimator.close)
            camera = cv2.VideoCapture(camera_index)
            resources.callback(camera.release)
            if not camera.isOpened():
                raise RuntimeError(f'웹캠 {camera_index}을 열 수 없습니다. 카메라 권한과 다른 앱의 사용 여부를 확인하세요.')
            print(f'Webcam {camera_index} opened; initializing MediaPipe.', flush=True)
            service = IntegrationService()
            sid = service.start_session('Huang webcam test', config={
                'config_version': runtime.config.config_version,
                'participation_model': {'model_id': runtime.model_id, 'model_version': runtime.model_version},
                'stt_model': None,
            })
            resources.callback(service.stop_session, sid)
            face = FaceAnalyzer(show_landmarks=True)
            if compare_face_transform:
                face = FaceTransformAnalyzer(face, validated_pnp=validated_pose)
            elif validated_pose:
                face = ValidatedFaceAnalyzer(face)
            resources.callback(face.close)
            if gaze_estimator is not None:
                from video_recognition.gaze_estimation import GazeFaceAnalyzer
                face = GazeFaceAnalyzer(face, gaze_estimator)
            pose = PoseAnalyzer(show_landmarks=True)
            resources.callback(pose.close)
            capture = PreviewCapture(WebcamCapture(camera))
            pipeline = VisionPipeline(service, sid, capture, face,
                                      model_consumer=consume if reference is not None else runtime,
                                      pose_analyzer=pose)
            pipeline.head_calibration = DualWebcamCalibration(rotation) if rotation is not None else WebcamHeadCalibration()
            active_calibration = rotation if rotation is not None and primary == 'gaussian' else pipeline.head_calibration
            if compare_face_transform:
                transform_comparison = FaceTransformComparison(WebcamHeadCalibration(), models['gaussian'].config)
                models = {**models, 'gaussian_face_transform': transform_comparison.runtime}
                print(f'FACE_TRANSFORM_COMPARE enabled: shared detection; display primary={pose_source}; coordinate A=diag(1,-1,-1).', flush=True)
                if pose_source == 'face-transform':
                    active_calibration = transform_comparison.calibration
            # 분석기/카메라 종료는 ExitStack이 담당한다. pipeline은 메모리 후처리만 소유한다.
            resources.callback(pipeline.face_postprocessor.reset)
            smoother = LandmarkOverlaySmoother()
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            resources.callback(close_window)
            cv2.resizeWindow(WINDOW, 1200, 650)
            controls = CalibrationControls(calibrate)
            cv2.setMouseCallback(WINDOW, controls.mouse)
            started = time.monotonic()
            next_report = 0.0
            show_overlay = True
            pose_input_diagnostics = PoseInputDiagnostics() if yaw_log else None
            debug_page = 0
            print('Click video window; C/Space: calibrate, D: debug, V: overlay, Q/Esc: quit. '
                  + ('Yaw summary logging enabled.' if yaw_log else 'No files are saved.'), flush=True)
            if guided is not None:
                print('수동 좌우 진단: 정면을 보고 준비한 뒤 CALIBRATE 버튼 또는 C/Space를 한 번 누르세요.\n'
                      f'화면 안내의 LEFT/RIGHT는 본인의 좌우입니다. 보정 후 {len(guided.STAGES)}구간이 자동 진행됩니다.\n'
                      '카메라를 고정하고 좌우를 비슷한 크기로 돌려주세요. 숫자만 메모리에 보관합니다.', flush=True)
            while not duration or time.monotonic() - started < duration:
                if controls.consume(-1) and active_calibration.status != 'collecting':
                    if guided is not None:
                        guided.calibration_started()
                    pipeline.head_calibration.start()
                    if transform_comparison is not None:
                        transform_comparison.start()
                    for model in models.values():
                        model.reset()
                    if gaze_estimator is not None:
                        gaze_estimator.reset()
                        if 'gaussian' in models:
                            models['gaussian'].downward_policy.reset()
                    if diagnostics is not None:
                        diagnostics.reset()
                    if uncertainty is not None:
                        uncertainty.reset()
                    if score_trace is not None:
                        score_trace.records.clear()
                    print('Calibration started: hold face forward for 3 seconds.', flush=True)
                if guided is not None:
                    guided.tick(service.elapsed_ms(sid))
                feature = pipeline.process_one()
                if feature is not None:
                    frames += 1
                    trace_record = (score_trace.update(feature, models, baselines={
                        'matrix': rotation.baseline, 'euler': pipeline.head_calibration.baseline})
                        if score_trace is not None else None)
                    if rotation is not None and primary == 'gaussian':
                        feature = models['gaussian'].input_feature
                    transform_row = None
                    if transform_comparison is not None:
                        phase = guided.phase(feature['start_ms']) if guided is not None and guided.state == 'running' else None
                        transform_row = transform_comparison.update(feature, face, models['gaussian'].latest,
                            stage=guided.STAGES[phase[0]] if phase else 'manual',
                            phase=phase[1] if phase else 'hold', pnp_baseline=pipeline.head_calibration.baseline)
                        transform_row['display_primary'] = pose_source
                    display_feature, display_models, display_calibration = pose_display_view(
                        feature, models, active_calibration, transform_comparison, pose_source)
                    if uncertainty is not None:
                        uncertainty.update(display_feature, display_models['gaussian'].latest)
                    if guided is not None:
                        pose_quality = (pose_input_diagnostics.observe(face.last_landmarks, capture.last_frame) if pose_input_diagnostics is not None else None)
                        guided.update(display_feature, display_models['gaussian'], display_calibration, pose_quality)
                    if diagnostics is not None:
                        diagnostics.update(display_feature, display_models['gaussian'], baseline=display_calibration.baseline)
                        pose_info = getattr(face, 'diagnostics', None) if validated_pose else None
                        if pose_info:
                            diagnostics.lines.extend([
                                f"Pose validation: {pose_info['status']} / {pose_info['reason']}",
                                f"Old raw P/Y/R: {pose_info['original']}",
                                f"Validated raw P/Y/R: {pose_info['validated']}",
                                f"Old/new depth: {pose_info.get('original_min_depth')}/{pose_info.get('min_depth')}",
                                f"RMSE px: {pose_info.get('reprojection_rmse_pixels')} | jump: {pose_info.get('jump_degrees')}"])
                    if validated_pose:
                        pose_records.append(deepcopy(dict(time_ms=feature['start_ms'],
                            pose=getattr(face, 'diagnostics', None), baseline=pipeline.head_calibration.baseline,
                            relative=models[primary].latest.get('source_angles'),
                            state=models[primary].latest.get('orientation_state'),
                            score=models[primary].latest.get('current_direction_score', models[primary].latest.get('score')),
                            away_seconds=models[primary].latest.get('away_seconds'),
                            reasons=models[primary].latest.get('reason_codes'))))
                    panel = render_panel(capture.last_frame, display_feature, display_models,
                                         overlay_smoother=smoother, face=face, pose=pose,
                                         show_overlay=show_overlay, primary=primary, debug=debug,
                                         diagnostics=diagnostics, guided=guided, debug_page=debug_page,
                                         calibration_range=pipeline.head_calibration.observed_range)
                    cv2.imshow(WINDOW, panel)
                    if (debug or score_trace is not None or validated_pose or compare_face_transform) and time.monotonic() >= next_report:
                        if transform_row is not None:
                            print('FACE_TRANSFORM_COMPARE ' + json.dumps(transform_row), flush=True)
                        if validated_pose:
                            print('POSE_VALIDATION ' + json.dumps(getattr(face, 'diagnostics', None)), flush=True)
                        print(json.dumps(trace_record if trace_record is not None else {name: {'score': model.latest['score'],
                                                'current_direction_score': model.latest.get('current_direction_score'),
                                                'relative': model.latest.get('source_angles'),
                                                'away_seconds': model.latest.get('away_seconds'),
                                                'state': model.latest['orientation_state'],
                                                'reasons': model.latest['reason_codes']}
                                          for name, model in models.items()}), flush=True)
                        next_report = time.monotonic() + 2
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if controls.consume(key):
                    controls.requested = True
                elif key in (ord('v'), ord('V')):
                    show_overlay = not show_overlay
                elif key in (ord('d'), ord('D')):
                    debug = not debug
                elif key == ord('[') and debug:
                    debug_page -= 1
                elif key == ord(']') and debug:
                    debug_page += 1
    finally:
        if guided is not None and guided.state in {'collecting', 'running'}:
            guided.finish('종료 또는 오류로 진단 미완료')
        print(f'Analyzed frames: {frames}', flush=True)
        if transform_comparison is not None:
            print('FACE_TRANSFORM_SUMMARY (instruction stages, not measured rotation truth): ' +
                  json.dumps(transform_comparison.summary()), flush=True)
            print('FACE_TRANSFORM_RECOVERY ' + json.dumps(transform_comparison.recovery_summary()), flush=True)
        if uncertainty is not None:
            print('Uncertainty comparison (frame ratios; overlapping reasons): ' +
                  json.dumps(uncertainty.summary()), flush=True)
        if validated_pose:
            negative = [r for r in pose_records if r['pose'] and r['pose'].get('original_min_depth', 0) < 0]
            invalid = [r for r in pose_records if r['pose'] and r['pose']['status'] != 'valid']
            examples = []
            for state in ('front', 'away', 'ambiguous', 'looking_down'):
                row = next((r for r in pose_records if r['state'] == state), None)
                if row:
                    examples.append(row)
            if negative:
                examples.append(min(negative, key=lambda r: r['pose']['original_min_depth']))
            print('Validated pose summary: ' + json.dumps(dict(retained_frames=len(pose_records),
                original_negative_depth_frames=len(negative), rejected_pose_frames=len(invalid),
                examples=examples)), flush=True)
        if score_trace is not None:
            print('Score trace summary (retained frames, not accuracy): ' +
                  json.dumps(score_trace.summary()), flush=True)
            print('Calibration disagreements (same-frame numeric examples): ' +
                  json.dumps(score_trace.disagreements()), flush=True)
        if compare:
            print('Aggregate frame-based comparison (not accuracy): ' + json.dumps(runtime.summary()), flush=True)


def close_window():
    """사용자가 이미 닫았더라도 이 실행기의 창만 안전하게 정리한다."""
    try:
        cv2.destroyWindow(WINDOW)
    except cv2.error:
        pass


def main():
    parser = argparse.ArgumentParser(description='참여도 모델 독립 웹캠 비교 (legacy/Huang/Gaussian, 저장 없음)')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--duration', type=float, default=0, help='자동 종료 초, 기본 0은 수동 종료')
    parser.add_argument('--huang-only', action='store_true', help='기존 모델 비교 없이 Huang만 표시')
    parser.add_argument('--model', choices=('legacy', 'huang', 'gaussian'), default=None)
    parser.add_argument('--debug', action='store_true', help='집계 점수·상대각·사유·모델 비교 표시')
    parser.add_argument('--gaze', action='store_true', help='Gaussian 전용: 지점 보정 없는 MobileGaze 시선 후보 및 하향 보류 실험')
    parser.add_argument('--calibrate', action='store_true', help='키 입력 없이 시작할 때 정면 보정')
    parser.add_argument('--yaw-diagnostic', action='store_true',
                        help='수동 보정 후 정면/왼쪽/정면/오른쪽/정면 자동 수치 진단')
    parser.add_argument('--yaw-log', action='store_true', help='좌우 진단 요약과 초별 각도 통계를 로컬 파일에 저장')
    parser.add_argument('--diagnostic-move-seconds', type=float, default=2, help='각 구간 자세 이동·정착 시간 (기본 2초)')
    parser.add_argument('--diagnostic-hold-seconds', type=float, default=3, help='각 구간 유지 표본 시간 (기본 3초)')
    parser.add_argument('--single-model', action='store_true', help='선택한 모델만 표시')
    parser.add_argument('--sigma-pitch', type=float, default=15, help='Gaussian sigma pitch (도, 실험 초기값)')
    parser.add_argument('--sigma-yaw', type=float, default=20, help='Gaussian sigma yaw (도, 실험 초기값)')
    parser.add_argument('--gaussian-angle-mode', choices=('relative', 'softened'), default='softened',
                        help='relative: 이전 Gaussian 입력, softened: 중립 부근 완화 (기본)')
    parser.add_argument('--validated-pose', action='store_true', help='독립 웹캠에서 양의 깊이 SQPnP와 자세 품질 검사 적용')
    parser.add_argument('--compare-face-transform', action='store_true', help='동일 검출의 MediaPipe 얼굴 변환행렬과 PnP 비교 (기본 경로 유지)')
    parser.add_argument('--pose-source', choices=('pnp', 'face-transform'), default='pnp',
                        help='독립 웹캠 주 표시·진단 각도 선택. face-transform도 PnP 비교를 유지')
    parser.add_argument('--front-stability', action='store_true', help='정면 고정·작은 몸 이동·정면 복귀 안내 추가')
    parser.add_argument('--diagnostic-down', action='store_true', help='좌우 자동 진단 끝에 아래 보기·정면 복귀 구간 추가')
    parser.add_argument('--calibration-method', choices=('matrix', 'euler'), default='euler',
                        help='Gaussian 웹캠 보정: 기존 Euler 차감(기본) 또는 실험용 matrix')
    parser.add_argument('--compare-calibration', action='store_true', help='같은 영상의 이전 Euler Gaussian 현재 점수도 표시')
    args = parser.parse_args()
    args.model = args.model or ('huang' if args.huang_only else 'gaussian')
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error('--duration은 유한한 0 이상의 값이어야 합니다.')
    if args.huang_only and args.model != 'huang':
        parser.error('--huang-only는 --model huang과 함께 사용하세요.')
    if args.yaw_diagnostic and (args.model != 'gaussian' or args.calibrate):
        parser.error('--yaw-diagnostic은 Gaussian에서 수동 보정합니다. --calibrate를 제외하세요.')
    if args.yaw_log and not args.yaw_diagnostic:
        parser.error('--yaw-log는 --yaw-diagnostic과 함께 사용하세요.')
    try:
        config = ModelConfig.gaussian(gaussian_sigma_pitch_degrees=args.sigma_pitch,
                                      gaussian_sigma_yaw_degrees=args.sigma_yaw,
                                      gaussian_angle_mode=args.gaussian_angle_mode)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        run(args.camera, args.duration, not (args.huang_only or args.single_model),
            primary=args.model, gaussian_config=config, debug=args.debug, calibrate=args.calibrate,
            yaw_diagnostic=args.yaw_diagnostic, diagnostic_move_seconds=args.diagnostic_move_seconds,
            diagnostic_hold_seconds=args.diagnostic_hold_seconds, gaze=args.gaze, yaw_log=args.yaw_log,
            calibration_method=args.calibration_method, compare_calibration=args.compare_calibration,
            validated_pose=args.validated_pose, diagnostic_down=args.diagnostic_down,
            compare_face_transform=args.compare_face_transform, front_stability=args.front_stability,
            pose_source=args.pose_source)
    except KeyboardInterrupt:
        print('웹캠 테스트를 종료합니다.')
    except Exception as exc:
        parser.exit(1, f'웹캠 실행 실패: {type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    main()
