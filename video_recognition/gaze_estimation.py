"""MobileGaze ONNX를 이용한 카메라 기준 시선 후보. 결과는 메모리에만 둔다."""
from pathlib import Path
import math
import time

import cv2
import numpy as np


DEFAULT_MODEL = Path(__file__).resolve().parent / 'models' / 'mobileone_s0_gaze.onnx'


class GazeEstimator:
    """기존 얼굴 경계를 재사용한다. 개인 보정이나 별도 얼굴 검출기는 없다."""

    def __init__(self, model_path=DEFAULT_MODEL, *, session=None, interval_ms=333):
        if session is None:
            if not Path(model_path).is_file():
                raise FileNotFoundError('시선 모델이 없습니다. python -m video_recognition.download_gaze_model 을 실행하세요.')
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(model_path), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        self.session = session
        self.interval_ms = interval_ms
        input_spec = session.get_inputs()[0]
        self.input_name = input_spec.name
        self.input_size = tuple(input_spec.shape[2:][::-1])
        self.output_names = [output.name for output in session.get_outputs()]
        if len(self.output_names) != 2 or any(not isinstance(n, int) or n <= 0 for n in self.input_size):
            raise ValueError('MobileGaze의 고정 크기 입력/두 출력 ONNX 모델이 필요합니다.')
        self.last_box = None
        self.latest = self.unknown('waiting', None)

    @staticmethod
    def unknown(reason, timestamp):
        return dict(state='unknown', reason=reason, yaw=None, pitch=None,
                    sample_ms=timestamp, inference_ms=None)

    def reset(self):
        self.last_box = None
        self.latest = self.unknown('waiting', None)

    def close(self):
        self.reset()
        self.session = None

    @staticmethod
    def decode(logits):
        values = np.asarray(logits, dtype=np.float32)
        if values.shape != (1, 90) or not np.isfinite(values).all():
            raise ValueError('invalid_logits')
        probabilities = np.exp(values - np.max(values, axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        if probabilities.max() < .02:
            raise ValueError('uncertain_logits')
        return float(np.sum(probabilities[0] * np.arange(90)) * 4 - 180)

    def update(self, image, boxes, faces, timestamp):
        reason = None
        if len(faces) != 1 or len(boxes) != 1:
            reason = 'single_face_required'
        elif faces[0].get('quality', {}).get('status') not in {'valid', 'partial'}:
            reason = 'face_unavailable'
        else:
            openness = faces[0].get('eye_openness')
            if (not isinstance(openness, (int, float)) or not math.isfinite(openness)
                    or openness < .12):
                reason = 'eyes_unavailable_or_closed'
                if isinstance(openness, (int, float)) and math.isfinite(openness):
                    reason += f' (EAR={openness:.3f}, min=0.120)'
        if reason:
            self.reset()
            self.latest = self.unknown(reason, timestamp)
            return self.latest
        box = boxes[0]
        if len(box) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in box):
            self.reset()
            self.latest = self.unknown('invalid_face_box', timestamp)
            return self.latest
        height, width = image.shape[:2]
        left, top, right, bottom = [round(v * scale) for v, scale in zip(box, (width, height, width, height))]
        if right - left < 80 or bottom - top < 80:
            self.reset()
            self.latest = self.unknown('face_too_small', timestamp)
            return self.latest
        previous_ms = self.latest['sample_ms']
        changed = self.last_box is not None and max(abs(a-b) for a, b in zip(box, self.last_box)) > .15
        if (not changed and previous_ms is not None and self.last_box is not None
                and 0 < timestamp - previous_ms < self.interval_ms):
            return self.latest
        self.last_box = list(box)
        started = time.perf_counter()
        try:
            # 공식 MobileGaze ONNX 전처리: BGR→RGB, 입력 크기, ImageNet 정규화.
            crop = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB)
            tensor = cv2.resize(crop, self.input_size).astype(np.float32) / 255.
            tensor = (tensor - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
            tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])
            outputs = self.session.run(self.output_names, {self.input_name: tensor})
            # upstream outputs[0]=yaw, outputs[1]=pitch; 각도는 도 단위로 유지한다.
            yaw, pitch = self.decode(outputs[0]), self.decode(outputs[1])
            state = ('forward_candidate' if abs(yaw) <= 20 and abs(pitch) <= 15 else
                     'away_candidate' if abs(yaw) >= 35 or abs(pitch) >= 30 else 'unknown')
            self.latest = dict(state=state, reason='angle_boundary' if state == 'unknown' else None,
                               yaw=yaw, pitch=pitch, sample_ms=timestamp,
                               inference_ms=round((time.perf_counter() - started) * 1000, 1))
        except (ValueError, RuntimeError, cv2.error) as error:
            self.latest = self.unknown('inference_failed:' + type(error).__name__, timestamp)
        return self.latest

    def lines(self):
        value = self.latest
        lines = ['Gaze: ' + value['state'] + ' (camera direction candidate)',
                 'Not a verified screen on/off detector; no point calibration']
        if value['yaw'] is not None:
            lines.append(f"Gaze Y/P: {value['yaw']:.1f}/{value['pitch']:.1f} deg | inference {value['inference_ms']:.1f} ms")
        if value['reason']:
            lines.append('Gaze reason: ' + value['reason'])
        return lines


class GazeFaceAnalyzer:
    """동일 분석 프레임에서만 얼굴과 시선을 연결하며 반환 특징은 변경하지 않는다."""

    def __init__(self, analyzer, gaze):
        self.analyzer = analyzer
        self.gaze = gaze

    def __getattr__(self, name):
        return getattr(self.analyzer, name)

    def analyze(self, frame):
        faces = self.analyzer.analyze(frame)
        self.gaze.update(frame.image, self.analyzer.last_face_boxes, faces, frame.capture_ms)
        return faces


class GazeDownPolicy:
    """머리 숙임과 모델의 아래 방향이 함께 지속될 때만 Gaussian을 보류한다."""

    def __init__(self, estimator):
        self.estimator = estimator
        self.reset()

    def reset(self):
        self.since = None
        self.last_ms = None
        self.track = None
        self.diagnostic = 'Down gate: waiting for observation'

    def __call__(self, feature, result):
        timestamp = feature['start_ms']
        faces = feature['features'].get('faces', [])
        track = faces[0].get('transient_track_key') if len(faces) == 1 else None
        if self.track is not None and track != self.track:
            # 얼굴 후처리의 임시 키는 분석기 호출 뒤 부여된다. 바뀐 키에는 캐시를 넘기지 않는다.
            self.estimator.reset()
        if (self.last_ms is not None and (timestamp <= self.last_ms or timestamp-self.last_ms > 600)
                or track != self.track):
            self.since = None
        self.track, self.last_ms = track, timestamp
        value = self.estimator.latest
        angles = result.get('source_angles') or result.get('angles')
        fresh = value.get('sample_ms') is not None and 0 <= timestamp-value['sample_ms'] <= 500
        # 진단 화면 전용. 시선 결과를 저장하거나 미충족 조건을 자동 완화하지 않는다.
        head_hint = (f"head P {angles['pitch']:.1f} >= 15: {'yes' if angles['pitch'] >= 15 else 'no'}"
                     if angles is not None else 'head P unavailable (calibration/quality)')
        gaze_hint = (f"gaze P {value['pitch']:.1f} <= -30: {'yes' if value['pitch'] <= -30 else 'no'}; "
                     f"abs(Y) {abs(value['yaw']):.1f} < 35: {'yes' if abs(value['yaw']) < 35 else 'no'}"
                     if value.get('pitch') is not None and value.get('yaw') is not None else 'gaze unavailable')
        self.diagnostic = f'Down gate: {head_hint}; {gaze_hint}; fresh: {"yes" if fresh else "no"}'
        down = (len(faces) == 1 and angles is not None and angles['pitch'] >= 15
                and fresh and value['state'] != 'unknown' and value['pitch'] <= -30
                and abs(value['yaw']) < 35)
        if not down:
            self.since = None
            return False, 'gaze_unavailable' if not fresh or value['state'] == 'unknown' else 'no_downward_evidence'
        if self.since is None:
            self.since = timestamp
        confirmed = value['sample_ms'] - self.since >= 400
        self.diagnostic += f"; confirmed sample span {max(0, value['sample_ms']-self.since)} / 400 ms"
        return confirmed, 'gaze_down_confirmed' if confirmed else 'gaze_down_pending'
