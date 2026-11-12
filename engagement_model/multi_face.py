"""세션 메모리 전용 다인 보정·점수와 익명 Gaussian 집계."""

from collections import Counter
from copy import deepcopy

from video_recognition.head_calibration import HeadCalibration
from .config import ModelConfig
from .gaussian import GaussianRuntime, gaussian_angle_score
from .input_adapter import HeadAngles, parse_feature_window


class MultiFaceGaussianRuntime:
    """ID는 화면 표에만 제공한다. 특징 및 발행용 latest에는 넣지 않는다."""

    model_id = 'multi-face-gaussian'
    model_version = 'multi-gaussian-0.2'
    max_participants = 32
    ttl_ms = 120000
    calibration_grace_ms = 3000

    def __init__(self, config=None):
        self.config = config or ModelConfig.gaussian()
        if self.config.scoring_method != 'gaussian':
            raise ValueError('Gaussian 설정이 필요합니다')
        self.reset()

    def _empty(self, reason):
        return dict(model_id=self.model_id, model_version=self.model_version,
                    config_version=self.config.config_version, status='deferred',
                    orientation_state='not_observable', angles=None, score=None,
                    current_direction_score=None, gaussian_observation=[],
                    reason_codes=[reason], detected_count=0, observable_count=0,
                    scoreable_count=0)

    def reset(self):
        self._identities = {}
        self._people = {}
        self.latest = self._empty('waiting_for_observation')

    def update_identities(self, mapping):
        """이 프레임에서 확정된 임시 트랙 연결만 별도로 받는다."""
        self._identities = {key: value for key, value in mapping.items()
                            if isinstance(key, int) and not isinstance(key, bool)
                            and key >= 0 and isinstance(value, str) and value}

    def participants(self):
        return [deepcopy(person['row']) for person in self._people.values()]

    def recalibrate(self):
        for person in self._people.values():
            if person['present']:
                person['calibration'].start()
                person['runtime'].reset()
                person['row'].update(calibration_status='collecting', status='calibrating',
                                     current_direction_score=None, score=None,
                                     reason_codes=['baseline_unavailable'])
        self.latest = self._empty('baseline_unavailable')

    def _new_person(self, identity, timestamp):
        calibration = HeadCalibration()
        calibration.start()
        return dict(calibration=calibration, runtime=GaussianRuntime(self.config),
                    last_seen=timestamp, last_valid_ms=None, present=False, track=None,
                    row=dict(id=identity, status='calibrating', calibration_status='collecting',
                             current_direction_score=None, score=None,
                             reason_codes=['baseline_unavailable']))

    def __call__(self, feature):
        try:
            # 헤더/공통 품질 검증은 기존 어댑터를 그대로 사용한다.
            probe = deepcopy(feature)
            probe['features']['faces'] = []
            parse_feature_window(probe, self.config)
            faces = feature['features']['faces']
            if not isinstance(faces, list) or any(not isinstance(f, dict) for f in faces):
                raise ValueError('얼굴 목록이 필요합니다')
        except (KeyError, TypeError, ValueError):
            for person in self._people.values():
                person['runtime'].reset()
                person['present'] = False
                person['row'].update(status='unavailable', score=None,
                                     current_direction_score=None, reason_codes=['invalid_feature_window'])
            self.latest = self._empty('invalid_feature_window')
            self.latest['status'] = 'input_error'
            return self.latest
        timestamp = feature['start_ms']
        for identity in list(self._people):
            if timestamp - self._people[identity]['last_seen'] > self.ttl_ms:
                del self._people[identity]
        connections = [(face, self._identities.get(face.get('transient_track_key'))) for face in faces]
        counts = Counter(identity for _, identity in connections if identity is not None)
        active = set()
        outputs = []
        for face, identity in connections:
            if identity is None or counts[identity] != 1:
                continue
            if identity not in self._people:
                if len(self._people) >= self.max_participants:
                    oldest = min(self._people, key=lambda key: self._people[key]['last_seen'])
                    del self._people[oldest]
                self._people[identity] = self._new_person(identity, timestamp)
            person = self._people[identity]
            track = face.get('transient_track_key')
            if not person['present'] or track != person['track']:
                person['runtime'].reset()
                # 미완료 보정은 연속 관측만 수집한다. 완료 기준은 복귀 후 유지한다.
                if person['calibration'].status != 'ready':
                    person['calibration'].start()
            person.update(present=True, track=track, last_seen=timestamp)
            active.add(identity)
            single = deepcopy(feature)
            single_face = deepcopy(face)
            single['features']['faces'] = [single_face]
            calibration = person['calibration']
            input_reason = calibration._input_reason([single_face])
            last_valid = person['last_valid_ms']
            # 확정 ID가 복귀했더라도 긴 관측 공백 뒤에는 기존 기준을 재사용하지 않는다.
            if (calibration.status == 'ready' and last_valid is not None and
                    (timestamp < last_valid or timestamp - last_valid > self.calibration_grace_ms)):
                calibration.start()
                person['runtime'].reset()
            # 불량 입력으로 완료된 개인 기준을 삭제하지 않는다.
            if input_reason is None:
                calibration.update([single_face], timestamp)
                person['last_valid_ms'] = timestamp
            else:
                single_face['head_angles_relative_degrees'] = None
                if calibration.status != 'ready':
                    calibration.update([single_face], timestamp)
            single['features']['head_calibration_status'] = calibration.status
            result = person['runtime'](single)
            person['row'].update(status=(result['status'] if calibration.status == 'ready' else 'calibrating'),
                                 calibration_status=calibration.status,
                                 current_direction_score=result.get('current_direction_score'),
                                 score=result['score'], reason_codes=list(result['reason_codes']))
            outputs.append(result)
        for identity, person in self._people.items():
            if identity not in active:
                person['present'] = False
                person['runtime'].reset()
                person['row'].update(status='missing', score=None, current_direction_score=None,
                                     reason_codes=['identity_not_observed'])
        primitives = [r['gaussian_observation'] for r in outputs if r.get('gaussian_observation') is not None]
        observable_count = sum(r['angles'] is not None for r in outputs)
        current = [r['current_direction_score'] for r in outputs if r.get('current_direction_score') is not None]
        recent = [r['score'] for r in outputs if r['score'] is not None]
        self.latest = self._empty('no_scoreable_observation')
        self.latest.update(angles={} if observable_count else None,
                           gaussian_observation=primitives,
                           current_direction_score=round(sum(current) / len(current), 1) if current else None,
                           score=round(sum(recent) / len(recent), 1) if recent else None,
                           detected_count=len(faces), observable_count=observable_count,
                           scoreable_count=len(primitives))
        candidates = [r.get('eye_candidate') for r in outputs]
        # 다인 비교는 모든 검출 얼굴이 ID와 후보 점수를 가진 경우에만 표시한다.
        if (candidates and len(outputs) == len(faces) and all(
                isinstance(item, dict) and item.get('candidate_total') is not None for item in candidates)):
            self.latest['eye_candidate'] = {
                key: round(sum(item[key] for item in candidates) / len(candidates), 1)
                for key in ('candidate_total', 'head_score', 'eye_score')}
            self.latest['eye_candidate'].update(status='group_candidate', count=len(candidates))
        else:
            self.latest['eye_candidate'] = {'candidate_total': None, 'head_score': None,
                                            'eye_score': None, 'status': 'group_unavailable'}
        self.latest['status'] = 'scored' if recent else 'deferred'
        self.latest['reason_codes'] = list(dict.fromkeys(
            reason for result in outputs for reason in result['reason_codes']))
        if len(active) < len(faces):
            self.latest['reason_codes'].append('identity_unconfirmed')
        if not outputs:
            self.latest['reason_codes'].append('no_scoreable_observation')
        return self.latest

    def aggregate_window(self, segments, start, end):
        """개별 원시 점수를 한 번 계산하여 사람·시간 최근성으로 합친다.

        품질 gate는 관측 가능한 벽시계 시간, 점수 분모는 유효 사람·시간이다.
        익명 각도만 받으므로 ID별 이력이나 누락 인원수를 추정하지 않는다.
        """
        duration = end - start
        window = round(self.config.window_seconds * 1000)
        if duration <= 0 or duration > window:
            raise ValueError('집계 구간은 설정한 양수 시간 범위 안이어야 합니다')
        observable_ms = scoreable_ms = person_ms = 0
        numerator = denominator = 0.0
        origin = end - window
        for left, right, observable, _, primitives in segments:
            left, right = max(start, left), min(end, right)
            if right <= left:
                continue
            if observable:
                observable_ms += right - left
            if not observable or not primitives:
                continue
            scoreable_ms += right - left
            u, v = (left - origin) / window, (right - origin) / window
            weight = (v - u) * (v + u) / 2
            for primitive in primitives:
                score = (primitive[2] if len(primitive) == 3 else
                         gaussian_angle_score(HeadAngles(primitive[0], primitive[1], 0), self.config))
                numerator += weight * score
                denominator += weight
                person_ms += right - left
        reasons = []
        if observable_ms / duration < self.config.min_observable_ratio:
            reasons.append('insufficient_observation')
        if scoreable_ms / duration < self.config.min_scoreable_ratio:
            reasons.append('insufficient_scoreable_time')
        if denominator <= 0:
            reasons.append('no_scoreable_time')
        return dict(score=round(numerator / denominator, 1) if not reasons else None,
                    observable_duration_ms=observable_ms, scoreable_duration_ms=scoreable_ms,
                    scoreable_person_duration_ms=person_ms, reason_codes=reasons)
