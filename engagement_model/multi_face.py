"""세션 메모리 전용 다인 보정·점수와 익명 Gaussian 집계."""

from collections import Counter, deque
from copy import deepcopy
import math

from video_recognition.head_calibration import HeadCalibration
from .config import ModelConfig
from .gaussian import GaussianRuntime, gaussian_angle_score, gaussian_window_score
from .input_adapter import HeadAngles, parse_feature_window


class MultiFaceGaussianRuntime:
    """ID는 화면 표에만 제공한다. 특징 및 발행용 latest에는 넣지 않는다."""

    model_id = 'multi-face-gaussian'
    model_version = 'multi-gaussian-0.2'
    max_participants = 32
    ttl_ms = 120000
    calibration_grace_ms = 5000
    component_hold_ms = 2000

    def __init__(self, config=None, *, eye_fusion=False):
        self.config = config or ModelConfig.gaussian()
        self.eye_fusion = eye_fusion
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

    def recalibrate(self, identity=None):
        changed = False
        for key, person in self._people.items():
            if person['present'] and (identity is None or key == identity):
                changed = True
                person['calibration'].start()
                person['runtime'].reset()
                person['window_samples'].clear()
                person['components'] = {}
                person['row'].update(calibration_status='collecting', status='calibrating',
                                     current_direction_score=None, score=None,
                                     head_score=None, eye_score=None, current_head_score=None, current_eye_score=None,
                                     effective_head_score=None, effective_eye_score=None, held_components=[], held_component_ages_ms={},
                                     eye_state=None, eye_closed_seconds=None,
                                     downward_status=None, looking_down_seconds=None,
                                     eye_coverage=None, relative_angles=None,
                                     activity_state='activity_unavailable', activity_down_start_ms=None,
                                     activity_down_duration_seconds=None,
                                     reason_codes=['baseline_unavailable'])
        if changed:
            self.latest = self._empty('baseline_unavailable')
        return changed

    def _new_person(self, identity, timestamp):
        calibration = HeadCalibration()
        calibration.start()
        return dict(calibration=calibration, runtime=GaussianRuntime(self.config),
                    last_seen=timestamp, last_valid_ms=None, present=False, track=None, components={},
                    window_samples=deque(),
                    row=dict(id=identity, status='calibrating', calibration_status='collecting',
                             current_direction_score=None, score=None,
                             activity_state='activity_unavailable', activity_down_start_ms=None,
                             activity_down_duration_seconds=None,
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
                person['window_samples'].clear()
                person['components'].clear()
                person['present'] = False
                person['row'].update(status='unavailable', score=None,
                                     current_direction_score=None, head_score=None, eye_score=None,
                                     current_head_score=None, current_eye_score=None, eye_coverage=None,
                                     effective_head_score=None, effective_eye_score=None, held_components=[], held_component_ages_ms={},
                                     eye_state=None, eye_closed_seconds=None,
                                     downward_status=None, looking_down_seconds=None,
                                     activity_state='activity_unavailable', activity_down_start_ms=None,
                                     activity_down_duration_seconds=None,
                                     relative_angles=None, reason_codes=['invalid_feature_window'])
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
                if track != person['track'] or not person['present']:
                    person['components'].clear()
                short_gap = (not person['present'] and track == person['track']
                             and 0 < timestamp - person['last_seen'] <= self.calibration_grace_ms)
                if not short_gap:
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
                person['window_samples'].clear()
                person['components'].clear()
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
            eye = (result.get('eye_candidate') or {}).get('eye_score')
            eye_current = (result.get('eye_candidate') or {}).get('current_eye_score')
            # 같은 추적에서 한 항목만 빠진 경우에 한해 최근 값을 제한적으로 사용한다.
            head_current = result.get('current_direction_score')
            components = person['components']
            for name, value in (('head', head_current), ('eye', eye_current)):
                if value is not None:
                    components[name] = (value, timestamp)
            held = []
            effective = {'head': head_current, 'eye': eye_current}
            if self.eye_fusion and (head_current is None) != (eye_current is None):
                name = 'head' if head_current is None else 'eye'
                previous = components.get(name)
                if previous and 0 <= timestamp - previous[1] <= self.component_hold_ms:
                    effective[name] = previous[0]
                    held.append(name)
            fused_current = (round((1 - self.config.eye_weight) * effective['head'] + self.config.eye_weight * effective['eye'], 1)
                             if self.eye_fusion and effective['head'] is not None
                             and effective['eye'] is not None else None)
            fused_recent = (result.get('eye_candidate') or {}).get('candidate_total') if self.eye_fusion else None
            if held and fused_current is not None:
                fused_recent = fused_current
            person['row'].update(status=(('deferred' if self.eye_fusion and fused_current is None else result['status'])
                                         if calibration.status == 'ready' else 'calibrating'),
                                 calibration_status=calibration.status,
                                 current_direction_score=fused_current if self.eye_fusion else result.get('current_direction_score'),
                                 score=fused_recent if self.eye_fusion else result['score'],
                                 head_score=result.get('score'), eye_score=eye,
                                 current_head_score=result.get('current_direction_score'), current_eye_score=eye_current,
                                 effective_head_score=effective['head'], effective_eye_score=effective['eye'],
                                 held_components=held, held_component_ages_ms={name: timestamp-components[name][1] for name in held},
                                 eye_diagnostics=person['runtime']._eye_candidate.diagnostics(),
                                 eye_state=(result.get('eye_candidate') or {}).get('eye_state'),
                                 eye_closed_seconds=(result.get('eye_candidate') or {}).get('continuous_closed_seconds'),
                                 downward_status=result.get('downward_status'),
                                 looking_down_seconds=result.get('looking_down_seconds'),
                                 eye_coverage=(result.get('eye_candidate') or {}).get('coverage'),
                                 relative_angles=result.get('angles'),
                                 activity_state=result['activity_state'],
                                 activity_down_start_ms=result['activity_down_start_ms'],
                                 activity_down_duration_seconds=result['activity_down_duration_seconds'],
                                 reason_codes=list(result['reason_codes']) + (['eye_observation_unavailable']
                                     if self.eye_fusion and eye is None else []))
            outputs.append({**result, '_fused': fused_current, '_held': bool(held), '_recent': fused_recent})
            primitive = result.get('gaussian_observation')
            if self.eye_fusion:
                primitive = (None, None, fused_current) if fused_current is not None else None
            self._record_window_sample(person, timestamp,
                                       fused_current is not None if self.eye_fusion else result['angles'] is not None,
                                       primitive)
        for identity, person in self._people.items():
            if identity not in active:
                person['components'].clear()
                was_present = person['present']
                person['present'] = False
                if counts[identity] > 1:
                    person['calibration'].start()
                    person['runtime'].reset()
                    person['window_samples'].clear()
                elif timestamp - person['last_seen'] > self.calibration_grace_ms:
                    person['runtime'].reset()
                elif was_present:
                    person['runtime'].pause_observation()
                person['row'].update(status='missing', score=None, current_direction_score=None,
                                     head_score=None, eye_score=None, current_head_score=None, current_eye_score=None,
                                     effective_head_score=None, effective_eye_score=None, held_components=[], held_component_ages_ms={},
                                     eye_state=None, eye_closed_seconds=None,
                                     downward_status=None, looking_down_seconds=None,
                                     eye_diagnostics={'reason': 'face_not_detected'},
                                     eye_coverage=None, relative_angles=None,
                                     activity_state='activity_unavailable', activity_down_start_ms=None,
                                     activity_down_duration_seconds=None,
                                     reason_codes=['identity_not_observed'])
                self._record_window_sample(person, timestamp, False, None)
        if self.eye_fusion:
            # 같은 시각의 머리·눈 점수만 결합해 publisher에서 최근 구간을 한 번 집계한다.
            primitives = [(None, None, r['_fused'], r['_held']) for r in outputs if r['_fused'] is not None]
        else:
            primitives = [r['gaussian_observation'] for r in outputs if r.get('gaussian_observation') is not None]
        observable_count = sum((r['_fused'] is not None if self.eye_fusion else r['angles'] is not None) for r in outputs)
        current = ([p[2] for p in primitives] if self.eye_fusion else
                   [r['current_direction_score'] for r in outputs if r.get('current_direction_score') is not None])
        recent = ([r['_recent'] for r in outputs if r['_recent'] is not None] if self.eye_fusion else
                  [r['score'] for r in outputs if r['score'] is not None])
        self.latest = self._empty('no_scoreable_observation')
        self.latest.update(angles={} if observable_count else None,
                           gaussian_observation=primitives,
                           current_direction_score=round(sum(current) / len(current), 1) if current else None,
                           score=round(sum(recent) / len(recent), 1) if recent else None,
                           detected_count=len(faces), observable_count=observable_count,
                           scoreable_count=len(primitives))
        candidates = [r.get('eye_candidate') for r in outputs]
        # 최종 경로는 유효 관측만 집계하고, 비교 전용 경로는 모든 얼굴이 유효할 때만 표시한다.
        usable_candidates = [item for item in candidates if isinstance(item, dict)
                             and item.get('candidate_total') is not None]
        if usable_candidates and (self.eye_fusion or len(usable_candidates) == len(faces)):
            self.latest['eye_candidate'] = {
                key: round(sum(item[key] for item in usable_candidates) / len(usable_candidates), 1)
                for key in ('candidate_total', 'head_score', 'eye_score')}
            self.latest['eye_candidate'].update(status='group_candidate', count=len(usable_candidates),
                                                detected_count=len(faces))
        else:
            self.latest['eye_candidate'] = {'candidate_total': None, 'head_score': None,
                                            'eye_score': None, 'status': 'group_unavailable'}
        self.latest['status'] = 'scored' if recent else 'deferred'
        self.latest['reason_codes'] = list(dict.fromkeys(
            reason for result in outputs for reason in result['reason_codes']))
        if self.eye_fusion and len(primitives) < len(outputs):
            self.latest['reason_codes'].append('eye_observation_unavailable')
        if len(active) < len(faces):
            self.latest['reason_codes'].append('identity_unconfirmed')
        if not outputs:
            self.latest['reason_codes'].append('no_scoreable_observation')
        return self.latest

    def _record_window_sample(self, person, timestamp, observable, primitive):
        """하위 집계용 원시 점수만 메모리에 유지한다. 개인 키/이력은 발행하지 않는다."""
        samples = person['window_samples']
        if samples and timestamp <= samples[-1][0]:
            samples.clear()
        samples.append((timestamp, observable, primitive))
        cutoff = timestamp - round(self.config.window_seconds * 1000)
        # 구간 왼쪽 경계를 가로지르는 관측 하나는 유지한다.
        while len(samples) > 1 and samples[1][0] <= cutoff:
            samples.popleft()

    @staticmethod
    def _lower_20_fields(scores):
        values = sorted(value for value in scores if isinstance(value, (int, float))
                        and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 100)
        count = max(1, math.ceil(len(values) * 0.2)) if values else 0
        return dict(lower_20_mean_score=round(sum(values[:count]) / count, 1) if count else None,
                    lower_20_count=count, lower_20_valid_count=len(values))

    def _lower_20_window(self, start, end):
        """같은 발행 구간의 개인별 시간 집계 후 하위 인원을 선정한다.

        미검출/보류 또는 관측 유지 상한을 지난 대상은 제외한다. 과거 이동평균을
        다시 평균하지 않고 기존 선형 시간 적분과 개인별 품질 gate를 재사용한다.
        """
        scores = []
        hold_ms = round(self.config.max_observation_hold_seconds * 1000)
        for person in self._people.values():
            samples = [sample for sample in person['window_samples'] if sample[0] < end]
            if not samples:
                continue
            timestamp, observable, primitive = samples[-1]
            if not observable or primitive is None or end - timestamp > hold_ms:
                continue
            segments = []
            for index, (timestamp, observable, primitive) in enumerate(samples):
                stop = min(end, timestamp + hold_ms,
                           samples[index + 1][0] if index + 1 < len(samples) else end)
                if stop > timestamp:
                    segments.append((timestamp, stop, observable, None, primitive))
            value = gaussian_window_score(segments, start, end, self.config)['score']
            if value is not None:
                scores.append(value)
        return self._lower_20_fields(scores)

    def aggregate_window(self, segments, start, end):
        """개별 원시 점수를 한 번 계산하여 사람·시간 최근성으로 합친다.

        품질 gate는 관측 가능한 벽시계 시간, 점수 분모는 유효 사람·시간이다.
        기존 전체 점수는 익명 각도로 계산한다. 추가 하위 지표만 실행 메모리의
        개인별 원시 표본을 사용하며 키나 개인별 결과를 발행하지 않는다.
        """
        duration = end - start
        window = round(self.config.window_seconds * 1000)
        if duration <= 0 or duration > window:
            raise ValueError('집계 구간은 설정한 양수 시간 범위 안이어야 합니다')
        observable_ms = scoreable_ms = person_ms = held_ms = 0
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
            if any(len(p) > 3 and p[3] for p in primitives):
                held_ms += right - left
            u, v = (left - origin) / window, (right - origin) / window
            weight = (v - u) * (v + u) / 2
            for primitive in primitives:
                score = (primitive[2] if len(primitive) >= 3 else
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
                    scoreable_person_duration_ms=person_ms, held_duration_ms=held_ms, reason_codes=reasons,
                    **(self._lower_20_window(start, end) if not reasons else self._lower_20_fields([])))
