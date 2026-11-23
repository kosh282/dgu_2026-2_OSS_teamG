"""재식별 없이 현재 얼굴 위치만으로 유지하는 단기 다인 점수."""
import math
from .multi_face import MultiFaceGaussianRuntime


class PositionTrackGaussianRuntime(MultiFaceGaussianRuntime):
    model_id = 'position-track-gaussian'
    model_version = 'position-gaussian-0.1'
    association_gap_ms = 5000
    maximum_center_shift = 0.10
    minimum_separation = 0.12
    calibration_retention_ms = 5000
    aggregation_label = 'position-track-time-weighted-gaussian'

    def __init__(self, config=None, *, eye_fusion=False):
        self._positions = {}
        self._position_seen = {}
        self._position_time = None
        super().__init__(config, eye_fusion=eye_fusion)

    def reset(self):
        super().reset()
        self._positions = {}
        self._position_seen = {}
        self._position_time = None

    def __call__(self, feature):
        timestamp = feature.get('start_ms') if isinstance(feature, dict) else None
        features = feature.get('features') if isinstance(feature, dict) else None
        faces = features.get('faces', []) if isinstance(features, dict) else []
        positions = {}
        valid = (isinstance(timestamp, int) and not isinstance(timestamp, bool)
                 and isinstance(features, dict) and isinstance(faces, list))
        for face in faces if valid else []:
            if not isinstance(face, dict):
                valid = False
                break
            key, p = face.get('transient_track_key'), face.get('head_position')
            if (not isinstance(key, int) or isinstance(key, bool) or key < 0 or key in positions
                    or not isinstance(p, dict)):
                valid = False
                break
            x, y = p.get('x'), p.get('y')
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   or not 0 <= v <= 1 for v in (x, y)):
                valid = False
                break
            positions[key] = (x, y)
        uncertain = set()
        pairs = list(positions.items())
        for i, (key_a, a) in enumerate(pairs):
            for key_b, b in pairs[i+1:]:
                if math.dist(a, b) < self.minimum_separation:
                    uncertain.update((key_a, key_b))
        for key in uncertain:
            positions.pop(key, None)
            self._people.pop(f'T{key}', None)
            self._position_seen.pop(key, None)
        old = self._positions
        gap = (self._position_time is not None and (timestamp is None or
               not 0 < timestamp-self._position_time <= self.association_gap_ms))
        removed, added = set(old) - set(positions), set(positions) - set(old)
        # 교체가 의심되는 트랙만 버리고 계속 관측된 다른 참가자는 유지한다.
        ambiguous = not valid or gap
        association_reset = ambiguous and self._position_time is not None
        if ambiguous:
            self._people.clear()
        else:
            if removed and added:
                for key in removed:
                    self._people.pop(f'T{key}', None)
                    self._position_seen.pop(key, None)
            # 크게 이동한 트랙은 다른 얼굴과의 연결 가능성을 고려해 기준을 버린다.
            for key in set(old) & set(positions):
                person = self._people.get(f'T{key}')
                expired = (person is not None and not person['present'] and
                           timestamp - person['last_seen'] > self.calibration_retention_ms)
                if expired or math.dist(old[key], positions[key]) > self.maximum_center_shift:
                    self._people.pop(f'T{key}', None)
        if ambiguous:
            self._position_seen.clear()
        retained = {}
        if valid:
            for key in removed:
                last_seen = self._position_seen.get(key)
                if last_seen is not None and 0 < timestamp - last_seen <= self.calibration_retention_ms:
                    retained[key] = old[key]
                else:
                    self._people.pop(f'T{key}', None)
                    self._position_seen.pop(key, None)
            for key in positions:
                self._position_seen[key] = timestamp
        self._positions = {**retained, **positions} if valid else {}
        self._position_time = timestamp if valid else None
        self.update_identities({key: f'T{key}' for key in positions} if valid else {})
        result = super().__call__(feature)
        if association_reset:
            result['reason_codes'] = list(dict.fromkeys(result['reason_codes'] + ['position_association_reset']))
        if not valid or uncertain:
            result['reason_codes'] = list(dict.fromkeys(result['reason_codes'] + ['position_association_uncertain']))
        return result
