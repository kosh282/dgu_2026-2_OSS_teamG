"""재식별 없이 현재 얼굴 위치만으로 유지하는 단기 다인 점수."""
import math
from .multi_face import MultiFaceGaussianRuntime


class PositionTrackGaussianRuntime(MultiFaceGaussianRuntime):
    model_id = 'position-track-gaussian'
    model_version = 'position-gaussian-0.1'
    association_gap_ms = 1000
    maximum_center_shift = 0.10
    minimum_separation = 0.12
    aggregation_label = 'position-track-time-weighted-gaussian'

    def __init__(self, config=None):
        self._positions = {}
        self._position_time = None
        super().__init__(config)

    def reset(self):
        super().reset()
        self._positions = {}
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
        for i, a in enumerate(positions.values()):
            for b in list(positions.values())[i+1:]:
                if math.dist(a, b) < self.minimum_separation:
                    valid = False
        old = self._positions
        changed = (not valid or set(old) != set(positions) or
                   (self._position_time is not None and (timestamp is None or
                    not 0 < timestamp-self._position_time <= self.association_gap_ms)))
        if not changed:
            changed = any(math.dist(old[key], value) > self.maximum_center_shift
                          for key, value in positions.items())
        association_reset = changed and self._position_time is not None
        if changed:
            # 인원 변화·누락·큰 이동은 이전 트랙별 기준을 모두 폐기한다.
            self._people.clear()
        self._positions = positions if valid else {}
        self._position_time = timestamp if valid else None
        self.update_identities({key: f'T{key}' for key in positions} if valid else {})
        result = super().__call__(feature)
        if association_reset:
            result['reason_codes'] = list(dict.fromkeys(result['reason_codes'] + ['position_association_reset']))
        if not valid:
            result['reason_codes'] = list(dict.fromkeys(result['reason_codes'] + ['position_association_uncertain']))
        return result
