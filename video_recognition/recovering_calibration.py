"""프런트 단일 얼굴의 짧은 관측 누락 복구. 위치 트랙은 신원 확인이 아니다."""
from .head_calibration import HeadCalibration


class RecoveringHeadCalibration(HeadCalibration):
    def __init__(self, *args, grace_ms=3000, **kwargs):
        self.grace_ms = grace_ms
        self._last_valid_ms = None
        self._baseline_track = None
        self._collecting_track = None
        super().__init__(*args, **kwargs)

    def start(self):
        super().start()
        self._last_valid_ms = None
        self._baseline_track = None
        self._collecting_track = None

    def invalidate(self, reason_code='calibration_invalidated'):
        super().invalidate(reason_code)
        self._last_valid_ms = None
        self._baseline_track = None
        self._collecting_track = None

    @staticmethod
    def _track(faces):
        key = faces[0].get('transient_track_key') if len(faces) == 1 else None
        return key if isinstance(key, int) and not isinstance(key, bool) and key >= 0 else None

    def update(self, faces, capture_ms):
        reason = self._input_reason(faces)
        track = self._track(faces)
        if self.status == 'collecting':
            if reason is not None:
                self._collecting_track = None
            elif track is None:
                self.start()
                self.last_reason_code = 'calibration_track_unavailable'
                return self.status
            elif self._collecting_track is not None and track != self._collecting_track:
                self.start()
                self.last_reason_code = 'calibration_track_changed'
            if reason is None:
                self._collecting_track = track
        if self.status in ('ready', 'recovering'):
            gap = capture_ms - self._last_valid_ms if self._last_valid_ms is not None else None
            # 복수 얼굴, 트랙 변경, 긴 공백은 기존 사람으로 확정할 수 없다.
            unsafe = (reason == 'multiple_faces_detected' or gap is None or gap < 0 or gap > self.grace_ms
                      or (reason is None and (track is None or track != self._baseline_track)))
            if unsafe:
                why = reason or ('calibration_gap_expired' if gap is None or gap < 0 or gap > self.grace_ms
                                 else 'calibration_track_changed')
                self.start()
                self.last_reason_code = why
                return self.update(faces, capture_ms)
            if reason:
                for face in faces:
                    face['head_angles_relative_degrees'] = None
                    face['head_angles_relative_raw_degrees'] = None
                self.status = 'recovering'
                self.last_reason_code = reason
                return self.status
            self.status = 'ready'
            self.last_reason_code = None
        status = super().update(faces, capture_ms)
        if reason is None:
            self._last_valid_ms = capture_ms
            if status == 'ready':
                self._baseline_track = track
                if track is None:
                    self.start()
                    self.last_reason_code = 'calibration_track_unavailable'
                    return self.status
        return status
