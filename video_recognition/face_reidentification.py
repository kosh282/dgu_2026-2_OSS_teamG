"""동의한 단일 인물 시험용 메모리 재식별. 결과를 이벤트/로그에 넣지 않는다."""
from pathlib import Path

import numpy as np


class IdentityGallery:
    """검증 전 임계값으로 벡터를 비교한다. 실제 신원을 보장하지 않는다."""

    def __init__(self, threshold=0.5, new_threshold=0.3, margin=0.1,
                 confirmations=3, retention_ms=120_000, max_identities=32):
        self.threshold = threshold
        self.new_threshold = new_threshold
        self.margin = margin
        self.confirmations = confirmations
        self.retention_ms = retention_ms
        self.max_identities = max_identities
        self.entries = {}
        self.next_id = 1
        self.pending = None
        self.last_ms = None

    def __len__(self):
        return len(self.entries)

    def clear(self):
        self.entries.clear()
        self.pending = None
        self.last_ms = None
        self.next_id = 1

    @staticmethod
    def _vector(value):
        if value is None:
            return None
        try:
            vector = np.asarray(value, dtype=np.float32)
        except (ValueError, TypeError):
            return None
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            return None
        norm = np.linalg.norm(vector)
        return vector / norm if np.isfinite(norm) and norm > 1e-8 else None

    def observe(self, embedding, timestamp_ms):
        result = {'identity': None, 'state': 'deferred', 'similarity': None}
        # 시간 역전/동일 시각으로 연속 확인 횟수를 부풀리지 않는다.
        if self.last_ms is not None and timestamp_ms <= self.last_ms:
            self.pending = None
            return result
        self.last_ms = timestamp_ms
        for key, entry in list(self.entries.items()):
            if timestamp_ms - entry['last_seen'] > self.retention_ms:
                del self.entries[key]
        vector = self._vector(embedding)
        if vector is None:
            self.pending = None
            return result
        if any(entry['vector'].shape != vector.shape for entry in self.entries.values()):
            self.pending = None
            return result
        scores = sorted(((float(vector @ entry['vector']), key)
                         for key, entry in self.entries.items()), reverse=True)
        best, candidate = scores[0] if scores else (-1.0, None)
        result['similarity'] = best if scores else None
        if scores and best >= self.threshold:
            if len(scores) > 1 and best - scores[1][0] < self.margin:
                self.pending = None
                return result
        elif scores and best >= self.new_threshold:
            self.pending = None
            return result
        else:
            candidate = None
        previous = self.pending
        consecutive = (previous is not None and previous['candidate'] == candidate
                       and timestamp_ms - previous['time'] <= 1500
                       and previous['vector'].shape == vector.shape
                       and float(previous['vector'] @ vector) >= self.threshold)
        count = previous['count'] + 1 if consecutive else 1
        self.pending = {'candidate': candidate, 'vector': vector.copy(),
                        'count': count, 'time': timestamp_ms}
        if count < self.confirmations:
            result['state'] = 'confirming'
            return result
        if candidate is None:
            if len(self.entries) >= self.max_identities:
                self.pending = None
                return result
            candidate = f'S{self.next_id:03d}'
            self.next_id += 1
            self.entries[candidate] = {'vector': vector.copy(), 'last_seen': timestamp_ms}
            result['state'] = 'new'
        else:
            entry = self.entries[candidate]
            result['state'] = ('reconnected' if timestamp_ms - entry['last_seen'] > 1500
                               else 'matched')
            entry['last_seen'] = timestamp_ms
        result['identity'] = candidate
        return result


class FaceReidentifier:
    """InsightFace 추론 결과를 표시 전용 ID에 연결하며 디스크에 기록하지 않는다."""

    def __init__(self, model_root=None):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError('재식별 패키지를 설치하세요: pip install -r '
                               'video_recognition/requirements-reidentification.txt') from exc
        root = Path(model_root) if model_root else Path(__file__).parent / '.venv' / 'reid-cache'
        self.app = FaceAnalysis(name='buffalo_l', root=str(root.resolve()),
                                allowed_modules=['detection', 'recognition'],
                                providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        self.gallery = IdentityGallery()
        self.last_inference_ms = None

    def annotate(self, frame, timestamp_ms, face_count):
        """표시용 복사본만 변경하며 FeatureWindow와 진단 로그는 수정하지 않는다."""
        import cv2
        output = frame.copy()
        # 추론 사이의 프레임에서도 미검출/복수 얼굴은 연속 확인을 끊는다.
        if face_count != 1:
            self.gallery.observe(None, timestamp_ms)
            cv2.putText(output, 'ReID --: deferred', (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            return output
        # 추론하지 않은 프레임에는 이전 ID를 새 판정처럼 표시하지 않는다.
        if self.last_inference_ms is not None and timestamp_ms - self.last_inference_ms < 500:
            cv2.putText(output, 'ReID: waiting for next sample', (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            return output
        self.last_inference_ms = timestamp_ms
        faces = self.app.get(frame)
        embedding = None
        box = None
        if len(faces) == 1:
            face = faces[0]
            box = np.asarray(face.bbox)
            if (box.shape == (4,) and np.isfinite(box).all()
                    and min(box[2] - box[0], box[3] - box[1]) >= 80
                    and float(face.det_score) >= 0.8):
                embedding = face.embedding
        result = self.gallery.observe(embedding, timestamp_ms)
        identity = result['identity'] or '--'
        label = f"ReID {identity}: {result['state']}"
        if result['similarity'] is not None:
            label += f" cosine={result['similarity']:.2f}"
        cv2.putText(output, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 220, 255), 2)
        if box is not None and embedding is not None:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 255), 2)
        return output

    def close(self):
        self.gallery.clear()
        self.app = None
        self.last_inference_ms = None
