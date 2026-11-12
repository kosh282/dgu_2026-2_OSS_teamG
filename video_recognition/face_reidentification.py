"""동의한 참가자의 세션 내 메모리 재식별. 결과를 이벤트/로그에 넣지 않는다."""
from pathlib import Path
from types import SimpleNamespace

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


class MultiIdentityGallery(IdentityGallery):
    """같은 프레임의 후보를 함께 판정해 중복 ID와 등록 순서 의존을 막는다."""

    def __init__(self):
        super().__init__()
        self.pending_tracks = {}

    def clear(self):
        super().clear()
        self.pending_tracks.clear()

    def observe_many(self, embeddings, timestamp_ms, reserved):
        results = {key: {'identity': None, 'state': 'deferred'} for key in embeddings}
        if self.last_ms is not None and timestamp_ms <= self.last_ms:
            self.pending_tracks.clear()
            return results
        self.last_ms = timestamp_ms
        for identity, entry in list(self.entries.items()):
            if identity not in reserved and timestamp_ms - entry['last_seen'] > self.retention_ms:
                del self.entries[identity]
        vectors = {key: self._vector(value) for key, value in embeddings.items()}
        candidates = {}
        for key, vector in vectors.items():
            if vector is None or any(e['vector'].shape != vector.shape for e in self.entries.values()):
                continue
            scores = sorted(((float(vector @ e['vector']), identity)
                             for identity, e in self.entries.items()), reverse=True)
            best, candidate = scores[0] if scores else (-1, None)
            if scores and best >= self.threshold:
                if candidate in reserved or (len(scores) > 1 and best - scores[1][0] < self.margin):
                    continue
            elif best >= self.new_threshold:
                continue
            else:
                candidate = None
            candidates[key] = candidate
        # 같은 ID 후보를 요구하거나 서로 비슷한 미확인 얼굴은 모두 보류한다.
        conflicts = set()
        keys = list(candidates)
        for i, key in enumerate(keys):
            for other in keys[i + 1:]:
                if (candidates[key] is not None and candidates[key] == candidates[other]) or (
                        vectors[key].shape == vectors[other].shape
                        and float(vectors[key] @ vectors[other]) >= self.new_threshold):
                    conflicts.update((key, other))
        next_pending = {}
        for key, candidate in candidates.items():
            if key in conflicts:
                continue
            vector = vectors[key]
            previous = self.pending_tracks.get(key)
            continuous = (previous is not None and previous['candidate'] == candidate
                          and 0 < timestamp_ms - previous['time'] <= 1500
                          and previous['vector'].shape == vector.shape
                          and float(previous['vector'] @ vector) >= self.threshold)
            count = previous['count'] + 1 if continuous else 1
            next_pending[key] = {'candidate': candidate, 'time': timestamp_ms,
                                 'vector': vector.copy(), 'count': count}
            if count < self.confirmations:
                results[key]['state'] = 'confirming'
                continue
            if candidate is None:
                if len(self.entries) >= self.max_identities:
                    continue
                candidate = f'S{self.next_id:03d}'
                self.next_id += 1
                self.entries[candidate] = {'vector': vector.copy(), 'last_seen': timestamp_ms}
                state = 'new'
            else:
                state = 'reconnected'
                self.entries[candidate]['last_seen'] = timestamp_ms
            results[key] = {'identity': candidate, 'state': state}
        self.pending_tracks = next_pending
        return results


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
        self.gallery = MultiIdentityGallery()
        self.connections = {}
        self.last_results = {}
        self.last_inference_ms = None
        self.last_observation_ms = None

    @staticmethod
    def _geometry(faces, boxes):
        """배열 순서는 같은 프레임의 대응에만 사용하고 트랙 키로 상태를 연결한다."""
        if len(faces) != len(boxes):
            return {}
        geometry = {}
        duplicates = set()
        for face, box in zip(faces, boxes):
            key = face.get('transient_track_key')
            box = np.asarray(box, dtype=float)
            if (isinstance(key, bool) or not isinstance(key, int) or key < 0
                    or box.shape != (4,) or not np.isfinite(box).all()
                    or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
                continue
            if key in geometry:
                duplicates.add(key)
            geometry[key] = box
        # 겹친 얼굴은 위치 연결만으로 사람을 구별하지 않는다.
        keys = list(geometry)
        for i, key in enumerate(keys):
            a = geometry[key]
            for other in keys[i + 1:]:
                b = geometry[other]
                overlap = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
                area = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
                if overlap > area * .15:
                    duplicates.update((key, other))
        return {key: box for key, box in geometry.items() if key not in duplicates}

    def identify(self, frame, timestamp_ms, faces, *, boxes):
        """표시 없이 세션 내 ID 연결을 갱신하고 독립 사본을 반환한다."""
        geometry = self._geometry(faces, boxes)
        gap = timestamp_ms - self.last_observation_ms if self.last_observation_ms is not None else None
        if gap is not None and (gap <= 0 or gap > 1500):
            self.connections.clear()
            self.gallery.pending_tracks.clear()
            self.last_inference_ms = None
        self.last_observation_ms = timestamp_ms
        self.connections = {key: identity for key, identity in self.connections.items() if key in geometry}
        self.gallery.pending_tracks = {key: state for key, state in self.gallery.pending_tracks.items() if key in geometry}
        reserved = set(self.connections.values())
        for identity in reserved:
            self.gallery.entries[identity]['last_seen'] = timestamp_ms
        unknown = [key for key in geometry if key not in self.connections]
        results = {}
        if unknown and (self.last_inference_ms is None or timestamp_ms - self.last_inference_ms >= 500):
            self.last_inference_ms = timestamp_ms
            embeddings = {key: None for key in unknown}
            reasons = {key: 'no_quality_match' for key in unknown}
            # 좁게 자른 영상에서는 SCRFD가 놓치는 얼굴이 있어 전체 문맥으로 검출한다.
            # 검출과 특징 추론을 분리해 이미 확인된 사람의 임베딩은 다시 계산하지 않는다.
            detected_boxes, keypoints = self.app.det_model.detect(frame, max_num=0, metric='default')
            height, width = frame.shape[:2]
            pixel_boxes = {key: box * np.array([width, height, width, height])
                           for key, box in geometry.items()}
            possible = {key: [] for key in geometry}
            for index, detected in enumerate(detected_boxes):
                detected = np.asarray(detected, dtype=float)
                if (detected.shape != (5,) or not np.isfinite(detected).all()
                        or detected[4] < .7
                        or min(detected[2]-detected[0], detected[3]-detected[1]) < 80):
                    continue
                for key, target in pixel_boxes.items():
                    cx, cy = (target[0]+target[2])*.5, (target[1]+target[3])*.5
                    intersection = max(0., min(target[2], detected[2])-max(target[0], detected[0])) * max(0., min(target[3], detected[3])-max(target[1], detected[1]))
                    target_area = (target[2]-target[0])*(target[3]-target[1])
                    detected_area = (detected[2]-detected[0])*(detected[3]-detected[1])
                    union = target_area + detected_area - intersection
                    if (detected[0] <= cx <= detected[2] and detected[1] <= cy <= detected[3]
                            and union > 0 and intersection/union >= .2):
                        possible[key].append(index)
            for key in unknown:
                matches = possible[key]
                if len(matches) != 1:
                    if len(matches) > 1:
                        reasons[key] = 'ambiguous_detection'
                    continue
                index = matches[0]
                if sum(index in values for values in possible.values()) != 1:
                    reasons[key] = 'shared_detection'
                    continue
                if (keypoints is None or index >= len(keypoints)
                        or np.asarray(keypoints[index]).shape != (5, 2)
                        or not np.isfinite(keypoints[index]).all()):
                    continue
                detected = detected_boxes[index]
                face = SimpleNamespace(bbox=detected[:4], det_score=detected[4], kps=keypoints[index])
                self.app.models['recognition'].get(frame, face)
                embeddings[key] = face.embedding
                reasons[key] = 'uncertain_identity'
            results = self.gallery.observe_many(embeddings, timestamp_ms, reserved)
            for key, result in results.items():
                if result['state'] == 'deferred':
                    result['reason'] = reasons[key]
                if result['identity'] is not None:
                    self.connections[key] = result['identity']
        else:
            for identity, entry in list(self.gallery.entries.items()):
                if identity not in reserved and timestamp_ms-entry['last_seen'] > self.gallery.retention_ms:
                    del self.gallery.entries[identity]
        self.last_results = results
        return dict(self.connections)

    def render(self, frame, faces, *, boxes):
        """마지막 분석 결과만 복사본에 그리며 추론·시간 상태를 바꾸지 않는다."""
        import cv2
        output = frame.copy()
        geometry = self._geometry(faces, boxes)
        results = getattr(self, 'last_results', {})
        for key, box in geometry.items():
            identity = self.connections.get(key)
            state = results.get(key, {}).get('state', 'tracked' if identity else 'waiting')
            label = f"ReID {identity or '--'}: {state}"
            if results.get(key, {}).get('reason'):
                label += f" ({results[key]['reason']})"
            if state == 'tracked':
                label += ' (inference paused)'
            x1, y1, x2, y2 = (box * np.array([frame.shape[1], frame.shape[0]] * 2)).astype(int)
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 255), 2)
            cv2.putText(output, label, (max(0, x1), max(20, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 255), 1)
        if len(geometry) != len(faces) or not faces:
            cv2.putText(output, 'ReID: missing/overlapping faces deferred', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 255), 1)
        return output

    def annotate(self, frame, timestamp_ms, faces, *, boxes):
        """기존 데모용 추론·표시 API를 유지한다."""
        self.identify(frame, timestamp_ms, faces, boxes=boxes)
        return self.render(frame, faces, boxes=boxes)

    def close(self):
        self.last_results = {}
        self.gallery.clear()
        self.connections.clear()
        self.app = None
        self.last_inference_ms = None
        self.last_observation_ms = None
