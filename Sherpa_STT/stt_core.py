"""Team contract v0.1.0 streaming STT. Raw audio is never written to JSON."""
from __future__ import annotations
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

SCHEMA_VERSION = '0.1.0'
MODEL_ID = 'sherpa-onnx-streaming-zipformer-korean-2024-06-16'


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AudioChunk:
    session_id: str
    chunk_id: int
    start_ms: int
    end_ms: int
    sample_rate_hz: int
    channels: int
    sample_format: str
    samples: np.ndarray


class Session:
    def __init__(self, title: str, session_id: str | None = None, clock_origin: float | None = None,
                 time_origin_ms: int = 0, model_id: str = MODEL_ID):
        self.session_id = session_id or str(uuid.uuid4())
        self.started_at_utc = now_utc()
        self.origin = time.monotonic() if clock_origin is None else clock_origin
        self.time_origin_ms = time_origin_ms
        self.title = title
        self.model_id = model_id
        self.events: list[dict[str, Any]] = []

    def elapsed_ms(self) -> int:
        return max(0, self.time_origin_ms + round((time.monotonic() - self.origin) * 1000))

    def envelope(self, event_type: str, start_ms: int, end_ms: int, payload: dict,
                 quality: str = 'valid', reasons: list[str] | None = None) -> dict:
        if not (0 <= start_ms <= end_ms):
            raise ValueError('Invalid time interval')
        return dict(schema_version=SCHEMA_VERSION, session_id=self.session_id,
                    event_id=str(uuid.uuid4()), event_type=event_type,
                    start_ms=start_ms, end_ms=end_ms, created_ms=self.elapsed_ms(),
                    created_at_utc=None, payload=payload,
                    quality={'status': quality, 'reason_codes': reasons or []},
                    provenance={'module': 'speech', 'config_version': SCHEMA_VERSION,
                                'model_id': self.model_id, 'model_version': '2024-06-16'})

    def add(self, event: dict) -> None:
        # Partials are never persisted; final utterance is unique.
        if event['event_type'] == 'transcript_partial':
            return
        if event['event_type'] == 'transcript_final':
            uid = event['payload']['utterance_id']
            self.events = [e for e in self.events if not (e['event_type'] == 'transcript_final'
                           and e['payload']['utterance_id'] == uid)]
        self.events.append(event)

    def save(self, path: str | Path, duration_ms: int | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        end = max([e['end_ms'] for e in self.events], default=0)
        doc = {'schema_version': SCHEMA_VERSION,
               'session': {'session_id': self.session_id, 'title': self.title,
                           'started_at_utc': self.started_at_utc,
                           'duration_ms': max(end, duration_ms or 0)},
               'config': {'config_version': SCHEMA_VERSION, 'speech_model_id': self.model_id,
                          'speech_model_version': '2024-06-16'},
               'events': self.events}
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding='utf-8')
        temp.replace(path)


def create_recognizer(model_dir: str, num_threads: int = 2):
    import sherpa_onnx
    p = Path(model_dir)
    names = ['tokens.txt', 'encoder-epoch-99-avg-1.int8.onnx',
             'decoder-epoch-99-avg-1.onnx', 'joiner-epoch-99-avg-1.int8.onnx']
    missing = [name for name in names if not (p / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing model files in {p}: {missing}')
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(p / names[0]), encoder=str(p / names[1]),
        decoder=str(p / names[2]), joiner=str(p / names[3]),
        num_threads=num_threads, sample_rate=16000,
        enable_endpoint_detection=True, rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2, rule3_min_utterance_length=20.0)


class STTProcessor:
    """Accepts contiguous 16k mono chunks; endpoint intervals are estimates, not word alignment."""
    def __init__(self, session: Session, recognizer: Any,
                 on_event: Callable[[dict], None] | None = None):
        self.session, self.recognizer = session, recognizer
        self.stream = recognizer.create_stream()
        self.on_event = on_event or (lambda e: None)
        self.utterance_id = str(uuid.uuid4())
        self.revision = 0
        self.partial_text = ''
        self.utterance_start_ms: int | None = None
        self.last_end_ms: int | None = None

    def _emit(self, event_type: str, start: int, end: int, text: str):
        self.revision += 1
        payload = {'utterance_id': self.utterance_id, 'revision': self.revision, 'text': text}
        if event_type == 'transcript_final':
            payload['is_edited'] = False
        event = self.session.envelope(event_type, start, end, payload,
                                      quality='partial' if event_type == 'transcript_partial' else 'valid')
        self.on_event(event)
        self.session.add(event)

    def accept(self, chunk: AudioChunk) -> None:
        if chunk.session_id != self.session.session_id:
            raise ValueError('AudioChunk session_id mismatch')
        if chunk.channels != 1 or chunk.sample_rate_hz != 16000 or chunk.sample_format != 'float32':
            raise ValueError('Required: 16000 Hz, mono, float32; resample at capture boundary')
        if chunk.start_ms < 0 or chunk.end_ms <= chunk.start_ms:
            raise ValueError('Invalid chunk timing')
        if self.last_end_ms is not None and abs(chunk.start_ms - self.last_end_ms) > 2:
            raise ValueError('Audio discontinuity; signal analysis_status and restart decoder')
        samples = np.asarray(chunk.samples, dtype=np.float32).reshape(-1)
        expected = round((chunk.end_ms - chunk.start_ms) * 16)
        if abs(len(samples) - expected) > 32:
            raise ValueError('Chunk duration and samples disagree')
        self.last_end_ms = chunk.end_ms
        if self.utterance_start_ms is None:
            self.utterance_start_ms = chunk.start_ms
        self.stream.accept_waveform(16000, samples)
        self._decode(chunk.end_ms)

    def _decode(self, end_ms: int) -> None:
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        result = self.recognizer.get_result(self.stream)
        text = (result.text if hasattr(result, 'text') else str(result)).strip()
        if text != self.partial_text and text:
            self.partial_text = text
            self._emit('transcript_partial', self.utterance_start_ms or 0, end_ms, text)
        if self.recognizer.is_endpoint(self.stream):
            self._finalize(end_ms, text)
            self.recognizer.reset(self.stream)

    def _finalize(self, end_ms: int, text: str) -> None:
        if text.strip():
            self._emit('transcript_final', self.utterance_start_ms or 0, end_ms, text.strip())
        self.utterance_id = str(uuid.uuid4())
        self.revision = 0
        self.partial_text = ''
        self.utterance_start_ms = end_ms

    def finish(self) -> None:
        if self.last_end_ms is None:
            return
        # Signal end-of-input and process pending output. Tail is for recognizer flushing only.
        self.stream.accept_waveform(16000, np.zeros(8000, dtype=np.float32))
        self.stream.input_finished()
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        result = self.recognizer.get_result(self.stream)
        text = (result.text if hasattr(result, 'text') else str(result)).strip()
        if text:
            self._finalize(self.last_end_ms, text)


def audio_chunks_from_f32(data: np.ndarray, session_id: str, chunk_id: int, start_sample: int, offset_ms: int = 0):
    data = np.asarray(data, dtype=np.float32)
    start = offset_ms + round(start_sample * 1000 / 16000)
    end = offset_ms + round((start_sample + len(data)) * 1000 / 16000)
    return AudioChunk(session_id, chunk_id, start, end, 16000, 1, 'float32', data)
