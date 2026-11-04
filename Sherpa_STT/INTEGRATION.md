# Team v0.1.0 integration

A/V team contract: `AudioChunk` in memory -> `STTProcessor` -> partial/final event callback -> UI/store.

```python
from stt_core import Session, STTProcessor, AudioChunk, create_recognizer

# Shared by team coordinator at live session start:
session = Session(
    title="수업 1", session_id=shared_session_id, clock_origin=shared_monotonic_start
)
recognizer = create_recognizer("models/sherpa-onnx-streaming-zipformer-korean-2024-06-16")
stt = STTProcessor(session, recognizer, on_event=event_queue.put)

# For each capture chunk (already 16000 Hz / 1 channel / float32):
# stt.accept(AudioChunk(...))
# stt.finish() on normal stop
```

- `start_ms/end_ms`: `[start, end)` whole-session millisecond time. For live capture, sample timestamp relative to shared `monotonic` start is preferred; do not independently start new clocks in vision and audio.
- `created_ms`: when event was created, based on session clock. Display time should be measured separately in the UI.
- `transcript_partial`: replace existing text with same `utterance_id` and larger `revision`; do not store in session JSON.
- `transcript_final`: store one per utterance, use `event_id` as source pointer for other events.
- `/output/session.json`: this stand-alone CLI writes a demo session; the integrated app's coordinator should own the single shared session and persistence.
- Media file mode: sample-index-based timeline begins at 0 or at `--timeline-offset-ms`. For media with nonzero audio PTS or independent camera timestamps, determine A/V offset using `ffprobe`/an anchor event. This starter code does not infer accurate media timestamp mapping.
- Endpoint boundaries are approximate and include trailing silence. They are not forced-alignment timestamps.
- Raw audio and video frames never go into the stored JSON. The optional `--events-jsonl` is debug-only and may contain personal data; disable it for real classes.
