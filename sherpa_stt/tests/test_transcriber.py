import unittest

import numpy as np

from integration import IntegrationService
from integration.contracts import validate_event
from sherpa_stt.audio import AudioChunk, to_mono_float32
from sherpa_stt.config import SttConfig
from sherpa_stt.transcriber import SherpaTranscriber


class FakeEngine:
    """decode 호출마다 미리 정한 (텍스트, 엔드포인트 여부)를 돌려준다."""

    def __init__(self, script):
        self.script = list(script)
        self.text = ""
        self.endpoint = False
        self.resets = 0
        self.finished = False
        self.received = []

    def create_stream(self):
        return object()

    def accept_waveform(self, stream, rate, samples):
        self.received.append((rate, samples))

    def decode(self, stream):
        if self.script:
            self.text, self.endpoint = self.script.pop(0)

    def get_text(self, stream):
        return self.text

    def is_endpoint(self, stream):
        return self.endpoint

    def reset(self, stream):
        self.resets += 1
        self.text, self.endpoint = "", False

    def input_finished(self, stream):
        self.finished = True


class ClockService(IntegrationService):
    """테스트에서 세션 시계를 직접 정한다."""

    now = 60_000

    def elapsed_ms(self, session_id):
        with self._lock:
            state = self._get(session_id)
            return self.now if state["running"] else state["session"]["duration_ms"]


def chunk(sid, chunk_id, start, end, n=1600):
    return AudioChunk(sid, chunk_id, start, end, 16000, 1, "float32", np.zeros(n, dtype=np.float32))


class TranscriberTests(unittest.TestCase):
    def setUp(self):
        self.service = ClockService()
        self.sid = self.service.start_session("stt test")

    def make(self, script, max_gap_ms=500):
        self.engine = FakeEngine(script)
        return SherpaTranscriber(self.engine, self.service, self.sid, "fake-model", "1", max_gap_ms)

    def test_partial_then_final_on_endpoint(self):
        t = self.make([("", False), ("안녕", False), ("안녕하세요", False), ("안녕하세요", True)])
        events = []
        for i in range(4):
            events += t.accept_chunk(chunk(self.sid, i, i * 100, (i + 1) * 100))
        kinds = [e["event_type"] for e in events]
        self.assertEqual(kinds, ["transcript_partial", "transcript_partial", "transcript_final"])
        final = events[-1]
        self.assertEqual(final["payload"]["text"], "안녕하세요")
        self.assertEqual(final["payload"]["revision"], 2)
        self.assertFalse(final["payload"]["is_edited"])
        self.assertEqual((final["start_ms"], final["end_ms"]), (0, 400))
        self.assertEqual(final["provenance"]["module"], "speech")
        self.assertEqual(final["provenance"]["model_id"], "fake-model")
        self.assertEqual(len({e["payload"]["utterance_id"] for e in events}), 1)
        for event in events:
            validate_event(event, self.sid)
        snapshot = self.service.get_dashboard_snapshot(self.sid)
        self.assertEqual([e["payload"]["text"] for e in snapshot["transcripts"]], ["안녕하세요"])
        self.assertEqual(snapshot["partial_transcripts"], [])

    def test_empty_segment_emits_nothing(self):
        t = self.make([("", False), ("", True)])
        events = t.accept_chunk(chunk(self.sid, 0, 0, 100)) + t.accept_chunk(chunk(self.sid, 1, 100, 200))
        self.assertEqual(events, [])
        self.assertEqual(self.engine.resets, 1)

    def test_next_utterance_gets_new_id_and_start(self):
        t = self.make([("하나", True), ("둘", True)])
        first = t.accept_chunk(chunk(self.sid, 0, 0, 100))
        second = t.accept_chunk(chunk(self.sid, 1, 100, 200))
        self.assertNotEqual(first[-1]["payload"]["utterance_id"], second[-1]["payload"]["utterance_id"])
        self.assertEqual(second[-1]["start_ms"], 100)

    def test_gap_finalizes_and_reports_audio_unavailable(self):
        t = self.make([("앞 발화", False), ("", False)])
        t.accept_chunk(chunk(self.sid, 0, 0, 100))
        events = t.accept_chunk(chunk(self.sid, 19, 2000, 2100))  # 청크 1~18을 캡처가 버림
        self.assertEqual(events[0]["event_type"], "transcript_final")
        self.assertEqual(events[0]["end_ms"], 100)
        status = events[1]
        self.assertEqual(status["event_type"], "analysis_status")
        self.assertEqual(status["payload"]["component"], "stt")
        self.assertEqual(status["quality"]["reason_codes"], ["audio_unavailable"])
        self.assertEqual((status["start_ms"], status["end_ms"]), (100, 2000))

    def test_late_callback_with_contiguous_ids_keeps_utterance(self):
        """콜백 지연으로 시각만 건너뛰면 발화를 끊거나 누락으로 보고하지 않는다(실측 회귀)."""
        t = self.make([("지하철에서 다리", False), ("지하철에서 다리를 벌리고", False), ("지하철에서 다리를 벌리고", True)])
        first = t.accept_chunk(chunk(self.sid, 0, 0, 100))
        events = t.accept_chunk(chunk(self.sid, 1, 1500, 1600))  # 번호는 이어짐, 시각만 1.4초 건너뜀
        events += t.accept_chunk(chunk(self.sid, 2, 1600, 1700))
        self.assertEqual(self.engine.resets, 1)  # 마지막 엔드포인트에서만 초기화
        kinds = [e["event_type"] for e in events]
        self.assertNotIn("analysis_status", kinds)
        finals = [e for e in events if e["event_type"] == "transcript_final"]
        self.assertEqual([e["payload"]["text"] for e in finals], ["지하철에서 다리를 벌리고"])
        self.assertEqual(finals[0]["payload"]["utterance_id"], first[0]["payload"]["utterance_id"])
        self.assertEqual(finals[0]["start_ms"], 0)

    def test_finish_flushes_last_utterance(self):
        t = self.make([("마지막", False)])
        t.accept_chunk(chunk(self.sid, 0, 0, 100))
        events = t.finish()
        self.assertTrue(self.engine.finished)
        self.assertEqual(events[-1]["event_type"], "transcript_final")
        self.assertEqual(t.finish(), [])
        with self.assertRaises(RuntimeError):
            t.accept_chunk(chunk(self.sid, 1, 100, 200))
        self.service.stop_session(self.sid)
        self.assertEqual(len(self.service.get_dashboard_snapshot(self.sid)["transcripts"]), 1)

    def test_missing_single_chunk_reports_discontinuity(self):
        t = self.make([('앞 발화', False), ('', False)])
        t.accept_chunk(chunk(self.sid, 0, 0, 100))
        events = t.accept_chunk(chunk(self.sid, 2, 200, 300))
        self.assertEqual([event['event_type'] for event in events], ['transcript_final', 'analysis_status'])
        self.assertEqual(events[1]['quality']['reason_codes'], ['audio_unavailable'])

    def test_rejects_wrong_session_and_non_increasing_ids(self):
        t = self.make([("", False)] * 3)
        with self.assertRaises(ValueError):
            t.accept_chunk(chunk("00000000-0000-4000-8000-000000000000", 0, 0, 100))
        t.accept_chunk(chunk(self.sid, 5, 0, 100))
        with self.assertRaises(ValueError):
            t.accept_chunk(chunk(self.sid, 5, 100, 200))

    def test_saved_session_has_no_partial_or_audio(self):
        t = self.make([("저장", False), ("저장 확인", True)])
        t.accept_chunk(chunk(self.sid, 0, 0, 100))
        t.accept_chunk(chunk(self.sid, 1, 100, 200))
        self.service.stop_session(self.sid)
        import json, tempfile, os
        with tempfile.TemporaryDirectory() as folder:
            path = self.service.save_session(self.sid, os.path.join(folder, "s.json"))
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        kinds = [e["event_type"] for e in data["events"]]
        self.assertNotIn("transcript_partial", kinds)
        self.assertEqual(kinds.count("transcript_final"), 1)


class SpanEngine(FakeEngine):
    """토큰 구간(스트림 기준 초)을 함께 주는 가짜 엔진."""

    def __init__(self, script):
        super().__init__([(text, endpoint) for text, endpoint, _ in script])
        self.spans = [span for _, _, span in script]
        self.span = None

    def decode(self, stream):
        if self.spans:
            self.span = self.spans.pop(0)
        super().decode(stream)

    def get_result(self, stream):
        return self.text, self.span


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.service = ClockService()
        self.sid = self.service.start_session("timeline test")

    def test_interval_uses_token_span_mapped_to_session_clock(self):
        # 청크 10개(각 100ms)가 세션 5000ms부터 들어오고, 토큰은 오디오 0.4~0.6초에 있다.
        script = [("", False, None)] * 6 + [("말", False, (0.4, 0.6))] + [("말", False, (0.4, 0.6))] * 2 + [("말", True, (0.4, 0.6))]
        engine = SpanEngine(script)
        t = SherpaTranscriber(engine, self.service, self.sid, "fake", "1")
        events = []
        for i in range(10):
            events += t.accept_chunk(chunk(self.sid, i, 5000 + i * 100, 5100 + i * 100))
        final = events[-1]
        self.assertEqual(final["event_type"], "transcript_final")
        self.assertEqual((final["start_ms"], final["end_ms"]), (5400 - 200, 5600 + 400))

    def test_interval_clamped_to_segment(self):
        engine = SpanEngine([("말", True, (0.0, 0.09))])
        t = SherpaTranscriber(engine, self.service, self.sid, "fake", "1")
        final = t.accept_chunk(chunk(self.sid, 0, 1000, 1100))[-1]
        self.assertEqual((final["start_ms"], final["end_ms"]), (1000, 1100))

    def test_chunk_clock_contiguous_and_resyncs(self):
        from sherpa_stt.audio import ChunkClock
        clock = ChunkClock(self.service, self.sid, 16000, resync_ms=200)
        self.service.now = 1000
        self.assertEqual(clock.stamp(1600), (900, 1000))
        self.service.now = 1130  # 도착 지터: 샘플 기준으로 이어 붙인다
        self.assertEqual(clock.stamp(1600), (1000, 1100))
        self.service.now = 2000  # 크게 밀리면 세션 시계로 다시 맞춘다
        self.assertEqual(clock.stamp(1600), (1900, 2000))
        self.assertEqual(clock.resync_count, 1)
        self.service.now = 2050  # 샘플 위치가 앞서면 끝이 현재 시각이 되도록 끌어내린다
        self.assertEqual(clock.stamp(1600), (1950, 2050))


class AudioTests(unittest.TestCase):
    def test_int16_stereo_to_mono_float32(self):
        samples = np.array([[32767, -32768], [0, 0]], dtype=np.int16)
        c = AudioChunk("s", 0, 0, 1, 16000, 2, "int16", samples)
        mono = to_mono_float32(c)
        self.assertEqual(mono.dtype, np.float32)
        self.assertEqual(mono.shape, (2,))
        self.assertAlmostEqual(float(mono[1]), 0.0)

    def test_channel_mismatch(self):
        c = AudioChunk("s", 0, 0, 1, 16000, 2, "float32", np.zeros(10, dtype=np.float32))
        with self.assertRaises(ValueError):
            to_mono_float32(c)

    def test_invalid_times(self):
        with self.assertRaises(ValueError):
            AudioChunk("s", 0, 10, 5, 16000, 1, "float32", np.zeros(1))


class ConfigTests(unittest.TestCase):
    def test_public_info_excludes_local_path(self):
        info = SttConfig(model_dir="C:/models/x").public_info()
        self.assertNotIn("model_dir", info)
        self.assertEqual(info["sample_rate_hz"], 16000)

    def test_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            SttConfig(model_dir="")
        with self.assertRaises(ValueError):
            SttConfig(model_dir="m", sample_rate_hz=0)
        with self.assertRaises(ValueError):
            SttConfig(model_dir="m", rule2_min_trailing_silence=float("nan"))


if __name__ == "__main__":
    unittest.main()
