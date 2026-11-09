"""강의 조교 단위 테스트. 외부 API를 호출하지 않는다(Claude는 가짜 클라이언트)."""
import io
import json
from types import SimpleNamespace
import unittest
from uuid import uuid4

from integration import IntegrationService
from integration.contracts import validate_event
from ai_agent import (ClaudeWriter, LiveAssistant, LocalWriter, MaterialError, MaterialIndex,
                      generate_post_session, load_material)
from ai_agent.materials import Material, MaterialPage


class ClockService(IntegrationService):
    """테스트에서 세션 시계를 직접 정한다."""

    now = 0

    def elapsed_ms(self, session_id):
        with self._lock:
            state = self._get(session_id)
            return self.now if state["running"] else state["session"]["duration_ms"]


def material(name, *texts):
    document_id = str(uuid4())
    return Material(document_id, name, tuple(MaterialPage(document_id, name, i, t) for i, t in enumerate(texts, 1)))


LECTURE = material(
    "운영체제 8주차.pdf",
    "프로세스와 스레드\n프로세스는 실행 중인 프로그램이며 스레드는 실행 흐름 단위다",
    "CPU 스케줄링\n라운드 로빈 스케줄링은 타임 퀀텀마다 프로세스를 교대로 실행한다. 선점형 스케줄링",
    "교착 상태\n교착 상태의 네 가지 조건은 상호 배제, 점유 대기, 비선점, 순환 대기",
)


def say(service, sid, start, end, text):
    service.now = max(service.now, end)
    event = service.make_event(sid, "transcript_final",
                               {"utterance_id": str(uuid4()), "revision": 1, "text": text, "is_edited": False},
                               start, end, "speech")
    service.receive_event(event)
    return event


def new_session(title="운영체제"):
    service = ClockService()
    sid = service.start_session(title)
    return service, sid


def score_window(service, sid, value=50, *, start=0, end=10000, status='valid'):
    service.now = max(service.now, end)
    running = service.make_event(sid, 'analysis_status', {'component': 'model', 'state': 'running', 'message': 'test'}, 0, 0, 'model')
    service.receive_event(running)
    event = service.make_event(sid, 'engagement_window', {
        'model_output': None if value is None else {'value': value},
        'observation': {'valid_count': None, 'eligible_count': None, 'observable_ratio': None, 'count_unit': None}, 'signals': {},
    }, start, end, 'model', quality_status=status, model_id='test', model_version='1')
    service.receive_event(event)
    return event


class RetrievalTests(unittest.TestCase):
    def test_finds_page_despite_particles(self):
        index = MaterialIndex([LECTURE])
        hits = index.search("라운드 로빈 스케줄링에서는 타임 퀀텀이 끝나면 다음 프로세스로 넘어갑니다")
        self.assertTrue(hits)
        self.assertEqual(hits[0].page.page, 2)  # '스케줄링에서는'처럼 조사가 붙어도 2-gram으로 일치
        self.assertTrue(any("스케줄링" in k or "퀀텀" in k for k in hits[0].keywords))

    def test_unrelated_speech_returns_nothing(self):
        self.assertEqual(MaterialIndex([LECTURE]).search("오늘 점심 메뉴는 김치찌개였습니다"), [])

    def test_empty_index(self):
        index = MaterialIndex([])
        self.assertFalse(index)
        self.assertEqual(index.search("스케줄링"), [])


class MaterialTests(unittest.TestCase):
    def test_pptx_slides_and_notes(self):
        from pptx import Presentation
        from pptx.util import Inches
        deck = Presentation()
        for title, note in [("교착 상태 조건", "순환 대기 설명"), ("페이지 교체", "LRU 예시")]:
            slide = deck.slides.add_slide(deck.slide_layouts[5])
            slide.shapes.title.text = title
            slide.shapes.add_textbox(Inches(1), Inches(2), Inches(4), Inches(1)).text_frame.text = "본문 " + title
            slide.notes_slide.notes_text_frame.text = note
        buffer = io.BytesIO()
        deck.save(buffer)
        data = buffer.getvalue()
        loaded = load_material("8주차.pptx", data)
        self.assertEqual([p.page for p in loaded.pages], [1, 2])
        self.assertIn("순환 대기 설명", loaded.pages[0].text)
        self.assertIn("본문 페이지 교체", loaded.pages[1].text)
        # 같은 파일은 같은 자료 ID(수업 중·사후 검토 연결)
        self.assertEqual(loaded.document_id, load_material("다른이름.pptx", data).document_id)

    def test_pdf_text(self):
        content = b"BT /F1 12 Tf 72 720 Td (Round robin scheduling) Tj ET"
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
        pdf = io.BytesIO()
        pdf.write(b"%PDF-1.4\n")
        offsets = []
        for number, body in enumerate(objects, 1):
            offsets.append(pdf.tell())
            pdf.write(b"%d 0 obj\n" % number + body + b"\nendobj\n")
        xref = pdf.tell()
        pdf.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1))
        for offset in offsets:
            pdf.write(b"%010d 00000 n \n" % offset)
        pdf.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref))
        loaded = load_material("slides.pdf", pdf.getvalue())
        self.assertEqual(len(loaded.pages), 1)
        self.assertIn("Round robin", loaded.pages[0].text)

    def test_rejects_unsupported(self):
        with self.assertRaises(MaterialError):
            load_material("old.ppt", b"x")
        with self.assertRaises(MaterialError):
            load_material("broken.pdf", b"not a pdf")


class LiveAssistantTests(unittest.TestCase):
    def test_suggestion_links_evidence_and_source(self):
        service, sid = new_session()
        assistant = LiveAssistant(service, sid, MaterialIndex([LECTURE]))
        self.assertIsNone(assistant.step())  # 전사 없음
        say(service, sid, 1_000, 4_000, "오늘은 점심 이야기부터 하겠습니다")
        evidence = say(service, sid, 5_000, 9_000, "라운드 로빈 스케줄링은 타임 퀀텀마다 프로세스를 바꿉니다")
        score = score_window(service, sid)
        event = assistant.step()
        self.assertIsNotNone(event)
        validate_event(event, sid)
        payload = event["payload"]
        self.assertEqual(payload["source"]["page"], 2)
        self.assertEqual(payload["source"]["document_id"], LECTURE.document_id)
        self.assertEqual(payload["status"], "suggested")
        self.assertEqual(payload["evidence_event_ids"], [evidence["event_id"], score["event_id"]])
        self.assertEqual((event["start_ms"], event["end_ms"]), (5_000, 9_000))
        self.assertEqual(event["provenance"]["module"], "assistant")
        self.assertEqual(event["provenance"]["model_id"], LocalWriter.model_id)
        self.assertEqual(len(service.get_dashboard_snapshot(sid)["suggestions"]), 1)
        # 새 전사가 없으면 다시 제안하지 않고, 같은 페이지는 쿨다운 동안 반복하지 않는다.
        self.assertIsNone(assistant.step())
        say(service, sid, 10_000, 14_000, "라운드 로빈 스케줄링에서 타임 퀀텀이 짧으면 문맥 교환이 늘어납니다")
        self.assertIsNone(assistant.step())

    def test_without_materials_waits_for_events(self):
        service, sid = new_session()
        assistant = LiveAssistant(service, sid, MaterialIndex([]))
        self.assertTrue(assistant.start())
        self.assertEqual(service.get_dashboard_snapshot(sid)["status"]["assistant"], "running")
        assistant.stop()

    def test_thread_stops_with_session(self):
        service, sid = new_session()
        assistant = LiveAssistant(service, sid, MaterialIndex([LECTURE]), interval_s=0.01)
        self.assertTrue(assistant.start())
        self.assertEqual(service.get_dashboard_snapshot(sid)["status"]["assistant"], "running")
        service.stop_session(sid)
        assistant._thread.join(2)
        self.assertFalse(assistant.is_running())
        assistant.stop()


class PostSessionTests(unittest.TestCase):
    def test_feedback_and_timeline_suggestions(self):
        service, sid = new_session()
        say(service, sid, 10_000, 15_000, "프로세스는 실행 중인 프로그램이고 스레드는 실행 흐름 단위입니다")
        say(service, sid, 310_000, 316_000, "교착 상태의 조건은 상호 배제 점유 대기 비선점 순환 대기입니다")
        note = service.make_event(sid, "note_created", {"text": "예시 더 들기", "linked_event_ids": []}, 312_000, 312_000, "ui")
        service.now = 320_000
        service.receive_event(note)
        window = service.make_event(
            sid, "engagement_window",
            {"model_output": None, "observation": {"valid_count": None, "eligible_count": None,
                                                    "observable_ratio": None, "count_unit": None}, "signals": {}},
            300_000, 310_000, "model", quality_status="not_observable", reason_codes=["face_not_found"],
            model_id="m", model_version="1")
        service.receive_event(window)
        service.stop_session(sid)

        result = generate_post_session(service, sid, MaterialIndex([LECTURE]))
        self.assertIsNone(result["skipped"])
        self.assertEqual(result["suggestions"], 1)  # 메모가 있는 두 번째 구간만 제안
        self.assertEqual(result["feedback"], 3)  # 구간 2개 + 수업 전체
        snapshot = service.get_dashboard_snapshot(sid)
        for event in snapshot["feedback"] + snapshot["suggestions"]:
            validate_event(event, sid)
            self.assertIsNone(event["created_ms"])  # 종료 후 생성
            self.assertIsNotNone(event["created_at_utc"])
        pages = sorted(e["payload"]["source"]["page"] for e in snapshot["suggestions"])
        self.assertEqual(pages, [3])
        second = next(e for e in snapshot["feedback"] if e["start_ms"] == 300_000)
        self.assertIn("예시 더 들기", second["payload"]["text"])
        self.assertIn("관측 불가", second["payload"]["text"])
        self.assertNotIn("0점", second["payload"]["text"])
        overall = next(e for e in snapshot["feedback"] if e["start_ms"] == 0 and e["end_ms"] == snapshot["session"]["duration_ms"])
        self.assertIn("2쪽", overall["payload"]["text"])  # 다루지 않은 자료 페이지
        # 다시 실행해도 중복 생성하지 않는다. 저장·재열기 뒤에도 유지된다.
        self.assertIsNotNone(generate_post_session(service, sid, MaterialIndex([LECTURE]))["skipped"])
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            path = service.save_session(sid, Path(folder) / "s.json")
            reopened = IntegrationService()
            rsid = reopened.load_session(path)
            again = reopened.get_dashboard_snapshot(rsid)
        self.assertEqual(len(again["feedback"]), 3)
        self.assertEqual(len(again["suggestions"]), 1)

    def test_skips_existing_live_suggestion_section(self):
        service, sid = new_session()
        say(service, sid, 1_000, 5_000, "라운드 로빈 스케줄링은 타임 퀀텀마다 프로세스를 바꿉니다")
        score_window(service, sid)
        LiveAssistant(service, sid, MaterialIndex([LECTURE])).step()
        service.stop_session(sid)
        result = generate_post_session(service, sid, MaterialIndex([LECTURE]))
        self.assertEqual(result["suggestions"], 0)
        feedback = service.get_dashboard_snapshot(sid)["feedback"]
        section = next(e for e in feedback if e["payload"]["evidence_event_ids"])  # 전체 피드백은 근거 없음
        suggestion_id = service.get_dashboard_snapshot(sid)["suggestions"][0]["event_id"]
        self.assertIn(suggestion_id, section["payload"]["evidence_event_ids"])

    def test_no_transcripts(self):
        service, sid = new_session()
        service.stop_session(sid)
        self.assertIsNotNone(generate_post_session(service, sid, MaterialIndex([LECTURE]))["skipped"])


class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text=json.dumps(reply, ensure_ascii=False))])


def fake_client(*replies):
    messages = FakeMessages(replies)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


class ClaudeWriterTests(unittest.TestCase):
    def test_suggestion_uses_claude_choice(self):
        client, messages = fake_client({"choice": 0, "title": "라운드 로빈 슬라이드", "description": "타임 퀀텀 설명과 연결됩니다."})
        service, sid = new_session()
        say(service, sid, 1_000, 5_000, "라운드 로빈 스케줄링은 타임 퀀텀마다 프로세스를 바꿉니다")
        score_window(service, sid)
        event = LiveAssistant(service, sid, MaterialIndex([LECTURE]), ClaudeWriter(client=client)).step()
        self.assertEqual(event["payload"]["title"], "라운드 로빈 슬라이드")
        self.assertEqual(event["provenance"]["model_id"], "claude-opus-5-5")
        call = messages.calls[0]
        self.assertEqual(call["model"], "claude-opus-5-5")
        self.assertEqual(call["output_config"]["format"]["type"], "json_schema")
        # 전송 내용은 확정 전사와 후보 자료 페이지뿐이다.
        self.assertIn("라운드 로빈 스케줄링은", call["messages"][0]["content"])

    def test_claude_none_choice_creates_nothing(self):
        client, _ = fake_client({"choice": -1, "title": "", "description": ""})
        service, sid = new_session()
        say(service, sid, 1_000, 5_000, "라운드 로빈 스케줄링은 타임 퀀텀마다 프로세스를 바꿉니다")
        score_window(service, sid)
        self.assertIsNone(LiveAssistant(service, sid, MaterialIndex([LECTURE]), ClaudeWriter(client=client)).step())

    def test_failure_falls_back_to_local(self):
        import anthropic
        client, _ = fake_client(anthropic.AnthropicError("no credentials"))
        service, sid = new_session()
        say(service, sid, 1_000, 5_000, "라운드 로빈 스케줄링은 타임 퀀텀마다 프로세스를 바꿉니다")
        score_window(service, sid)
        writer = ClaudeWriter(client=client)
        event = LiveAssistant(service, sid, MaterialIndex([LECTURE]), writer).step()
        self.assertEqual(event["provenance"]["model_id"], LocalWriter.model_id)
        self.assertIn("no credentials", writer.last_error)
        self.assertEqual(service.get_dashboard_snapshot(sid)["status"]["assistant"], "running")

    def test_feedback_from_claude(self):
        client, messages = fake_client({"sections": [{"section_index": 0, "feedback": "첫 구간 피드백"}], "overall": "전체 피드백"})
        service, sid = new_session()
        say(service, sid, 1_000, 5_000, "프로세스는 실행 중인 프로그램입니다")
        service.stop_session(sid)
        result = generate_post_session(service, sid, MaterialIndex([]), ClaudeWriter(client=client))
        self.assertIsNone(result["writer_error"])
        texts = sorted(e["payload"]["text"] for e in service.get_dashboard_snapshot(sid)["feedback"])
        self.assertEqual(texts, ["전체 피드백", "첫 구간 피드백"])
        self.assertIn("not observable", messages.calls[0]["system"])  # 관측 불가를 낮은 참여로 해석하지 않게 지시


if __name__ == "__main__":
    unittest.main()
