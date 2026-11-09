import tempfile
import unittest
from pathlib import Path

from ai_agent import LiveAssistant, MaterialIndex, generate_post_session
from ai_agent.tests.test_assistant import LECTURE, new_session, say, score_window
from integration import IntegrationService


class FakeExternal:
    uses_external_service = True
    model_id = 'fake/model'
    model_version = 'test'
    last_error = None
    calls = 0

    def suggest_teaching(self, texts, hits=(), references=()):
        self.calls += 1
        return {'title': '예시 제안', 'description': '방금 설명한 개념의 예시를 보여주세요.', 'search_query': '개념 예제'}


class ScoreTriggerTests(unittest.TestCase):
    def test_no_materials_generates_one_ai_suggestion_with_both_evidence(self):
        service, sid = new_session()
        transcript = say(service, sid, 1000, 5000, '프로세스의 상태 전이를 설명합니다')
        score = score_window(service, sid, 55)
        writer = FakeExternal()
        assistant = LiveAssistant(service, sid, MaterialIndex([]), writer, searcher=lambda query: [])
        event = assistant.step()
        self.assertEqual(event['event_type'], 'teaching_suggestion')
        self.assertEqual(event['payload']['evidence_event_ids'], [score['event_id'], transcript['event_id']])
        self.assertEqual(event['payload']['origin'], 'ai_generated')
        self.assertIsNone(assistant.step())
        self.assertEqual(writer.calls, 1)
        service.stop_session(sid)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session.json'
            service.save_session(sid, path)
            reader = IntegrationService()
            restored = reader.load_session(path)
            self.assertEqual(len(reader.get_dashboard_snapshot(restored)['teaching_suggestions']), 1)

    def test_no_score_high_score_unavailable_and_stale_do_not_trigger(self):
        for value, quality, now in [(80, 'valid', 10000), (None, 'not_observable', 10000), (50, 'valid', 30000)]:
            with self.subTest(value=value, quality=quality, now=now):
                service, sid = new_session()
                say(service, sid, 1000, 5000, '전사')
                score_window(service, sid, value, status=quality)
                service.now = now
                writer = FakeExternal()
                self.assertIsNone(LiveAssistant(service, sid, MaterialIndex([]), writer).step())
                self.assertEqual(writer.calls, 0)
        service, sid = new_session()
        say(service, sid, 1000, 5000, '전사')
        self.assertIsNone(LiveAssistant(service, sid, MaterialIndex([]), FakeExternal()).step())

    def test_no_transcript_and_cooldown_do_not_generate(self):
        service, sid = new_session()
        score_window(service, sid, 50)
        writer = FakeExternal()
        assistant = LiveAssistant(service, sid, MaterialIndex([]), writer, searcher=lambda query: [])
        self.assertIsNone(assistant.step())
        say(service, sid, 10000, 11000, '첫 전사')
        self.assertIsNotNone(assistant.step())
        score_window(service, sid, 40, start=2000, end=12000)
        self.assertIsNone(assistant.step())
        self.assertEqual(writer.calls, 1)

    def test_material_and_search_results_reach_ai_and_are_saved(self):
        service, sid = new_session()
        say(service, sid, 1000, 5000, '라운드 로빈 스케줄링은 타임 퀀텀마다 프로세스를 바꿉니다')
        score_window(service, sid)
        class Writer(FakeExternal):
            def suggest_teaching(self, texts, hits=(), references=()):
                self.observed = (hits, references)
                return super().suggest_teaching(texts, hits, references)
        writer = Writer()
        reference = {'title': '운영 체제', 'url': 'https://ko.wikipedia.org/wiki/example'}
        event = LiveAssistant(service, sid, MaterialIndex([LECTURE]), writer,
                              searcher=lambda q: [reference]).step()
        self.assertTrue(writer.observed[0])
        self.assertEqual(writer.observed[1], [reference])
        self.assertEqual(event['payload']['external_sources'], [reference])
        self.assertTrue(event['payload']['material_sources'])

    def test_post_session_without_materials_saves_timed_suggestion(self):
        service, sid = new_session()
        transcript = say(service, sid, 1000, 5000, '프로세스 상태 전이')
        score_window(service, sid)
        service.stop_session(sid)
        class Writer(FakeExternal):
            def feedback(self, sections, overall):
                return ['구간 피드백'] * len(sections), '전체 피드백'
        result = generate_post_session(service, sid, MaterialIndex([]), Writer(), searcher=lambda q: [])
        self.assertEqual(result['suggestions'], 1)
        event = service.get_dashboard_snapshot(sid)['teaching_suggestions'][0]
        self.assertEqual(event['start_ms'], transcript['start_ms'])
        self.assertIn(transcript['event_id'], event['payload']['evidence_event_ids'])

    def test_post_suggestions_require_low_score_note_or_explicit_range(self):
        service, sid = new_session()
        say(service, sid, 1000, 5000, '첫 번째 강의 구간')
        say(service, sid, 301000, 305000, '두 번째 강의 구간')
        say(service, sid, 601000, 605000, '세 번째 강의 구간')
        say(service, sid, 901000, 905000, '네 번째 강의 구간')
        low = score_window(service, sid, 55, start=300000, end=310000)
        note = service.make_event(sid, 'note_created', {'text': '다시 확인', 'linked_event_ids': []}, 602000, 602000, 'ui')
        service.receive_event(note)
        service.stop_session(sid)
        class Writer(FakeExternal):
            def feedback(self, sections, overall):
                return ['구간 피드백'] * len(sections), '전체 피드백'
        result = generate_post_session(service, sid, MaterialIndex([]), Writer(), searcher=lambda q: [],
                                       selected_range=(900000, 910000))
        self.assertEqual(result['suggestions'], 3)
        suggestions = service.get_dashboard_snapshot(sid)['teaching_suggestions']
        self.assertEqual({e['start_ms'] for e in suggestions}, {301000, 601000, 901000})
        self.assertIn(low['event_id'], next(e for e in suggestions if e['start_ms'] == 301000)['payload']['evidence_event_ids'])
        self.assertIn(note['event_id'], next(e for e in suggestions if e['start_ms'] == 601000)['payload']['evidence_event_ids'])


if __name__ == '__main__':
    unittest.main()
