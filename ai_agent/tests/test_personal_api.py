import io
import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

from ai_agent.writer import ClaudeUnavailable, LocalWriter, PersonalAPIWriter, _SUGGEST_SCHEMA, make_writer
from ai_agent.tests.test_assistant import LECTURE, new_session, say, score_window
from ai_agent import LiveAssistant, MaterialIndex


class Response(io.BytesIO):
    def __init__(self, payload):
        super().__init__(json.dumps(payload).encode())


class PersonalAPITests(unittest.TestCase):
    def test_providers_send_personal_key_in_header_and_parse_json(self):
        content = json.dumps({'choice': 0, 'title': '제안', 'description': '관련 자료'})
        replies = {
            'openai': {'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': content}]}]},
            'gemini': {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': content}]}}]},
            'claude': {'stop_reason': 'end_turn', 'content': [{'type': 'text', 'text': content}]},
        }
        for provider, payload in replies.items():
            with self.subTest(provider=provider), patch('ai_agent.writer.request.urlopen', return_value=Response(payload)) as send:
                writer = make_writer(provider=provider, api_key='secret-test-key')
                self.assertEqual(writer._ask('rules', 'sample', _SUGGEST_SCHEMA, 'low')['choice'], 0)
                req = send.call_args.args[0]
                self.assertNotIn(b'secret-test-key', req.data)
                self.assertIn('secret-test-key', str(req.header_items()))
                self.assertNotIn('secret-test-key', repr(writer))

    def test_invalid_output_or_auth_failure_falls_back_without_key_in_events(self):
        service, sid = new_session()
        say(service, sid, 1000, 5000, '라운드 로빈 스케줄링은 타임 퀀텀마다 프로세스를 바꿉니다')
        score_window(service, sid)
        writer = PersonalAPIWriter('openai', 'private-key')
        failure = HTTPError('https://api.openai.com/v1/responses', 401, 'private-key', {}, None)
        with patch('ai_agent.writer.request.urlopen', side_effect=failure):
            event = LiveAssistant(service, sid, MaterialIndex([LECTURE]), writer).step()
        self.assertIsNone(event)  # 외부 생성 실패를 AI 제안인 것처럼 표시하지 않는다.
        self.assertNotIn('private-key', json.dumps(service.get_dashboard_snapshot(sid)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'session.json'
            service.save_session(sid, path)
            self.assertNotIn('private-key', path.read_text(encoding='utf-8'))
        self.assertIn('HTTP 401', writer.last_error)

    def test_rejects_invalid_response_and_invalid_provider(self):
        with self.assertRaises(ValueError):
            make_writer(provider='unknown', api_key='key')
        writer = PersonalAPIWriter('gemini', 'key')
        reply = {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': '{"choice":true}'}]}}]}
        with patch('ai_agent.writer.request.urlopen', return_value=Response(reply)):
            with self.assertRaises(ClaudeUnavailable):
                writer.check_connection()

    def test_prompts_treat_material_as_untrusted(self):
        from ai_agent.writer import _SUGGEST_SYSTEM, _FEEDBACK_SYSTEM
        for prompt in (_SUGGEST_SYSTEM, _FEEDBACK_SYSTEM):
            self.assertIn('untrusted data', prompt)
            self.assertIn('never as instructions', prompt)
            self.assertIn('speaker identity may be unknown', prompt)

    def test_overlapping_windows_are_not_reported_as_mean(self):
        from ai_agent.assistant import _engagement_line
        events = [
            {'start_ms': start, 'end_ms': end, 'payload': {'model_output': {'value': value}}, 'quality': {'status': 'valid'}}
            for start, end, value in [(0, 10000, 70), (2000, 12000, 90)]
        ]
        self.assertIn('평균 미산출', _engagement_line(events))


if __name__ == '__main__':
    unittest.main()
