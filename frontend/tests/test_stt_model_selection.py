import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from frontend.live_session import LiveSession


class ModelSelectionTests(unittest.TestCase):
    def test_both_runners_use_system_loopback_even_with_video_window(self):
        for model in ('sherpa', 'whisper-base'):
            with self.subTest(model=model):
                started = threading.Event()
                arguments = []

                class FakeRunner:
                    last_error = None
                    def __init__(self, service, sid, config, **kwargs): arguments.append(kwargs)
                    def start(self): started.set(); return True
                    def is_alive(self): return True
                    def stop(self): pass

                with patch('frontend.live_session._load_stt_engine', return_value=object()), \
                     patch('faster_whisper_stt.runner.WhisperLiveTranscription', FakeRunner), \
                     patch('sherpa_stt.runner.LiveTranscription', FakeRunner), \
                     patch.object(LiveSession, '_supervise_vision', lambda self: self.stop_event.wait()):
                    live = LiveSession(123, audio_source='loopback', video_enabled=True, stt_model=model)
                    live.start('전체 출력 테스트')
                    try:
                        self.assertTrue(started.wait(2))
                        self.assertEqual(arguments[0]['source'], 'loopback')
                        self.assertIsNone(arguments[0].get('target_process_id'))
                    finally:
                        live.stop()

    def test_audio_only_loopback_does_not_require_window(self):
        from streamlit.testing.v1 import AppTest
        root = Path(__file__).resolve().parents[2]
        app = AppTest.from_file(str(root / 'frontend' / 'app.py'), default_timeout=20).run()
        next(c for c in app.checkbox if c.label == '영상 분석 사용').uncheck().run()
        next(c for c in app.checkbox if c.label == '실시간 음성 전사 사용').check().run()
        next(t for t in app.text_input if t.label == '수업 제목').set_value('오디오 테스트').run()
        next(c for c in app.checkbox if c.label.startswith('수업 영상·음성')).check().run()
        next(c for c in app.checkbox if c.label.startswith('전사·메모')).check().run()
        self.assertFalse(next(b for b in app.button if b.label == '세션 시작').disabled)

    def test_preparation_offers_each_model(self):
        from streamlit.testing.v1 import AppTest
        root = Path(__file__).resolve().parents[2]
        app = AppTest.from_file(str(root / 'frontend' / 'app.py'), default_timeout=20).run()
        next(c for c in app.checkbox if c.label == '실시간 음성 전사 사용').check().run()
        select = next(s for s in app.selectbox if s.label == '음성 전사 모델')
        self.assertEqual(select.options, ['sherpa 한국어 스트리밍 (빠름, 강의 녹화 음성에 약함)',
                                          'faster-whisper base (말 멈춤 단위)',
                                          'faster-whisper small (말 멈춤 단위, 권장)'])
        import importlib.util
        expected = 'whisper-small' if importlib.util.find_spec('faster_whisper') else 'sherpa'
        self.assertEqual(select.value, expected)
        select.select('whisper-base').run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['stt_model'], 'whisper-base')

    def test_only_selected_runner_and_model_are_started(self):
        started = threading.Event()
        captured = {}

        class FakeRunner:
            last_error = None
            def __init__(self, service, sid, config, **kwargs):
                self.config = config
                captured.update(kwargs)
            def start(self): started.set(); return True
            def is_alive(self): return True
            def stop(self): pass

        with patch('frontend.live_session._load_stt_engine', return_value=object()) as load, \
             patch('faster_whisper_stt.runner.WhisperLiveTranscription', FakeRunner), \
             patch('sherpa_stt.runner.LiveTranscription') as sherpa:
            live = LiveSession(None, audio_source='microphone', video_enabled=False, stt_model='whisper-small',
                               stt_hint_terms=['인터페이스', '추상'])
            live.start('선택 테스트')
            try:
                self.assertTrue(started.wait(2))
                self.assertEqual(load.call_args.args[0].model, 'small')
                self.assertEqual(captured['hint_terms'], ('인터페이스', '추상'))
                sherpa.assert_not_called()
                metadata = live.service._get(live.sid)['config']['stt_model']
                self.assertEqual(metadata['model_id'], 'faster-whisper-small')
            finally:
                live.stop()


if __name__ == '__main__':
    unittest.main()
