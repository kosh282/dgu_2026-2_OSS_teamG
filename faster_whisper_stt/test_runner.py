import queue
import unittest
from types import SimpleNamespace

import numpy as np

from faster_whisper_stt.runner import WhisperConfig, WhisperLiveTranscription
from integration import IntegrationService
from faster_whisper_stt.audio import AudioChunk


class FakeCapture:
    def __init__(self, chunks):
        self.chunks = queue.Queue()
        for chunk in chunks:
            self.chunks.put(chunk)
        self.error = None
        self.dropped_chunks = self.overflow_count = self.discontinuity_count = 0

    def start(self): pass
    def stop(self): pass


class FakeWhisper:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(audio.copy())
        return [SimpleNamespace(text=' 테스트')], None


class WhisperRunnerTests(unittest.TestCase):
    def test_invalid_window_is_rejected_before_processing_can_spin(self):
        for value in (0, -1, True, 0.5, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                WhisperConfig(window_seconds=value)

    def test_invalid_capture_or_model_settings_fail_before_loading(self):
        for settings in ({'sample_rate_hz': 8000}, {'chunk_ms': 0},
                         {'num_threads': 0}, {'model': ''}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                WhisperConfig(**settings)

    SHORT = WhisperConfig(min_segment_seconds=3, window_seconds=10)

    def test_default_segments_are_long_enough_for_whisper(self):
        config = WhisperConfig()
        self.assertEqual((config.min_segment_seconds, config.window_seconds, config.num_threads), (8, 20, 2))

    def run_chunks(self, audio, model, config=None):
        service = IntegrationService()
        sid = service.start_session('test', {'config_version': 'test', 'participation_model': None, 'stt_model': None})
        service.elapsed_ms = lambda _: 60000
        chunks = [AudioChunk(sid, i, i * 100, (i + 1) * 100, 16000, 1, 'float32', audio[i*1600:(i+1)*1600])
                  for i in range(len(audio) // 1600)]
        runner = WhisperLiveTranscription(service, sid, config or WhisperConfig(), source='microphone',
                                          capture=FakeCapture(chunks), engine=model)
        self.assertTrue(runner.start())
        runner.stop()
        self.assertIsNone(runner.last_error)
        return service.get_dashboard_snapshot(sid), runner.stats()

    def test_segments_end_at_pauses_and_shutdown_tail_is_processed(self):
        t = np.arange(16000 * 4) / 16000
        tone = (np.sin(2 * np.pi * 220 * t) * 0.1).astype(np.float32)
        pause = np.zeros(16000, np.float32)
        audio = np.concatenate([tone, pause, tone[:16000 * 2]])  # 4초 발화, 1초 멈춤, 2초 발화(종료 시 남음)
        model = FakeWhisper()
        snapshot, stats = self.run_chunks(audio, model, self.SHORT)
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(4.0 <= len(model.calls[0]) / 16000 <= 4.5)
        events = snapshot['transcripts']
        self.assertEqual([e['payload']['text'] for e in events], ['테스트', '테스트'])
        self.assertTrue(4000 <= events[0]['end_ms'] <= 4500)
        self.assertEqual(events[1]['end_ms'], 7000)
        self.assertTrue(all(e['provenance']['model_id'] == 'faster-whisper-base' for e in events))
        self.assertEqual(snapshot['partial_transcripts'], [])
        self.assertEqual((stats['processed_windows'], stats['empty_windows']), (2, 0))
        self.assertIsNotNone(stats['capture_level_dbfs'])

    def test_continuous_speech_is_split_before_maximum_length(self):
        data = np.linspace(-0.5, 0.5, 168000, dtype=np.float32)
        model = FakeWhisper()
        self.run_chunks(data, model, self.SHORT)
        self.assertTrue(all(len(x) <= 160000 for x in model.calls))
        np.testing.assert_array_equal(np.concatenate(model.calls), data)

    def test_silent_input_is_not_sent_to_model(self):
        model = FakeWhisper()
        snapshot, stats = self.run_chunks(np.zeros(160000, np.float32), model)
        self.assertEqual(model.calls, [])
        self.assertEqual((stats['processed_windows'], stats['empty_windows']), (0, 0))
        self.assertEqual(stats['capture_level_dbfs'], -200.0)
        self.assertEqual(snapshot['transcripts'], [])

    def test_spoken_phrases_are_kept_and_unreliable_segments_are_dropped(self):
        class Model:
            def transcribe(self, audio, **kwargs):
                return [SimpleNamespace(text=' 인터페이스는 추상 메서드를 가집니다.'),
                        SimpleNamespace(text=' 시청해 주셔서 감사합니다!'),
                        SimpleNamespace(text=' 가나다라마바사가나다라마바사', compression_ratio=3.1),
                        SimpleNamespace(text=' OnClick, OnClick, OnClick, OnClick', compression_ratio=2.8),
                        SimpleNamespace(text=' 음', no_speech_prob=0.9, avg_logprob=-1.4)], None
        t = np.arange(16000 * 4) / 16000
        audio = np.concatenate([(np.sin(2 * np.pi * 220 * t) * 0.1).astype(np.float32), np.zeros(16000, np.float32)])
        snapshot, stats = self.run_chunks(audio, Model())
        # 문장 내용은 손대지 않고 모델의 확률·압축률 기준으로 불안정한 결과만 버린다.
        self.assertEqual([e['payload']['text'] for e in snapshot['transcripts']],
                         ['인터페이스는 추상 메서드를 가집니다. 시청해 주셔서 감사합니다!'])

    def test_real_repetition_is_kept_and_hint_copy_is_dropped(self):
        from faster_whisper_stt.runner import segment_text
        # 실제로 반복한 말은 고치지 않고 그대로 둔다.
        for text in ('네 네 네', '빨리 빨리 빨리 정리합시다', 'OnClick, OnClick, OnClick'):
            self.assertEqual(segment_text(SimpleNamespace(text=f' {text}', compression_ratio=1.3)), text)
        hints = ('안드로이드', '액티비티', '추상 클래스', '인터페이스', '상속')
        copied = SimpleNamespace(text='안드로이드, 액티비티, 추상 클래스, 인터페이스')
        self.assertEqual(segment_text(copied, hints), '')
        self.assertEqual(segment_text(copied), '안드로이드, 액티비티, 추상 클래스, 인터페이스')
        self.assertEqual(segment_text(SimpleNamespace(text='인터페이스를 구현합니다'), hints), '인터페이스를 구현합니다')

    def test_decoding_is_bounded(self):
        options = WhisperConfig().transcribe_options('', None, 20.0)
        self.assertFalse(options['condition_on_previous_text'])
        self.assertEqual(options['temperature'], [0.0])
        self.assertEqual(options['max_new_tokens'], 130)
        self.assertNotIn('no_repeat_ngram_size', options)
        self.assertEqual(WhisperConfig(no_repeat_ngram_size=3).transcribe_options()['no_repeat_ngram_size'], 3)
        for bad in ({'temperatures': ()}, {'temperatures': (1.5,)}, {'no_repeat_ngram_size': -1},
                    {'max_tokens_per_second': 0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                WhisperConfig(**bad)

    def test_empty_model_output_is_counted(self):
        class EmptyModel:
            def transcribe(self, audio, **kwargs):
                return [], None
        t = np.arange(16000 * 4) / 16000
        audio = np.concatenate([(np.sin(2 * np.pi * 220 * t) * 0.1).astype(np.float32), np.zeros(16000, np.float32)])
        snapshot, stats = self.run_chunks(audio, EmptyModel())
        self.assertEqual((stats['processed_windows'], stats['empty_windows']), (1, 1))
        self.assertEqual(snapshot['transcripts'], [])

    def test_vad_and_beam_options_are_passed(self):
        model = FakeWhisper()
        captured = {}
        original = model.transcribe
        def transcribe(audio, **kwargs):
            captured.update(kwargs)
            return original(audio, **kwargs)
        model.transcribe = transcribe
        t = np.arange(16000 * 4) / 16000
        audio = np.concatenate([(np.sin(2 * np.pi * 220 * t) * 0.1).astype(np.float32), np.zeros(16000, np.float32)])
        self.run_chunks(audio, model, WhisperConfig(beam_size=3))
        self.assertEqual((captured['language'], captured['beam_size'], captured['vad_filter']), ('ko', 3, True))
        with self.assertRaises(ValueError):
            WhisperConfig(min_segment_seconds=20, window_seconds=10)

    def test_lecture_terms_are_passed_as_hotwords(self):
        seen = []
        class Model:
            def transcribe(self, audio, **kwargs):
                seen.append(kwargs.get('hotwords'))
                return [SimpleNamespace(text=' 인터페이스')], None
        t = np.arange(16000 * 4) / 16000
        audio = np.concatenate([(np.sin(2 * np.pi * 220 * t) * 0.1).astype(np.float32), np.zeros(16000, np.float32)])
        service = IntegrationService()
        sid = service.start_session('test')
        chunks = [AudioChunk(sid, i, i * 100, (i + 1) * 100, 16000, 1, 'float32', audio[i*1600:(i+1)*1600])
                  for i in range(len(audio) // 1600)]
        terms = ['인터페이스', '추상', '인터페이스', ' ', 'X' * 200]
        runner = WhisperLiveTranscription(service, sid, WhisperConfig(), source='microphone',
                                          capture=FakeCapture(chunks), engine=Model(), hint_terms=terms)
        self.assertEqual(runner.hint_terms, ('인터페이스', '추상'))  # 중복·빈 값 제거, 150자 제한
        runner.start()
        runner.stop()
        self.assertEqual(seen, ['인터페이스, 추상'])
        self.assertEqual(runner.stats()['hint_term_count'], 2)
        self.assertNotIn('hotwords', WhisperConfig().transcribe_options('', None))

    def test_previous_text_is_passed_as_prompt_to_next_segment(self):
        prompts = []
        class Model:
            def transcribe(self, audio, **kwargs):
                prompts.append(kwargs.get('initial_prompt'))
                return [SimpleNamespace(text=f' 문장{len(prompts)}')], None
        t = np.arange(16000 * 4) / 16000
        tone = (np.sin(2 * np.pi * 220 * t) * 0.1).astype(np.float32)
        pause = np.zeros(16000, np.float32)
        self.run_chunks(np.concatenate([tone, pause, tone, pause]), Model(), self.SHORT)
        self.assertEqual(prompts, [None, '문장1'])
        self.run_chunks(np.concatenate([tone, pause, tone, pause]), Model(),
                        WhisperConfig(min_segment_seconds=3, window_seconds=10, prompt_previous_text=False))
        self.assertEqual(prompts[2:], [None, None])


if __name__ == '__main__':
    unittest.main()
