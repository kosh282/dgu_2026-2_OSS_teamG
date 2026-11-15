import unittest

import numpy as np

from faster_whisper_stt.segmenter import PauseSegmenter

RATE = 16000
rng = np.random.default_rng(0)


def speech(seconds, level=0.1):
    t = np.arange(int(seconds * RATE)) / RATE
    return (np.sin(2 * np.pi * 220 * t) * level).astype(np.float32)


def silence(seconds):
    return rng.normal(0, 1e-4, int(seconds * RATE)).astype(np.float32)


def feed(segmenter, audio, start_ms=0):
    out = []
    step = RATE // 10
    for i in range(0, len(audio), step):
        part = audio[i:i + step]
        begin = start_ms + i * 1000 // RATE
        out += segmenter.push(part, begin, begin + len(part) * 1000 // RATE)
    return out


class PauseSegmenterTests(unittest.TestCase):
    def test_cuts_at_pause_after_minimum_length(self):
        seg = PauseSegmenter(min_seconds=3, max_seconds=12, pause_ms=500)
        audio = np.concatenate([silence(1), speech(4), silence(1), speech(2), silence(1)])
        pieces = feed(seg, audio) + seg.flush()
        self.assertEqual(len(pieces), 2)
        first, second = pieces
        # 앞 무음은 대부분 버리고, 첫 발화(1~5초)는 멈춤 지점 직후에서 끝난다.
        self.assertLessEqual(first.start_ms, 1000)
        self.assertGreaterEqual(first.start_ms, 300)
        self.assertTrue(5000 <= first.end_ms <= 5500)
        self.assertTrue(second.start_ms >= first.end_ms)
        self.assertTrue(second.end_ms >= 8000)

    def test_short_pause_before_minimum_does_not_cut(self):
        seg = PauseSegmenter(min_seconds=3, max_seconds=12, pause_ms=500)
        pieces = feed(seg, np.concatenate([speech(0.8), silence(0.8), speech(0.8), silence(0.4)]))
        self.assertEqual(pieces, [])
        rest = seg.flush()
        self.assertEqual(len(rest), 1)
        self.assertGreaterEqual(len(rest[0].audio), 2.7 * RATE)

    def test_long_speech_is_cut_at_quietest_point_before_maximum(self):
        seg = PauseSegmenter(min_seconds=3, max_seconds=10, pause_ms=500)
        audio = np.concatenate([speech(8.5), speech(0.2, level=0.01), speech(4)])
        pieces = feed(seg, audio)
        self.assertEqual(len(pieces), 1)
        self.assertTrue(8400 <= pieces[0].end_ms <= 8800)
        self.assertLessEqual(len(pieces[0].audio), 10 * RATE)

    def test_silence_only_is_never_sent(self):
        seg = PauseSegmenter()
        self.assertEqual(feed(seg, silence(30)) + seg.flush(), [])

    def test_audio_is_preserved_in_order(self):
        seg = PauseSegmenter(min_seconds=2, max_seconds=5, pause_ms=300, keep_silence_ms=200)
        audio = np.concatenate([speech(3), silence(0.5), speech(6), silence(0.5), speech(1)])
        pieces = feed(seg, audio) + seg.flush()
        joined = np.concatenate([p.audio for p in pieces])
        # 처음부터 말소리라 버린 무음이 없으므로 입력과 같아야 한다.
        np.testing.assert_array_equal(joined, audio)
        for before, after in zip(pieces, pieces[1:]):
            self.assertEqual(before.end_ms, after.start_ms)

    def test_rejects_invalid_lengths(self):
        with self.assertRaises(ValueError):
            PauseSegmenter(min_seconds=5, max_seconds=3)


if __name__ == '__main__':
    unittest.main()
