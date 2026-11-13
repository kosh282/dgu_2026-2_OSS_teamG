"""자동 음량 조절(AutoGain) 단위 테스트. 실제 모델 없이 신호 크기만 확인한다."""
import unittest

import numpy as np

from sherpa_stt.audio import AudioChunk, AutoGain
from sherpa_stt.config import SttConfig
from sherpa_stt.tests.test_transcriber import ClockService, FakeEngine
from sherpa_stt.transcriber import SherpaTranscriber

RATE = 16000
rng = np.random.default_rng(1)


def dbfs(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


def noise(level_dbfs, n=1600):
    return rng.normal(0, 10 ** (level_dbfs / 20), n).astype(np.float32)


def tone(level_dbfs, n=1600):
    """RMS가 level_dbfs인 말소리 대용 신호."""
    t = np.arange(n) / RATE
    return (np.sin(2 * np.pi * 220 * t) * np.sqrt(2) * 10 ** (level_dbfs / 20)).astype(np.float32)


class AutoGainTests(unittest.TestCase):
    def run_signal(self, gain, chunks):
        return [gain.process(c) for c in chunks]

    def test_quiet_speech_is_raised_toward_target(self):
        gain = AutoGain(target_dbfs=-20, max_gain_db=30)
        chunks = [noise(-70)] * 10 + [tone(-45) + noise(-70) for _ in range(20)]
        out = self.run_signal(gain, chunks)
        # 첫 말소리 청크부터 바로 키운다(첫 단어 보존).
        self.assertGreater(dbfs(out[10]), -30)
        self.assertAlmostEqual(dbfs(out[-1]), -20, delta=1.5)
        self.assertAlmostEqual(gain.input_level_dbfs, -45, delta=1.0)

    def test_gain_is_capped(self):
        gain = AutoGain(target_dbfs=-20, max_gain_db=30)
        out = self.run_signal(gain, [noise(-90)] * 5 + [tone(-65) for _ in range(20)])
        self.assertLessEqual(gain.gain_db, 30.0)
        self.assertAlmostEqual(dbfs(out[-1]), -35, delta=1.5)

    def test_loud_speech_is_not_changed(self):
        gain = AutoGain(target_dbfs=-20)
        chunks = [noise(-70)] * 5 + [tone(-12) for _ in range(10)]
        out = self.run_signal(gain, chunks)
        self.assertEqual(gain.gain_db, 0.0)
        np.testing.assert_array_equal(out[-1], chunks[-1])

    def test_silence_only_keeps_original(self):
        gain = AutoGain()
        chunks = [noise(-70) for _ in range(30)]
        out = self.run_signal(gain, chunks)
        self.assertEqual(gain.gain_db, 0.0)
        self.assertIsNone(gain.input_level_dbfs)
        np.testing.assert_array_equal(out[-1], chunks[-1])

    def test_no_clipping_when_loud_follows_quiet(self):
        """조용한 첫머리 직후 큰 소리가 와도 찌그러지지 않는다(실측 회귀)."""
        gain = AutoGain(target_dbfs=-20, max_gain_db=30)
        out = self.run_signal(gain, [noise(-70)] * 5 + [tone(-45)] + [tone(-10) for _ in range(5)])
        for chunk in out:
            self.assertLessEqual(float(np.max(np.abs(chunk))), 10 ** (-1 / 20) + 1e-4)

    def test_quiet_speech_with_plosive_peaks_still_reaches_target(self):
        """평균은 작고 순간 최고값만 큰 말소리(노트북 마이크 실측: RMS -34, 피크 0.3)도 목표까지 키운다."""
        gain = AutoGain(target_dbfs=-20, max_gain_db=30)

        def plosive_chunk():
            chunk = tone(-40) + noise(-70)
            chunk[400:404] += 0.3  # 0.25ms 순간 최고값: 청크 평균 약 -35 dBFS, 최고값 약 -10 dBFS
            return chunk

        out = self.run_signal(gain, [noise(-70)] * 5 + [plosive_chunk() for _ in range(20)])
        # 예전 방식(최고값 -1 dBFS에 맞춰 청크 전체 이득 제한)은 약 +9dB에서 멈춘다.
        self.assertGreater(gain.gain_db, 13)
        self.assertGreater(dbfs(out[-1]), -24)
        for chunk in out:
            self.assertLessEqual(float(np.max(np.abs(chunk))), 10 ** (-1 / 20) + 1e-4)
        self.assertGreater(gain.limited_blocks, 0)

    def test_limiter_releases_after_peak(self):
        """파열음 뒤에는 리미터가 풀려 말소리를 계속 키운다."""
        gain = AutoGain(target_dbfs=-20, max_gain_db=30)
        spike = tone(-40)
        spike[0:4] += 0.3
        self.run_signal(gain, [noise(-70)] * 5 + [tone(-40)] * 10 + [spike])
        out = self.run_signal(gain, [tone(-40) for _ in range(15)])
        self.assertAlmostEqual(dbfs(out[-1]), -20, delta=1.5)

    def test_noise_floor_rise_is_not_speech(self):
        """잡음이 서서히 커지면 말소리로 보지 않아 잡음을 계속 키우지 않는다."""
        gain = AutoGain()
        for i in range(200):
            gain.process(noise(-70 + i * 0.05))  # 청크당 0.05dB 상승(잡음 추적 속도 0.1dB보다 느림)
        self.assertEqual(gain.gain_db, 0.0)


class ConfigTests(unittest.TestCase):
    def test_default_enabled_and_public(self):
        config = SttConfig()
        self.assertIsInstance(config.make_auto_gain(), AutoGain)
        self.assertEqual(config.public_info()["auto_gain"], {"enabled": True, "target_dbfs": -20.0, "max_gain_db": 30.0})

    def test_disable_and_validate(self):
        self.assertIsNone(SttConfig(auto_gain=False).make_auto_gain())
        with self.assertRaises(ValueError):
            SttConfig(auto_gain_max_db=100)
        with self.assertRaises(ValueError):
            SttConfig(auto_gain="yes")


class TranscriberGainTests(unittest.TestCase):
    def test_transcriber_feeds_raised_samples_and_reports_level(self):
        service = ClockService()
        sid = service.start_session("gain test")
        engine = FakeEngine([("", False)] * 30)
        t = SherpaTranscriber(engine, service, sid, "fake", "1", auto_gain=AutoGain())
        signal = [noise(-70)] * 5 + [tone(-45) for _ in range(10)]
        for i, samples in enumerate(signal):
            t.accept_chunk(AudioChunk(sid, i, i * 100, (i + 1) * 100, RATE, 1, "float32", samples))
        fed = engine.received[-1][1]
        self.assertGreater(dbfs(fed), dbfs(signal[-1]) + 15)
        info = t.level_info()
        self.assertAlmostEqual(info["input_level_dbfs"], -45, delta=1.0)
        self.assertGreater(info["gain_db"], 15)

    def test_without_gain_samples_are_unchanged(self):
        service = ClockService()
        sid = service.start_session("no gain")
        engine = FakeEngine([("", False)])
        t = SherpaTranscriber(engine, service, sid, "fake", "1")
        samples = tone(-45)
        t.accept_chunk(AudioChunk(sid, 0, 0, 100, RATE, 1, "float32", samples))
        np.testing.assert_array_equal(engine.received[0][1], samples)
        self.assertEqual(t.level_info(), {"input_level_dbfs": None, "gain_db": None})


if __name__ == "__main__":
    unittest.main()
