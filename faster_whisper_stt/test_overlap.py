import unittest

from faster_whisper_stt.overlap import OverlapMerger


class OverlapMergerTests(unittest.TestCase):
    def test_overlap_sliding_and_repeated_words(self):
        merger = OverlapMerger()
        self.assertEqual(merger.update([(0, 0.5, ' 안녕'), (0.6, 1, ' 하세요')]), '')
        self.assertEqual(merger.update([(0.1, 0.6, ' 안녕'), (0.7, 1.1, ' 하세요')]), '안녕 하세요')
        self.assertEqual(merger.update([(0.15, 0.65, ' 안녕'), (0.75, 1.15, ' 하세요'), (3, 3.5, ' 안녕')]), '')
        self.assertEqual(merger.update([(3.1, 3.6, ' 안녕')]), '안녕')
        self.assertEqual(merger.final_text(), '안녕 하세요 안녕')

    def test_corrected_hypothesis_and_unconfirmed_tail(self):
        merger = OverlapMerger()
        merger.update([(0, 0.5, ' 오늘'), (0.6, 1, ' 날이')])
        self.assertEqual(merger.update([(0, 0.5, ' 오늘'), (0.6, 1, ' 날씨')]), '오늘')
        self.assertEqual(merger.pending_text(), '날씨')
        self.assertEqual(merger.update([(0.6, 1, ' 날씨')]), '날씨')
        self.assertEqual(merger.final_text(), '오늘 날씨')

    def test_empty_result_breaks_stability(self):
        merger = OverlapMerger()
        merger.update([(0, 0.5, ' 안녕')])
        merger.update([])
        self.assertEqual(merger.update([(0, 0.5, ' 안녕')]), '')


if __name__ == '__main__':
    unittest.main()
