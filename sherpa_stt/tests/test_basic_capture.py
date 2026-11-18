import unittest

import numpy as np

from sherpa_stt.basic_demo import LiveCapture


class BasicCaptureTests(unittest.TestCase):
    def test_queue_overflow_is_counted_and_sequence_gap_is_visible(self):
        capture = LiveCapture('microphone', queue_size=1)
        samples = np.array([[0.1]], dtype=np.float32)
        capture._put(samples)
        samples[0, 0] = 0.5
        capture._put(samples)
        block_id, _, stored = capture.blocks.get_nowait()
        self.assertEqual(block_id, 1)
        self.assertAlmostEqual(float(stored[0, 0]), 0.1)
        self.assertEqual(capture.dropped, 1)
        capture._put(samples)
        self.assertEqual(capture.blocks.get_nowait()[0], 3)


if __name__ == '__main__':
    unittest.main()
