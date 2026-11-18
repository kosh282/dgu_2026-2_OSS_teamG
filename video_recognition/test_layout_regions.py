"""화면 구성 후보와 얼굴 중복 판정의 회귀 검사."""
import unittest

import cv2
import numpy as np

from video_recognition.face_analysis import _box_overlap, _same_face_box
from video_recognition.layout_regions import detect_layout_regions


class LayoutRegionsTests(unittest.TestCase):
    def assert_regions_cover(self, rectangles):
        image = np.zeros((540, 960, 3), dtype=np.uint8)
        for index, (x0, y0, x1, y1) in enumerate(rectangles):
            image[y0:y1, x0:x1] = [35 + (30 * index) % 200, 80 + (20 * index) % 160,
                                    180 - (15 * index) % 160]
            cv2.circle(image, ((x0 + x1) // 2, (y0 + y1) // 2), 25, (120, 130, 140), -1)
        regions = detect_layout_regions(image)
        for x0, y0, x1, y1 in rectangles:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            self.assertTrue(any(a <= cx < c and b <= cy < d for a, b, c, d in regions))
        self.assertTrue(all(0 <= a < c <= 960 and 0 <= b < d <= 540 for a, b, c, d in regions))
        self.assertLessEqual(len(regions), 12)

    def test_two_by_two(self):
        self.assert_regions_cover([(0, 20, 478, 268), (482, 20, 960, 268),
                                   (0, 272, 478, 520), (482, 272, 960, 520)])

    def test_three_by_two(self):
        self.assert_regions_cover([(x, y, x + 316, y + 245)
                                   for y in (20, 270) for x in (0, 320, 640)])

    def test_three_by_three(self):
        self.assert_regions_cover([(x, y, x + 316, y + 174)
                                   for y in (0, 180, 360) for x in (0, 320, 640)])

    def test_screen_share_and_vertical_strip(self):
        self.assert_regions_cover([(0, 40, 635, 500), (640, 40, 960, 190),
                                   (640, 195, 960, 345), (640, 350, 960, 500)])

    def test_no_boundaries_returns_no_candidates(self):
        self.assertEqual(detect_layout_regions(np.zeros((540, 960, 3), dtype=np.uint8)), [])

    def test_close_duplicate_and_distinct_boxes(self):
        first = (0, 0, 1, 1)
        duplicate = (0.37, -0.195, 0.924, 1.287)
        self.assertLess(_box_overlap(first, duplicate), 0.7)
        self.assertTrue(_same_face_box(first, duplicate))
        self.assertFalse(_same_face_box(first, (0.8, 0, 1.8, 1)))


if __name__ == '__main__':
    unittest.main()
