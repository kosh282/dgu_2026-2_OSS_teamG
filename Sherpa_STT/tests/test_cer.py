import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from evaluate_cer import normalize, edit_distance

class CerTests(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(edit_distance('안녕하세요', '안녕하세요'), 0)
    def test_edited(self):
        self.assertEqual(edit_distance('안녕', '안녕하'), 1)
    def test_normalize(self):
        self.assertEqual(normalize('안녕, 하세요!'), '안녕하세요')

if __name__ == '__main__':
    unittest.main()
