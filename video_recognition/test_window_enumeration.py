import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from video_recognition.integrated_analysis import list_windows


@unittest.skipUnless(sys.platform == 'win32', 'Windows 창 열거 검사')
class WindowEnumerationTests(unittest.TestCase):
    def test_skips_own_windows_before_reading_titles(self):
        user32 = MagicMock()
        user32.IsWindowVisible.return_value = True

        def process_id(hwnd, owner):
            owner._obj.value = os.getpid() if hwnd == 1 else os.getpid() + 1
            return 1

        def text_length(hwnd):
            if hwnd == 1:
                raise AssertionError('같은 프로세스 창의 제목을 읽어서는 안 됩니다')
            return 4

        def window_text(hwnd, buffer, size):
            buffer.value = 'Zoom'
            return 4

        def enumerate_windows(callback, _):
            callback(1, 0)
            callback(2, 0)
            return True

        user32.GetWindowThreadProcessId.side_effect = process_id
        user32.GetWindowTextLengthW.side_effect = text_length
        user32.GetWindowTextW.side_effect = window_text
        user32.EnumWindows.side_effect = enumerate_windows
        with patch('ctypes.WinDLL', return_value=user32):
            self.assertEqual(list_windows(), [(2, 'Zoom')])


if __name__ == '__main__':
    unittest.main()
