import time

import numpy as np

from facemesh.camera import ThreadedCamera


class FakeCap:
    def __init__(self):
        self.n = 0
        self._open = True

    def read(self):
        self.n += 1
        return True, np.full((48, 64, 3), self.n % 255, np.uint8)

    def isOpened(self):
        return self._open

    def get(self, prop):
        return 64.0 if prop == 3 else 48.0

    def set(self, *a):
        pass

    def release(self):
        self._open = False


def test_threaded_camera_reads_and_stops():
    cap = FakeCap()
    cam = ThreadedCamera(cap=cap, mirror=False).start()
    time.sleep(0.05)
    ok, frame = cam.read()
    assert ok and frame is not None and frame.shape == (48, 64, 3)
    assert cam.frame_size == (64, 48)
    cam.stop()
    assert cap._open is False


def test_mirror_flips_frame():
    # a frame with a distinct left edge; mirror should move it to the right
    class LR(FakeCap):
        def read(self):
            f = np.zeros((10, 10, 3), np.uint8)
            f[:, 0] = 255  # bright left column
            return True, f
    cam = ThreadedCamera(cap=LR(), mirror=True).start()
    time.sleep(0.03)
    ok, frame = cam.read()
    cam.stop()
    assert ok and frame[:, -1].mean() > frame[:, 0].mean()
