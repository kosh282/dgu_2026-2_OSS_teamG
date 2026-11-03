import numpy as np

from facemesh.recorder import VideoRecorder


class FakeWriter:
    def __init__(self):
        self.count = 0
        self.released = False

    def write(self, f):
        self.count += 1

    def release(self):
        self.released = True


def _factory_holder():
    made = {}

    def factory(path, fourcc, fps, size):
        w = FakeWriter()
        made["writer"] = w
        made["path"] = path
        return w
    return factory, made


def test_toggle_start_write_stop(tmp_path):
    factory, made = _factory_holder()
    rec = VideoRecorder(out_dir=tmp_path, writer_factory=factory)
    on, path = rec.toggle((64, 48), now=100.0)
    assert on and rec.is_recording and "recording_" in path.name
    for _ in range(5):
        rec.write(np.zeros((48, 64, 3), np.uint8))
    assert made["writer"].count == 5
    assert rec.elapsed(102.0) == 2.0
    off, _ = rec.toggle((64, 48), now=103.0)
    assert off is False and made["writer"].released is True


def test_write_is_noop_when_not_recording():
    rec = VideoRecorder(writer_factory=lambda *a: FakeWriter())
    rec.write(np.zeros((48, 64, 3), np.uint8))  # should not raise
    assert rec.is_recording is False
