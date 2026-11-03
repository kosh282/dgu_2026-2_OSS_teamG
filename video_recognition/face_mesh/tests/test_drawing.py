import numpy as np

from facemesh.drawing import color_for_id, draw_face_label


class LM:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def test_color_for_id_is_stable_and_cycles():
    assert color_for_id(0) == color_for_id(0)          # deterministic
    assert color_for_id(0) != color_for_id(1)          # distinct neighbours
    from facemesh.drawing import _ID_PALETTE
    n = len(_ID_PALETTE)
    assert color_for_id(0) == color_for_id(n)          # wraps around


def test_draw_face_label_runs():
    img = np.zeros((200, 200, 3), np.uint8)
    face = [LM(0.4 + 0.001 * i, 0.4 + 0.001 * (i % 5)) for i in range(20)]
    draw_face_label(img, face, "#0", color_for_id(0))
    assert img.any()  # something was drawn
