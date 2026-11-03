from facemesh.recognition import FaceTracker


def _face(lm_cls, cx, cy):
    return [lm_cls(cx + 0.001 * i, cy + 0.001 * (i % 3)) for i in range(10)]


def test_ids_stable_through_reorder(lm_cls):
    tr = FaceTracker()
    assert tr.update([_face(lm_cls, 0.3, 0.5), _face(lm_cls, 0.7, 0.5)]) == [0, 1]
    # list order swapped, positions barely moved -> IDs stick
    assert tr.update([_face(lm_cls, 0.71, 0.5), _face(lm_cls, 0.31, 0.5)]) == [1, 0]


def test_new_face_gets_new_id(lm_cls):
    tr = FaceTracker()
    tr.update([_face(lm_cls, 0.3, 0.5)])
    ids = tr.update([_face(lm_cls, 0.31, 0.5), _face(lm_cls, 0.8, 0.9)])
    assert ids[0] == 0 and ids[1] == 1


def test_track_survives_brief_dropout(lm_cls):
    tr = FaceTracker(max_age=5)
    tr.update([_face(lm_cls, 0.3, 0.5), _face(lm_cls, 0.7, 0.5)])
    tr.update([_face(lm_cls, 0.3, 0.5)])                       # face 1 missing 1 frame
    ids = tr.update([_face(lm_cls, 0.3, 0.5), _face(lm_cls, 0.7, 0.5)])
    assert ids == [0, 1]
