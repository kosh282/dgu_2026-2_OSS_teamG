import json

from facemesh.export import (
    CsvExporter,
    JsonlExporter,
    create_exporter,
    flatten_face,
)


FACE = {
    "ear": 0.29,
    "blink": {"left": 1, "right": 1, "total": 1, "rate_per_min": 12.0},
    "gaze": {"h": 0.5, "v": 0.5, "dir": "center"},
    "head_pose": {"pitch": 1.0, "yaw": -8.0, "roll": 0.5},
}


def test_flatten_keys():
    flat = flatten_face(FACE)
    assert "blink_left" in flat and "gaze_dir" in flat and "head_pose_yaw" in flat


def test_flatten_landmarks():
    flat = flatten_face({"landmarks": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]})
    assert flat["lm0_x"] == 0.1 and flat["lm1_z"] == 0.6


def test_jsonl_roundtrip(tmp_path):
    p = tmp_path / "s.jsonl"
    with JsonlExporter(p) as e:
        e.write_frame(0, [FACE])
        e.write_frame(1, [])
    lines = p.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["faces"][0]["ear"] == 0.29


def test_csv_header_and_rows(tmp_path):
    p = tmp_path / "s.csv"
    with CsvExporter(p) as e:
        e.write_frame(0, [FACE])
        e.write_frame(1, [FACE, FACE])
    rows = p.read_text().splitlines()
    assert rows[0].startswith("frame,face,")
    assert len(rows) == 4  # header + 1 + 2


def test_factory_picks_by_extension(tmp_path):
    assert isinstance(create_exporter(tmp_path / "a.csv"), CsvExporter)
    assert isinstance(create_exporter(tmp_path / "a.jsonl"), JsonlExporter)
