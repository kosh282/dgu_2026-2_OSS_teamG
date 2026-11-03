import json
import math

import session_io
import session_trends


def _write_jsonl(tmp_path, rows, name="s.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def _write_csv(tmp_path, header, rows, name="s.csv"):
    lines = [",".join(header)]
    lines += [",".join("" if v is None else str(v) for v in row) for row in rows]
    p = tmp_path / name
    p.write_text("\n".join(lines))
    return p


def test_load_jsonl_and_axis(tmp_path):
    rows = [{"t": i * 0.1, "hr": 60 + i, "face_id": 0} for i in range(10)]
    recs = session_io.load_records(_write_jsonl(tmp_path, rows))
    assert len(recs) == 10
    xs, label = session_io.time_axis(recs)
    assert label == "time (s)"
    assert math.isclose(xs[1], 0.1)


def test_load_csv_coerces_numbers(tmp_path):
    p = _write_csv(tmp_path, ["frame", "hr"], [[0, 61.0], [1, 62.0]])
    recs = session_io.load_records(p)
    assert recs[0]["hr"] == 61.0 and isinstance(recs[0]["hr"], float)
    assert recs[0]["frame"] == 0 and isinstance(recs[0]["frame"], int)


def test_numeric_field_autodetect_skips_axes(tmp_path):
    rows = [{"frame": i, "t": i * 0.1, "face_id": 0, "hr": 60, "resp": 15}
            for i in range(6)]
    recs = session_io.load_records(_write_jsonl(tmp_path, rows))
    fields = session_io.numeric_fields(recs)
    assert set(fields) == {"hr", "resp"}


def test_multiface_ids_and_filter(tmp_path):
    rows = []
    for i in range(4):
        rows.append({"frame": i, "face_id": 0, "hr": 60})
        rows.append({"frame": i, "face_id": 1, "hr": 80})
    recs = session_io.load_records(_write_jsonl(tmp_path, rows))
    assert session_io.face_ids(recs) == [0, 1]
    f1 = session_io.filter_face(recs, 1)
    assert all(r["face_id"] == 1 for r in f1) and len(f1) == 4


def test_rolling_mean_handles_gaps():
    ys = [1.0, None, 3.0, 5.0]
    out = session_trends.rolling_mean(ys, window=2)
    assert out[0] == 1.0
    assert out[1] == 1.0  # gap carries previous buffer
    assert math.isclose(out[3], 4.0)


def test_trends_plot_writes_png(tmp_path):
    rows = [{"t": i * 0.1, "hr": 60 + (i % 5), "symmetry": 0.9} for i in range(40)]
    recs = session_io.load_records(_write_jsonl(tmp_path, rows))
    out = tmp_path / "trends.png"
    session_trends.plot_session(recs, ["hr", "symmetry"], 5, "test", out)
    assert out.exists() and out.stat().st_size > 0


def test_trends_main_autodetect(tmp_path):
    rows = [{"frame": i, "hr": 60 + i, "resp": 15} for i in range(20)]
    p = _write_jsonl(tmp_path, rows)
    out = tmp_path / "t.png"
    rc = session_trends.main(["--input", str(p), "--out", str(out), "--window", "4"])
    assert rc == 0 and out.exists()
