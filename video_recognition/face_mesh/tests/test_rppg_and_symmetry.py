import json

import compare_reference_hr as cmp_hr
import rppg_confidence as rc
import symmetry_composite as sym


def _write_jsonl(tmp_path, rows, name="s.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


# --- rppg_confidence ---------------------------------------------------------

def test_band_widens_as_quality_drops():
    good = rc.confidence_band(70.0, quality=0.95, prominence=0.95)
    bad = rc.confidence_band(70.0, quality=0.1, prominence=0.1)
    assert good["half_width"] < bad["half_width"]
    assert good["label"] == "high"
    assert bad["label"] == "reject"


def test_band_none_hr_is_no_estimate():
    b = rc.confidence_band(None, 0.9, 0.9)
    assert b["label"] == "no-estimate" and b["hr"] is None


def test_snr_db_is_normalized():
    # 20 dB SNR should read as top quality.
    b = rc.confidence_band(65.0, quality=20.0, prominence=1.0)
    assert b["label"] in ("high", "moderate")


def test_annotate_and_summarize(tmp_path):
    rows = [{"frame": i, "hr": 66, "quality": 0.9 if i % 2 else 0.1} for i in range(10)]
    recs = rc.load_records(_write_jsonl(tmp_path, rows))
    annotated, hr_k = rc.annotate(recs)
    assert hr_k == "hr"
    assert all("hr_confidence" in r for r in annotated)
    s = rc.summarize(annotated)
    assert s["frames"] == 10 and 0 < s["usable"] < 10


# --- compare_reference_hr ----------------------------------------------------

def test_agreement_perfect_match():
    pairs = [(60, 60), (70, 70), (80, 80)]
    s = cmp_hr.agreement_stats(pairs)
    assert s["mae"] == 0.0 and s["bias"] == 0.0 and s["pearson_r"] == 1.0


def test_agreement_constant_bias():
    pairs = [(62, 60), (72, 70), (82, 80)]
    s = cmp_hr.agreement_stats(pairs)
    assert s["bias"] == 2.0 and s["mae"] == 2.0


def test_pairing_within_tolerance(tmp_path):
    session = [{"t": float(i), "hr": 60 + i} for i in range(5)]
    recs = cmp_hr.load_records(_write_jsonl(tmp_path, session))
    ref = [(0.1, 60), (1.1, 61), (2.1, 62), (3.1, 63), (4.1, 64)]
    pairs = cmp_hr.pair_up(recs, ref, tolerance=0.5, session_offset=0.0)
    assert len(pairs) == 5
    stats = cmp_hr.agreement_stats(pairs)
    assert stats["mae"] == 0.0


def test_pairing_rejects_out_of_tolerance(tmp_path):
    session = [{"t": 0.0, "hr": 60}]
    recs = cmp_hr.load_records(_write_jsonl(tmp_path, session))
    ref = [(5.0, 60)]
    assert cmp_hr.pair_up(recs, ref, tolerance=1.0, session_offset=0.0) == []


def test_reference_csv_parsing(tmp_path):
    p = tmp_path / "ref.csv"
    p.write_text("t,hr\n0.0,58\n1.0,59\n")
    ref = cmp_hr.load_reference(str(p))
    assert ref == [(0.0, 58.0), (1.0, 59.0)]


# --- symmetry_composite ------------------------------------------------------

def test_composite_perfect_symmetry(tmp_path):
    rows = [{"frame": i, "eye_symmetry": 1.0, "mouth_symmetry": 1.0} for i in range(4)]
    recs = sym.load_records(_write_jsonl(tmp_path, rows))
    r = sym.composite_score(recs)
    assert r["score"] == 100.0


def test_composite_penalizes_synkinesis(tmp_path):
    rows = [{"frame": i, "mouth_symmetry": 1.0, "synkinesis": 1.0} for i in range(4)]
    recs = sym.load_records(_write_jsonl(tmp_path, rows))
    r = sym.composite_score(recs)
    # synkinesis=1.0 (max) should drag the composite well below 100.
    assert r["score"] is not None and r["score"] < 60


def test_composite_no_fields(tmp_path):
    rows = [{"frame": i, "hr": 60} for i in range(4)]
    recs = sym.load_records(_write_jsonl(tmp_path, rows))
    assert sym.composite_score(recs)["score"] is None


def test_composite_weighting(tmp_path):
    rows = [{"frame": i, "eye_symmetry": 1.0, "mouth_symmetry": 0.0} for i in range(4)]
    recs = sym.load_records(_write_jsonl(tmp_path, rows))
    heavy_mouth = sym.composite_score(recs, {"mouth": 3.0})["score"]
    equal = sym.composite_score(recs)["score"]
    assert heavy_mouth < equal  # weighting the bad region down-weights the score
