"""HTML session report generation."""
import json
import math


from scripts.report import build_report, load_session


def _rows(n=100, with_hr=True, bad_range=(20, 30)):
    rows = []
    for i in range(n):
        t = i / 30
        bad = bad_range[0] <= i < bad_range[1]
        med = {"quality": {"ok": not bad, "score": 0.5 if bad else 1.0,
                           "reasons": ["too dark"] if bad else []}}
        if not bad:
            med["symmetry"] = {"symmetry": 0.9, "asymmetry": 0.01, "worst_region": "mouth"}
            med["movement"] = {"expressivity": abs(math.sin(t)), "tremor_ratio": 0.1}
            med["blink"] = {"rate_per_min": 18.0, "incomplete_fraction": 0.1}
            if with_hr:
                med["heart_rate"] = {"bpm": 72.0, "reliable": True, "rmssd_ms": 40.0}
        rows.append({"frame": i, "faces": [{"medical": med}]})
    return rows


def test_report_contains_disclaimer_first():
    h = build_report(_rows())
    assert "Screening and educational use only" in h
    assert h.index("Screening and educational use only") < h.index("Capture quality")


def test_report_sections_present():
    h = build_report(_rows())
    for section in ["Capture quality", "Facial symmetry", "Expressivity",
                    "Blink", "Contactless heart rate"]:
        assert section in h


def test_report_reports_suppressed_frames():
    h = build_report(_rows(bad_range=(20, 40)))
    assert "too dark" in h
    assert "Frames suppressed" in h


def test_report_is_valid_standalone_html():
    h = build_report(_rows())
    assert h.startswith("<!doctype html>")
    assert h.strip().endswith("</html>")
    assert h.count("<svg") == h.count("</svg>")
    assert "<script" not in h          # no JS dependency


def test_report_without_heart_rate_omits_section():
    h = build_report(_rows(with_hr=False))
    assert "Contactless heart rate" not in h
    assert "Facial symmetry" in h


def test_report_with_protocol():
    proto = {
        "tasks": {"smile": {"task": "smile", "frames": 90, "peak_asymmetry": 0.09,
                            "peak_symmetry": 0.4, "worst_region": "mouth",
                            "peak_expressivity": 0.9, "mean_quality": 1.0}},
        "summary": {"tasks_completed": 1, "tasks_total": 5, "worst_task": "smile",
                    "worst_task_asymmetry": 0.09, "worst_task_region": "mouth",
                    "movement_excess_asymmetry": 0.08},
    }
    h = build_report(_rows(), protocol=proto)
    assert "Guided task protocol" in h and "smile" in h
    assert "Movement excess asym" in h


def test_empty_session_still_renders():
    h = build_report([])
    assert "Screening and educational use only" in h
    assert h.strip().endswith("</html>")


def test_html_escaping_of_titles():
    h = build_report([], title="<script>alert(1)</script>")
    assert "<script>alert(1)</script>" not in h
    assert "&lt;script&gt;" in h


def test_load_session(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in _rows(5)) + "\n")
    assert len(load_session(p)) == 5
