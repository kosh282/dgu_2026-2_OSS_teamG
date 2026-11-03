"""Regression tests for the rPPG validation harness (issue #2)."""
import json


from scripts.validate_rppg import cmd_report


def _write(path, n=100, bad=40, good_bpm=70.0, bad_bpm=120.0, new_format=True):
    lines = []
    for i in range(n):
        is_bad = i < bad
        q = {"ok": not is_bad, "score": 0.4 if is_bad else 1.0,
             "reasons": ["uneven lighting"] if is_bad else []}
        if new_format:
            # After the fix: bad frames never reached the estimator.
            hr = {"bpm": 0.0 if is_bad else good_bpm, "confidence": 0.0 if is_bad else 0.6,
                  "ready": not is_bad, "reliable": not is_bad}
            rec = {"t": i / 10, "hr": hr, "quality": q, "used": not is_bad}
        else:
            # Before the fix: bad frames produced garbage estimates.
            hr = {"bpm": bad_bpm if is_bad else good_bpm, "confidence": 0.6,
                  "ready": True, "reliable": True}
            rec = {"t": i / 10, "hr": hr, "quality": q}
        lines.append(json.dumps(rec))
    path.write_text("\n".join(lines))
    return path


class _Args:
    def __init__(self, input, reference):
        self.input = input
        self.reference = reference


def test_report_excludes_gated_frames(tmp_path, capsys):
    p = _write(tmp_path / "s.jsonl")
    cmd_report(_Args(str(p), [70, 70, 70]))
    out = capsys.readouterr().out
    assert "excluded by quality gate: 40" in out
    assert "used: 60" in out


def test_old_format_session_is_rescored(tmp_path, capsys):
    """Sessions recorded before the fix must be re-scored, not trusted."""
    p = _write(tmp_path / "old.jsonl", new_format=False, bad_bpm=120.0)
    cmd_report(_Args(str(p), [70, 70, 70]))
    out = capsys.readouterr().out
    assert "excluded by quality gate: 40" in out
    # The 120 bpm garbage frames must not drag the mean up.
    assert "mean 70" in out


def test_polluted_frames_would_wreck_the_estimate(tmp_path, capsys):
    """Demonstrates the bug: including gated frames corrupts the reported MAE."""
    p = _write(tmp_path / "all_good_flags.jsonl", new_format=False, bad_bpm=120.0)
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    for r in rows:                       # pretend the gate passed everything
        r["quality"]["ok"] = True
        r["quality"]["reasons"] = []
    p.write_text("\n".join(json.dumps(r) for r in rows))

    cmd_report(_Args(str(p), [70, 70, 70]))
    out = capsys.readouterr().out
    assert "excluded by quality gate: 0" in out
    # With the 120 bpm frames included the mean is badly inflated.
    mean_line = [l for l in out.splitlines() if "estimated bpm" in l][0]
    mean = float(mean_line.split("mean")[1].split()[0])
    assert mean > 85


def test_no_usable_frames_reports_cleanly(tmp_path, capsys):
    p = _write(tmp_path / "none.jsonl", n=50, bad=50)
    rc = cmd_report(_Args(str(p), [70]))
    out = capsys.readouterr().out
    assert "used: 0" in out or rc == 1


def test_report_without_reference_computes_no_accuracy(tmp_path, capsys):
    p = _write(tmp_path / "s.jsonl")
    cmd_report(_Args(str(p), []))
    out = capsys.readouterr().out
    assert "No --reference given" in out
    assert "MAE" not in out
