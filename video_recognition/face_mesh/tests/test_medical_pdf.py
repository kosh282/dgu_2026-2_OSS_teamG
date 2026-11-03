"""PDF export: backend selection, fallback, and valid output."""
import re

import pytest

from facemesh.medical import pdf as pdfmod
from facemesh.medical.pdf import (
    INSTALL_HINT, PdfBackendUnavailable, available_backends,
    render_native_pdf, write_pdf,
)

reportlab = pytest.importorskip("reportlab")


ROWS = [
    {"frame": i, "faces": [{"medical": {
        "quality": {"ok": i % 10 != 0, "score": 1.0,
                    "reasons": [] if i % 10 else ["too dark"]},
        "symmetry": {"symmetry": 0.9, "asymmetry": 0.01, "worst_region": "mouth"},
        "movement": {"expressivity": 0.5, "tremor_ratio": 0.1},
        "blink": {"rate_per_min": 18.0, "incomplete_fraction": 0.1},
        "heart_rate": {"bpm": 72.0, "reliable": True},
    }}]} for i in range(60)
]
PROTO = {
    "tasks": {"smile": {"task": "smile", "frames": 90, "peak_asymmetry": 0.09,
                        "peak_symmetry": 0.4, "worst_region": "mouth",
                        "peak_expressivity": 0.9, "mean_quality": 1.0}},
    "summary": {"tasks_completed": 1, "tasks_total": 5, "worst_task": "smile",
                "worst_task_region": "mouth", "movement_excess_asymmetry": 0.08},
}


def _is_pdf(path):
    data = path.read_bytes()
    return data.startswith(b"%PDF-") and b"%%EOF" in data[-2048:]


def test_native_pdf_is_structurally_valid(tmp_path):
    out = render_native_pdf(ROWS, PROTO, tmp_path / "r.pdf", "Test", "disclaimer text")
    assert out.exists() and _is_pdf(out)


def test_native_pdf_has_one_or_more_pages(tmp_path):
    out = render_native_pdf(ROWS, PROTO, tmp_path / "r.pdf")
    pages = len(re.findall(rb"/Type\s*/Page[^s]", out.read_bytes()))
    assert pages >= 1


def test_native_pdf_contains_disclaimer_and_data(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    out = render_native_pdf(ROWS, PROTO, tmp_path / "r.pdf", "Test", "not a medical device")
    text = pypdf.PdfReader(str(out)).pages[0].extract_text()
    assert "Screening and educational use only" in text
    assert "not a medical device" in text
    assert "smile" in text


def test_native_pdf_handles_empty_session(tmp_path):
    out = render_native_pdf([], None, tmp_path / "empty.pdf")
    assert _is_pdf(out)


def test_available_backends_reports_reportlab():
    assert "reportlab" in available_backends()


def test_write_pdf_falls_back_to_reportlab(tmp_path, monkeypatch):
    """When no HTML backend exists, the native renderer must still produce a PDF."""
    monkeypatch.setattr(pdfmod, "available_backends", lambda: ["reportlab"])
    out, backend = write_pdf("<html></html>", ROWS, PROTO, tmp_path / "f.pdf")
    assert backend == "reportlab" and _is_pdf(out)


def test_write_pdf_prefers_html_backend(tmp_path, monkeypatch):
    called = {}

    def fake_html_to_pdf(html, out):
        called["used"] = True
        from pathlib import Path
        Path(out).write_bytes(b"%PDF-1.4\nfake\n%%EOF")
        return Path(out)

    monkeypatch.setattr(pdfmod, "available_backends", lambda: ["weasyprint", "reportlab"])
    monkeypatch.setattr(pdfmod, "html_to_pdf", fake_html_to_pdf)
    out, backend = write_pdf("<html></html>", ROWS, PROTO, tmp_path / "h.pdf")
    assert backend == "weasyprint" and called.get("used")


def test_write_pdf_raises_with_instructions_when_no_backend(tmp_path, monkeypatch):
    """The common case on a fresh machine must fail clearly, not obscurely."""
    monkeypatch.setattr(pdfmod, "available_backends", lambda: [])
    with pytest.raises(PdfBackendUnavailable) as exc:
        write_pdf("<html></html>", ROWS, PROTO, tmp_path / "x.pdf")
    assert "pip install" in str(exc.value)
    assert "print to PDF" in str(exc.value)


def test_install_hint_mentions_all_options():
    for token in ("weasyprint", "pdfkit", "reportlab", "browser"):
        assert token in INSTALL_HINT.lower()
