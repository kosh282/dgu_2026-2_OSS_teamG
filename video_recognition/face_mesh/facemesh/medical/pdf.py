"""PDF export for session reports.

The HTML report is the source of truth; PDF is a convenience for sharing and
archiving. Because no single PDF toolchain is available everywhere, this module
tries backends in order of fidelity and falls back gracefully:

1. **WeasyPrint** — renders the HTML directly, best fidelity (CSS, inline SVG).
2. **wkhtmltopdf** (via ``pdfkit``) — also renders the HTML.
3. **ReportLab** — no HTML rendering; draws a native PDF from the underlying
   data. Plainer, but pure Python with no system dependencies.

If none is installed, :func:`write_pdf` raises :class:`PdfBackendUnavailable`
with installation instructions rather than failing obscurely. Printing the HTML
to PDF from a browser always works and needs nothing installed.

SCREENING / EDUCATIONAL USE ONLY — not a diagnostic device.
"""

from __future__ import annotations

import importlib
import shutil
import statistics
from pathlib import Path


class PdfBackendUnavailable(RuntimeError):
    """No usable PDF backend is installed."""


INSTALL_HINT = (
    "No PDF backend found. Install one of:\n"
    '  pip install "face-mesh[pdf]"      # WeasyPrint, best fidelity\n'
    "  pip install pdfkit                # plus the wkhtmltopdf binary\n"
    "  pip install reportlab             # pure Python, plainer output\n"
    "Or open the HTML report in a browser and print to PDF."
)


def _has(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def available_backends() -> list[str]:
    """Names of backends usable right now, best first."""
    out = []
    if _has("weasyprint"):
        out.append("weasyprint")
    if _has("pdfkit") and shutil.which("wkhtmltopdf"):
        out.append("wkhtmltopdf")
    if _has("reportlab"):
        out.append("reportlab")
    return out


# --- HTML-rendering backends --------------------------------------------
def _weasyprint(html: str, out: Path) -> Path:
    from weasyprint import HTML  # noqa: PLC0415
    HTML(string=html).write_pdf(str(out))
    return out


def _wkhtmltopdf(html: str, out: Path) -> Path:
    import pdfkit  # noqa: PLC0415
    pdfkit.from_string(html, str(out), options={"quiet": ""})
    return out


def html_to_pdf(html: str, out: str | Path) -> Path:
    """Render HTML to PDF with the best available HTML backend."""
    out = Path(out)
    for name in available_backends():
        if name == "weasyprint":
            return _weasyprint(html, out)
        if name == "wkhtmltopdf":
            return _wkhtmltopdf(html, out)
    raise PdfBackendUnavailable(INSTALL_HINT)


# --- native backend ------------------------------------------------------
def _fmt(v, digits=3):
    return f"{v:.{digits}f}" if isinstance(v, (int, float)) else str(v)


def render_native_pdf(rows, protocol, out: str | Path,
                      title: str = "Facial metrics session",
                      disclaimer: str = "") -> Path:
    """Draw a PDF directly from the data using ReportLab (no HTML)."""
    from reportlab.lib import colors  # noqa: PLC0415
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    out = Path(out)
    styles = getSampleStyleSheet()
    warn = ParagraphStyle("warn", parent=styles["BodyText"], textColor=colors.HexColor("#b00020"),
                          backColor=colors.HexColor("#fff1f1"), borderPadding=6, leading=13)
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8,
                           textColor=colors.HexColor("#666666"))

    story = [Paragraph(title, styles["Title"])]
    story.append(Paragraph(
        "<b>Screening and educational use only.</b> " + disclaimer +
        " Not clinically validated. Landmark models lose accuracy on faces "
        "affected by neuromuscular disease. Consult a qualified clinician.", warn))
    story.append(Spacer(1, 6 * mm))

    meds = [f[0]["medical"] for f in
            (r.get("faces") for r in rows) if f and "medical" in f[0]]
    story.append(Paragraph(f"{len(rows)} frames analysed", styles["BodyText"]))

    def table(data, col_widths=None):
        t = Table(data, colWidths=col_widths, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f1f3")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#555555")),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
        ]))
        return t

    # quality
    q_ok = [m.get("quality", {}).get("ok") for m in meds if "quality" in m]
    if q_ok:
        good = 100.0 * sum(1 for v in q_ok if v) / len(q_ok)
        reasons: dict[str, int] = {}
        for m in meds:
            for r in m.get("quality", {}).get("reasons", []):
                reasons[r] = reasons.get(r, 0) + 1
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph("Capture quality", styles["Heading2"]))
        data = [["Usable frames", f"{good:.0f}%"],
                ["Frames suppressed", str(sum(1 for v in q_ok if not v))]]
        data += [[k, str(v)] for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])]
        story.append(table(data, [70 * mm, 30 * mm]))

    # protocol
    if protocol:
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph("Guided task protocol", styles["Heading2"]))
        head = [["Task", "Peak asym", "Peak sym", "Region", "Peak expr", "Frames"]]
        body = [[t["task"], _fmt(t["peak_asymmetry"], 4), _fmt(t["peak_symmetry"]),
                 t.get("worst_region") or "-", _fmt(t["peak_expressivity"], 2),
                 str(t["frames"])] for t in protocol.get("tasks", {}).values()]
        story.append(table(head + body))
        summ = protocol.get("summary", {})
        if "worst_task" in summ:
            story.append(Spacer(1, 2 * mm))
            story.append(Paragraph(
                f"Worst task: <b>{summ['worst_task']}</b> ({summ.get('worst_task_region','-')}); "
                f"movement-excess asymmetry "
                f"{_fmt(summ.get('movement_excess_asymmetry', 0), 4)}", styles["BodyText"]))

    # metric summaries
    def series(*path):
        vals = []
        for m in meds:
            cur = m
            for k in path:
                cur = cur.get(k) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if isinstance(cur, (int, float)) and not isinstance(cur, bool):
                vals.append(cur)
        return vals

    blocks = [
        ("Facial symmetry", series("symmetry", "symmetry"), 3),
        ("Expressivity", series("movement", "expressivity"), 3),
        ("Blink rate (per min)", series("blink", "rate_per_min"), 1),
        ("Heart rate (bpm)", [v for v in series("heart_rate", "bpm") if v], 0),
    ]
    rows_out = [["Metric", "Mean", "Min", "Max", "n"]]
    for label, vals, d in blocks:
        if vals:
            rows_out.append([label, _fmt(statistics.fmean(vals), d),
                             _fmt(min(vals), d), _fmt(max(vals), d), str(len(vals))])
    if len(rows_out) > 1:
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph("Metrics", styles["Heading2"]))
        story.append(table(rows_out))

    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        "Generated by face-mesh. " + disclaimer + " Charts are available in the "
        "HTML report; this native PDF is a plain-text fallback.", small))

    doc = SimpleDocTemplate(str(out), pagesize=A4, title=title,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm)
    doc.build(story)
    return out


def write_pdf(html: str, rows, protocol, out: str | Path,
              title: str = "Facial metrics session", disclaimer: str = "") -> tuple[Path, str]:
    """Write a PDF using the best backend available.

    Returns ``(path, backend_used)``. Raises :class:`PdfBackendUnavailable`
    when nothing is installed.
    """
    backends = available_backends()
    if not backends:
        raise PdfBackendUnavailable(INSTALL_HINT)
    best = backends[0]
    if best in ("weasyprint", "wkhtmltopdf"):
        return html_to_pdf(html, out), best
    return render_native_pdf(rows, protocol, out, title, disclaimer), "reportlab"
