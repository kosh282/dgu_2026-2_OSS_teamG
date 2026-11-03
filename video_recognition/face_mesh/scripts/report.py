"""Generate a self-contained HTML session report.

Turns an exported session (``--export session.jsonl``) and/or a protocol result
(``--protocol-out protocol.json``) into a single standalone HTML file: no
JavaScript dependencies, no external assets, charts drawn as inline SVG. Open it
in a browser, print it to PDF if you want a document.

The report leads with the disclaimer, shows the capture-quality log alongside
the metrics (so a reader can see when measurements were suppressed), and never
presents a grade or a diagnosis.

    python scripts/report.py session.jsonl --protocol protocol.json -o report.html

SCREENING / EDUCATIONAL USE ONLY — not a diagnostic device.
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from facemesh.medical import DISCLAIMER  # noqa: E402
from facemesh.medical.pdf import (  # noqa: E402
    INSTALL_HINT,
    PdfBackendUnavailable,
    available_backends,
    write_pdf,
)

CSS = """
:root { --fg:#1c2024; --muted:#697077; --line:#e3e6e8; --accent:#0f62fe;
        --good:#198038; --warn:#b28600; --bad:#da1e28; --bg:#fff; }
* { box-sizing:border-box; }
body { margin:0; background:#f4f5f6; color:var(--fg);
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:940px; margin:32px auto; background:var(--bg); padding:40px 48px;
        border:1px solid var(--line); border-radius:6px; }
h1 { font-size:26px; margin:0 0 4px; letter-spacing:-.01em; }
h2 { font-size:17px; margin:38px 0 12px; padding-bottom:7px;
     border-bottom:1px solid var(--line); font-weight:600; }
.sub { color:var(--muted); font-size:13px; margin-bottom:26px; }
.disclaimer { border-left:3px solid var(--bad); background:#fff1f1; padding:14px 18px;
              margin:22px 0 8px; font-size:13.5px; }
.disclaimer strong { color:var(--bad); }
table { border-collapse:collapse; width:100%; font-size:14px; }
th,td { text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase;
     letter-spacing:.04em; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.cards { display:flex; gap:14px; flex-wrap:wrap; margin:6px 0 4px; }
.card { flex:1 1 160px; border:1px solid var(--line); border-radius:5px; padding:14px 16px; }
.card .k { color:var(--muted); font-size:11.5px; text-transform:uppercase;
           letter-spacing:.04em; }
.card .v { font-size:24px; font-weight:600; margin-top:3px; font-variant-numeric:tabular-nums; }
.card .u { color:var(--muted); font-size:12px; margin-left:3px; font-weight:400; }
.pill { display:inline-block; padding:2px 9px; border-radius:11px; font-size:11.5px;
        font-weight:600; }
.pill.good { background:#defbe6; color:var(--good); }
.pill.warn { background:#fcf4d6; color:var(--warn); }
.pill.bad  { background:#fff1f1; color:var(--bad); }
.note { color:var(--muted); font-size:12.5px; margin-top:8px; }
footer { color:var(--muted); font-size:12px; margin-top:34px; padding-top:14px;
         border-top:1px solid var(--line); }
svg { display:block; max-width:100%; }
"""


def _esc(s) -> str:
    return html.escape(str(s))


def _sparkline(values, width=860, height=110, colour="#0f62fe", label=""):
    """Inline SVG line chart. Returns '' when there is nothing to draw."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    pad = 22
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        x = pad + (width - 2 * pad) * i / (n - 1)
        y = height - pad - (height - 2 * pad) * (v - lo) / span
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    return f"""<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(label)}">
  <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}"
        stroke="#e3e6e8" stroke-width="1"/>
  <polyline fill="none" stroke="{colour}" stroke-width="1.8" points="{poly}"/>
  <text x="{pad}" y="14" font-size="11" fill="#697077">{_esc(label)}</text>
  <text x="{width-pad}" y="14" font-size="11" fill="#697077" text-anchor="end">
    max {hi:.3g}</text>
  <text x="{width-pad}" y="{height-6}" font-size="11" fill="#697077" text-anchor="end">
    min {lo:.3g}</text>
</svg>"""


def _card(key, value, unit="", pill=None):
    p = f' <span class="pill {pill[0]}">{_esc(pill[1])}</span>' if pill else ""
    u = f'<span class="u">{_esc(unit)}</span>' if unit else ""
    return (f'<div class="card"><div class="k">{_esc(key)}</div>'
            f'<div class="v">{_esc(value)}{u}{p}</div></div>')


def load_session(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _primary_med(rows):
    """Medical dicts for face 0, frame by frame."""
    out = []
    for r in rows:
        faces = r.get("faces") or []
        if faces and "medical" in faces[0]:
            out.append(faces[0]["medical"])
    return out


def _series(meds, *path):
    vals = []
    for m in meds:
        cur = m
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else None
            if cur is None:
                break
        vals.append(cur if isinstance(cur, (int, float)) else None)
    return vals


def build_report(rows, protocol=None, title="Facial metrics session") -> str:
    meds = _primary_med(rows)
    parts: list[str] = []

    parts.append(f"<h1>{_esc(title)}</h1>")
    parts.append(f'<div class="sub">{len(rows)} frames analysed</div>')
    parts.append(f'<div class="disclaimer"><strong>Screening and educational use only.</strong> '
                 f'{_esc(DISCLAIMER)} These measurements are not clinically validated and must '
                 f'not be used to make a health decision. Facial landmark models are trained '
                 f'predominantly on healthy faces and lose accuracy on faces affected by '
                 f'neuromuscular disease. Consult a qualified clinician.</div>')

    # --- capture quality ---
    q_scores = _series(meds, "quality", "score")
    q_ok = [m.get("quality", {}).get("ok") for m in meds if "quality" in m]
    reasons: dict[str, int] = {}
    for m in meds:
        for r in m.get("quality", {}).get("reasons", []):
            reasons[r] = reasons.get(r, 0) + 1
    if q_ok:
        good_pct = 100.0 * sum(1 for v in q_ok if v) / len(q_ok)
        pill = ("good", "usable") if good_pct > 80 else (
            ("warn", "marginal") if good_pct > 50 else ("bad", "poor"))
        parts.append("<h2>Capture quality</h2>")
        parts.append('<div class="cards">')
        parts.append(_card("Usable frames", f"{good_pct:.0f}", "%", pill))
        clean = [v for v in q_scores if v is not None]
        if clean:
            parts.append(_card("Mean quality", f"{statistics.fmean(clean):.2f}"))
        parts.append(_card("Frames suppressed", sum(1 for v in q_ok if not v)))
        parts.append("</div>")
        if reasons:
            rows_html = "".join(
                f"<tr><td>{_esc(k)}</td><td class='num'>{v}</td></tr>"
                for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]))
            parts.append("<table><tr><th>Issue</th><th style='text-align:right'>Frames</th></tr>"
                         f"{rows_html}</table>")
        parts.append('<p class="note">Metrics are suppressed on frames that fail the quality '
                     'gate, rather than being reported unreliably.</p>')

    # --- protocol ---
    if protocol:
        parts.append("<h2>Guided task protocol</h2>")
        summ = protocol.get("summary", {})
        parts.append('<div class="cards">')
        if "worst_task" in summ:
            parts.append(_card("Worst task", summ["worst_task"]))
            parts.append(_card("Region", summ.get("worst_task_region", "-")))
        if "movement_excess_asymmetry" in summ:
            parts.append(_card("Movement excess asym", f"{summ['movement_excess_asymmetry']:.3f}"))
        parts.append(_card("Tasks completed",
                           f"{summ.get('tasks_completed', 0)}/{summ.get('tasks_total', 0)}"))
        parts.append("</div>")
        trows = "".join(
            f"<tr><td>{_esc(t['task'])}</td>"
            f"<td class='num'>{t['peak_asymmetry']:.4f}</td>"
            f"<td class='num'>{t['peak_symmetry']:.3f}</td>"
            f"<td>{_esc(t.get('worst_region', '-') or '-')}</td>"
            f"<td class='num'>{t['peak_expressivity']:.2f}</td>"
            f"<td class='num'>{t['frames']}</td></tr>"
            for t in protocol.get("tasks", {}).values())
        parts.append("<table><tr><th>Task</th><th style='text-align:right'>Peak asym</th>"
                     "<th style='text-align:right'>Peak sym</th><th>Region</th>"
                     "<th style='text-align:right'>Peak expr</th>"
                     "<th style='text-align:right'>Frames</th></tr>" + trows + "</table>")
        parts.append('<p class="note">Peak (not mean) values are shown: asymmetry that appears '
                     'only transiently during a movement is the signal of interest.</p>')

    # --- symmetry ---
    sym = _series(meds, "symmetry", "symmetry")
    if any(v is not None for v in sym):
        parts.append("<h2>Facial symmetry</h2>")
        clean = [v for v in sym if v is not None]
        parts.append('<div class="cards">')
        parts.append(_card("Mean symmetry", f"{statistics.fmean(clean)*100:.0f}", "%"))
        parts.append(_card("Lowest", f"{min(clean)*100:.0f}", "%"))
        parts.append("</div>")
        parts.append(_sparkline(sym, colour="#0f62fe", label="symmetry (1.0 = symmetric)"))

    # --- expressivity / tremor ---
    expr = _series(meds, "movement", "expressivity")
    if any(v is not None for v in expr):
        parts.append("<h2>Expressivity</h2>")
        clean = [v for v in expr if v is not None]
        tremor = [v for v in _series(meds, "movement", "tremor_ratio") if v is not None]
        parts.append('<div class="cards">')
        parts.append(_card("Peak expressivity", f"{max(clean)*100:.0f}", "%"))
        if tremor:
            hi = max(tremor)
            pill = ("bad", "detected") if hi > 0.35 else ("good", "none")
            parts.append(_card("Peak tremor ratio", f"{hi:.2f}", "", pill))
        parts.append("</div>")
        parts.append(_sparkline(expr, colour="#198038", label="expressivity"))
        parts.append('<p class="note">Expressivity is only interpretable relative to a prompted '
                     'task; a resting face is legitimately still.</p>')

    # --- blink ---
    rate = _series(meds, "blink", "rate_per_min")
    if any(v is not None for v in rate):
        parts.append("<h2>Blink</h2>")
        clean = [v for v in rate if v is not None]
        inc = [v for v in _series(meds, "blink", "incomplete_fraction") if v is not None]
        parts.append('<div class="cards">')
        parts.append(_card("Final blink rate", f"{clean[-1]:.0f}", "/min"))
        if inc:
            f = inc[-1]
            pill = ("bad", "high") if f > 0.4 else ("good", "normal")
            parts.append(_card("Incomplete blinks", f"{f*100:.0f}", "%", pill))
        parts.append("</div>")
        parts.append(_sparkline(rate, colour="#8a3ffc", label="blink rate (per minute)"))

    # --- heart rate ---
    bpm = [v for v in _series(meds, "heart_rate", "bpm") if v]
    if bpm:
        parts.append("<h2>Contactless heart rate (rPPG)</h2>")
        rel = [m.get("heart_rate", {}).get("reliable") for m in meds if "heart_rate" in m]
        rel_pct = 100.0 * sum(1 for r in rel if r) / len(rel) if rel else 0.0
        parts.append('<div class="cards">')
        parts.append(_card("Mean", f"{statistics.fmean(bpm):.0f}", "bpm"))
        parts.append(_card("Range", f"{min(bpm):.0f}-{max(bpm):.0f}", "bpm"))
        pill = ("good", "ok") if rel_pct > 60 else ("warn", "low")
        parts.append(_card("Reliable frames", f"{rel_pct:.0f}", "%", pill))
        hrv = [v for v in _series(meds, "heart_rate", "rmssd_ms") if v]
        if hrv:
            parts.append(_card("RMSSD", f"{statistics.fmean(hrv):.0f}", "ms"))
        parts.append("</div>")
        parts.append(_sparkline(_series(meds, "heart_rate", "bpm"),
                                colour="#da1e28", label="heart rate (bpm)"))
        parts.append('<p class="note">rPPG accuracy on real faces is unverified unless you have '
                     'measured it against a reference device (see scripts/validate_rppg.py). '
                     'Never treat these values as vital signs.</p>')

    parts.append(f"<footer>Generated by face-mesh. {_esc(DISCLAIMER)}</footer>")
    body = "\n".join(parts)
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_esc(title)}</title><style>{CSS}</style></head>"
            f"<body><div class='wrap'>{body}</div></body></html>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build an HTML session report.")
    ap.add_argument("session", nargs="?", help="exported .jsonl session")
    ap.add_argument("--protocol", default=None, help="protocol results JSON")
    ap.add_argument("-o", "--output", default="report.html")
    ap.add_argument("--pdf", default=None, metavar="FILE",
                    help="also write a PDF (needs weasyprint, wkhtmltopdf or reportlab)")
    ap.add_argument("--list-pdf-backends", action="store_true",
                    help="show which PDF backends are installed, then exit")
    ap.add_argument("--title", default="Facial metrics session")
    args = ap.parse_args(argv)

    if args.list_pdf_backends:
        found = available_backends()
        print("PDF backends available:", ", ".join(found) if found else "none")
        if not found:
            print(INSTALL_HINT)
        return 0

    if not args.session and not args.protocol:
        ap.error("give a session .jsonl and/or --protocol results")

    rows = load_session(Path(args.session)) if args.session else []
    protocol = json.loads(Path(args.protocol).read_text()) if args.protocol else None

    html = build_report(rows, protocol, args.title)
    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote report -> {out}")

    if args.pdf:
        try:
            pdf_path, backend = write_pdf(html, rows, protocol, args.pdf,
                                          args.title, DISCLAIMER)
        except PdfBackendUnavailable as exc:
            print(f"\nCould not write PDF.\n{exc}", file=sys.stderr)
            return 1
        print(f"Wrote PDF -> {pdf_path} (backend: {backend})")
        if backend == "reportlab":
            print("note: reportlab cannot render the HTML charts; the PDF is a "
                  "plain-text summary. Install weasyprint for full fidelity.")
    else:
        print("Open it in a browser; print to PDF if you need a document, "
              "or pass --pdf report.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
