"""Generate report/report.pdf from report.md (preview-only).

This is a fallback for environments without LaTeX.  The canonical
report is the LaTeX source `report/report.tex`; the PDF produced here
is intentionally simple (single column, basic typography) and is
intended as a quick preview while the LaTeX file is being compiled
elsewhere.

Run with:
    python scripts/build_report_pdf.py
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
    Table, TableStyle, KeepTogether,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "report" / "report.md"
OUT = PROJECT_ROOT / "report" / "report.pdf"
FIG = PROJECT_ROOT / "report" / "figures"


# Map markdown image references to figure files used in the report body.
SECTION_FIGURES = {
    "3.3": [("01_class_balance.png", 3.0, "Class balance and marginal rates."),
            ("02_marginal_rates.png", 5.5, None)],
    "4.3": [("05_pr_curves.png", 3.5, "PR curves and calibration."),
            ("06_calibration.png", 6.0, None)],
    "4.4": [("08_feature_importance.png", 6.8, "Top-20 feature importances (LR coefficients vs RF importances).")],
    "5.2": [("14_persona_heatmap.png", 6.5, "Geography-free persona z-scores (k=5).")],
    "6":   [("15_persona_mix_at_cutoffs.png", 4.0, "Persona composition at each top-X% cutoff."),
            ("16_precision_recall_vs_cutoff.png", 3.4, "Precision and recall vs. cutoff.")],
}


def _styles():
    base = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica",
                          fontSize=9.5, leading=12, spaceAfter=4, alignment=4)
    title = ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold",
                           fontSize=15, leading=18, spaceAfter=4, alignment=1)
    subtitle = ParagraphStyle("subtitle", parent=base["BodyText"], fontName="Helvetica-Oblique",
                              fontSize=10, leading=12, spaceAfter=10, alignment=1)
    h1 = ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold",
                        fontSize=12, leading=14, spaceBefore=10, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold",
                        fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=2)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=2)
    return {"body": body, "title": title, "subtitle": subtitle,
            "h1": h1, "h2": h2, "bullet": bullet}


_INLINE = [
    (re.compile(r"\*\*(.+?)\*\*"), r"<b>\1</b>"),
    (re.compile(r"(?<!\*)\*([^*\n]+)\*"), r"<i>\1</i>"),
    (re.compile(r"`([^`]+)`"), r'<font face="Courier">\1</font>'),
]


def _md_inline(text: str) -> str:
    out = text
    for pat, repl in _INLINE:
        out = pat.sub(repl, out)
    out = out.replace("&", "&amp;").replace("<b>", "<b>").replace("</b>", "</b>")
    # The replacement above is a no-op but keeps the intent that we let our
    # tag substitutions through reportlab; we just guard ampersands.
    return out


def _table_from_lines(lines: list[str], styles) -> Table:
    rows = [[c.strip() for c in line.strip().strip("|").split("|")]
            for line in lines if line.strip()]
    # Drop the alignment row (e.g., "|------|").
    rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
    tbl = Table(rows, hAlign="CENTER")
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOX",      (0, 0), (-1, -1), 0.4, colors.black),
        ("LINEBELOW",(0, 0), (-1, 0), 0.6, colors.black),
        ("BACKGROUND",(0,0), (-1, 0), colors.HexColor("#EEEEEE")),
        ("ALIGN",    (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN",    (0, 0), (0, -1), "LEFT"),
        ("VALIGN",   (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING",(0,0), (-1,-1), 4),
        ("TOPPADDING",  (0,0), (-1,-1), 2),
        ("BOTTOMPADDING",(0,0),(-1,-1), 2),
    ]))
    return tbl


def _figure_block(fig_files, styles) -> list:
    blocks = []
    for fname, width_in, caption in fig_files:
        path = FIG / fname
        if not path.exists():
            continue
        img = Image(str(path), width=width_in * inch,
                    height=width_in * inch * _aspect(path))
        blocks.append(img)
        if caption:
            blocks.append(Paragraph(f"<i>Figure: {caption}</i>", styles["body"]))
        blocks.append(Spacer(1, 6))
    return blocks


def _aspect(path: Path) -> float:
    """Return image height/width ratio without loading PIL.

    Falls back to 0.65 (a typical landscape figure) when we cannot
    introspect the file.
    """
    try:
        from PIL import Image as PILImage  # noqa
        with PILImage.open(path) as im:
            w, h = im.size
            return h / w
    except Exception:
        return 0.65


def _emit_section(section_id: str, body: list, styles, story) -> None:
    figs = SECTION_FIGURES.get(section_id)
    if figs:
        story.extend(_figure_block(figs, styles))
    story.extend(body)


def render() -> None:
    text = SRC.read_text(encoding="utf-8")
    lines = text.splitlines()
    styles = _styles()

    doc = SimpleDocTemplate(
        str(OUT), pagesize=letter,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title="EE 5290 Final Project Report",
    )
    story = []

    paragraph_buffer: list[str] = []
    in_list: list[str] = []
    table_buffer: list[str] = []
    current_section = "0"

    def flush_paragraph():
        if paragraph_buffer:
            text = " ".join(paragraph_buffer).strip()
            if text:
                story.append(Paragraph(_md_inline(text), styles["body"]))
            paragraph_buffer.clear()

    def flush_list():
        if in_list:
            for item in in_list:
                story.append(Paragraph(f"- {_md_inline(item)}", styles["bullet"]))
            in_list.clear()

    def flush_table():
        if table_buffer:
            story.append(_table_from_lines(table_buffer, styles))
            story.append(Spacer(1, 6))
            table_buffer.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            flush_list()
            flush_table()
        elif stripped.startswith("# "):
            flush_paragraph(); flush_list(); flush_table()
            story.append(Paragraph(_md_inline(stripped[2:]), styles["title"]))
        elif stripped.startswith("## "):
            flush_paragraph(); flush_list(); flush_table()
            heading = stripped[3:]
            m = re.match(r"^(\d+)\. ", heading)
            if m:
                current_section = m.group(1)
            story.append(Paragraph(_md_inline(heading), styles["h1"]))
        elif stripped.startswith("### "):
            flush_paragraph(); flush_list(); flush_table()
            heading = stripped[4:]
            m = re.match(r"^(\d+\.\d+)", heading)
            if m:
                current_section = m.group(1)
            story.append(Paragraph(_md_inline(heading), styles["h2"]))
            # Insert section figures (if any) immediately after a sub-heading.
            figs = SECTION_FIGURES.get(current_section)
            if figs:
                story.extend(_figure_block(figs, styles))
                SECTION_FIGURES.pop(current_section)
        elif stripped.startswith("|"):
            flush_paragraph(); flush_list()
            table_buffer.append(stripped)
        elif stripped.startswith("- "):
            flush_paragraph(); flush_table()
            in_list.append(stripped[2:])
        elif stripped.startswith("---"):
            flush_paragraph(); flush_list(); flush_table()
            story.append(Spacer(1, 6))
        elif re.match(r"^\d+\.\s", stripped):
            flush_paragraph(); flush_table()
            in_list.append(stripped.split(". ", 1)[1])
        elif stripped.startswith("**EE 5290"):
            story.append(Paragraph(_md_inline(stripped), styles["subtitle"]))
        else:
            paragraph_buffer.append(stripped)

        # Section figure blocks for sections without a sub-heading (Section 6).
        if current_section in SECTION_FIGURES and stripped.startswith("**Operational"):
            story.extend(_figure_block(SECTION_FIGURES.pop(current_section), styles))

        i += 1
    flush_paragraph(); flush_list(); flush_table()

    doc.build(story)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    render()
