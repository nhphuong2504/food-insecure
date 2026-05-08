"""Generate report/slides.pdf -- a presentation deck for the EE 5290 final.

The deck mirrors the report structure but is paced for a ~12-15 minute
talk: title, problem framing, data, prediction track results,
segmentation track results, synthesis, and a conclusion / Q&A slide.

We use reportlab (already installed for the report preview) so the
deck can be rebuilt without LaTeX or Beamer.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle,
)
from reportlab.pdfgen import canvas


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "report" / "slides.pdf"
FIG = PROJECT_ROOT / "report" / "figures"

PAGE_SIZE = landscape(letter)  # 11 x 8.5 in


# --- styles ---------------------------------------------------------------

def styles():
    base = getSampleStyleSheet()
    return {
        "title":    ParagraphStyle("title",   parent=base["Title"],
                                   fontName="Helvetica-Bold", fontSize=24,
                                   leading=28, alignment=1, spaceAfter=8),
        "subtitle": ParagraphStyle("sub",     parent=base["BodyText"],
                                   fontName="Helvetica-Oblique", fontSize=14,
                                   leading=18, alignment=1, spaceAfter=12),
        "h1":       ParagraphStyle("h1",      parent=base["Heading1"],
                                   fontName="Helvetica-Bold", fontSize=20,
                                   leading=24, spaceAfter=10, spaceBefore=2),
        "body":     ParagraphStyle("body",    parent=base["BodyText"],
                                   fontName="Helvetica", fontSize=14,
                                   leading=18, spaceAfter=4),
        "bullet":   ParagraphStyle("bullet",  parent=base["BodyText"],
                                   fontName="Helvetica", fontSize=14,
                                   leading=18, leftIndent=18, bulletIndent=2,
                                   spaceAfter=2),
        "caption":  ParagraphStyle("caption", parent=base["BodyText"],
                                   fontName="Helvetica-Oblique", fontSize=11,
                                   leading=14, alignment=1),
        "small":    ParagraphStyle("small",   parent=base["BodyText"],
                                   fontName="Helvetica", fontSize=11,
                                   leading=14, spaceAfter=4),
    }


# --- helpers --------------------------------------------------------------

def _fig(name: str, width_in: float, caption: str | None = None) -> list:
    path = FIG / name
    out: list = []
    if path.exists():
        try:
            from PIL import Image as PILImage
            with PILImage.open(path) as im:
                w, h = im.size
                aspect = h / w
        except Exception:
            aspect = 0.65
        out.append(Image(str(path), width=width_in * inch,
                         height=width_in * inch * aspect))
    if caption:
        out.append(Paragraph(caption, styles()["caption"]))
    return out


def _bullets(items: list[str], style) -> list:
    return [Paragraph(f"&bull;&nbsp;&nbsp;{x}", style) for x in items]


def _two_col_image_text(image_name: str, image_w: float, text_blocks: list) -> Table:
    img_blocks = _fig(image_name, image_w)
    text_cell = list(text_blocks)
    return Table([[img_blocks, text_cell]],
                 colWidths=[image_w * inch + 0.2 * inch, None],
                 style=TableStyle([
                     ("VALIGN", (0, 0), (-1, -1), "TOP"),
                     ("LEFTPADDING", (0, 0), (-1, -1), 4),
                     ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                 ]))


# --- slides ---------------------------------------------------------------

def slide_title(s):
    body = [
        Spacer(1, 1.2 * inch),
        Paragraph("Predictive Targeting and Segmentation of Food-Insecure Households",
                  s["title"]),
        Spacer(1, 0.15 * inch),
        Paragraph("EE 5290 Final Project &middot; Spring 2026", s["subtitle"]),
        Spacer(1, 0.1 * inch),
        Paragraph("Iowa State University &middot; May 2026", s["subtitle"]),
    ]
    return body


def slide_problem(s):
    return [
        Paragraph("The Problem", s["h1"]),
        Spacer(1, 0.05 * inch),
        Paragraph(
            "Food-assistance demand is rising while organisational capacity is fixed. "
            "Food banks must decide <b>which households to prioritize</b> and "
            "<b>what programs to offer</b>.",
            s["body"]),
        Spacer(1, 0.1 * inch),
        Paragraph("Two questions:", s["body"]),
        *_bullets([
            "<b>Prediction:</b> Can we identify food-insecure households accurately enough to support triage?",
            "<b>Segmentation:</b> What <i>kinds</i> of food-insecure households exist, and how should that shape outreach?",
        ], s["bullet"]),
        Spacer(1, 0.15 * inch),
        Paragraph("Approach: a two-track ML analysis joined by a synthesis section.", s["body"]),
    ]


def slide_data(s):
    text = [
        Paragraph("Data", s["h1"]),
        *_bullets([
            "Provided FoodAPS extract: <b>4,826 households</b>, 38 columns",
            "Outcome: <code>food_insecure_flag_adult</code> (binary)",
            "<b>27.85% positive</b> &mdash; moderate imbalance",
            "Sentinel codes (-996/-997/-998) recoded; <code>caraccess</code> dropped (92% missing)",
            "Distance variables logged; one-hot encoding for nominals",
            "Stratified 80/20 split; 5-fold CV on train only",
        ], s["bullet"]),
    ]
    return [_two_col_image_text("01_class_balance.png", 4.6, text)]


def slide_eda(s):
    return [
        Paragraph("EDA: Where the Risk Lives", s["h1"]),
        Spacer(1, 0.05 * inch),
        *_fig("02_marginal_rates.png", 8.5),
        Paragraph(
            "Poverty band, employment, SNAP, and education move the food-insecurity rate"
            " by 20-30 percentage points each. Rural households are slightly elevated.",
            s["caption"]),
    ]


def slide_pred_setup(s):
    return [
        Paragraph("Track 1 &mdash; Predictive Modelling Setup", s["h1"]),
        *_bullets([
            "Three classifiers: <b>Logistic Regression</b>, <b>SVM (RBF)</b>, <b>Random Forest</b>",
            "Three imbalance strategies: <b>none</b>, <b>balanced class weights</b>, <b>F1-tuned threshold</b>",
            "Tuning metric: <b>average precision (PR-AUC)</b>",
            "Threshold tuned on training out-of-fold predictions (test set untouched)",
        ], s["bullet"]),
        Spacer(1, 0.15 * inch),
        Paragraph("Why PR-AUC? With 28% positives, accuracy is misleading "
                  "(72% trivially). PR-AUC ignores the abundant true negatives.",
                  s["body"]),
    ]


def slide_pred_table(s):
    rows = [
        ["Model (strategy)", "thr", "PR-AUC", "Prec.", "Rec.", "F1"],
        ["LR -- no adjust",            "0.50",  "0.568", "0.624", "0.364", "0.460"],
        ["LR -- balanced",             "0.50",  "0.568", "0.463", "0.762", "0.576"],
        ["LR -- F1-tuned thr",         "0.289", "0.568", "0.466", "0.755", "0.576"],
        ["SVM-RBF -- no adjust",       "0.50",  "0.529", "0.549", "0.208", "0.302"],
        ["SVM-RBF -- balanced",        "0.50",  "0.527", "0.564", "0.346", "0.429"],
        ["SVM-RBF -- F1-tuned thr",    "0.236", "0.529", "0.425", "0.747", "0.542"],
        ["RF -- no adjust",            "0.50",  "0.521", "0.602", "0.186", "0.284"],
        ["RF -- balanced",             "0.50",  "0.510", "0.485", "0.665", "0.561"],
        ["RF -- F1-tuned thr",         "0.301", "0.521", "0.453", "0.792", "0.577"],
    ]
    tbl = Table(rows, hAlign="CENTER",
                colWidths=[2.6 * inch, 0.7 * inch, 0.9 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch])
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("BOX",      (0, 0), (-1, -1), 0.5, colors.black),
        ("LINEBELOW",(0, 0), (-1, 0), 0.6, colors.black),
        ("BACKGROUND",(0,0), (-1, 0), colors.HexColor("#EEEEEE")),
        ("ALIGN",    (1, 1), (-1, -1), "RIGHT"),
        ("VALIGN",   (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING",(0,0), (-1,-1), 5),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
    ]))
    return [
        Paragraph("Track 1 &mdash; Test Set Results (9 cells)", s["h1"]),
        Spacer(1, 0.1 * inch),
        tbl,
        Spacer(1, 0.18 * inch),
        Paragraph(
            "<b>LR wins on every aggregate metric.</b> Threshold tuning &asymp; class weights. "
            "The choice of <i>threshold</i> matters more than the choice of <i>weighting</i>.",
            s["body"]),
    ]


def slide_pred_curves(s):
    return [
        Paragraph("Track 1 &mdash; PR Curves and Calibration", s["h1"]),
        _two_col_image_text("05_pr_curves.png", 4.5, [
            *_fig("06_calibration.png", 5.0),
        ]),
        Paragraph(
            "Left: LR dominates SVM and RF in the high-recall regime that matters for triage. "
            "Right: LR is well calibrated; RF mildly overconfident at the extremes.",
            s["caption"]),
    ]


def slide_pred_features(s):
    return [
        Paragraph("Track 1 &mdash; What the Models Look At", s["h1"]),
        *_fig("08_feature_importance.png", 9.5),
        Paragraph(
            "LR (left, signed): poverty band, food-pantry use, vehicles, education, distances. "
            "RF (right, unsigned) emphasises the same features plus the cluster of distance variables.",
            s["caption"]),
    ]


def slide_seg_overview(s):
    return [
        Paragraph("Track 2 &mdash; Segmentation Setup", s["h1"]),
        *_bullets([
            "Filter to the 1,344 food-insecure households",
            "Standardize &rarr; PCA (24 PCs explain 80% of variance)",
            "K-Means sweep <code>k=2..8</code>; choose by silhouette + elbow",
            "Stability check across 8 random initializations (mean ARI &gt; 0.9)",
        ], s["bullet"]),
        Spacer(1, 0.1 * inch),
        Paragraph("Two views:", s["body"]),
        *_bullets([
            "<b>Geographic</b> (all features): silhouette winner is <b>k=2</b>, a strong rural / urban access split.",
            "<b>Geography-free</b> (drop distances/region): exposes <b>5 personas</b> that differ on composition, employment, programs.",
        ], s["bullet"]),
    ]


def slide_seg_personas(s):
    return [
        Paragraph("Track 2 &mdash; Five Personas Among the Food Insecure", s["h1"]),
        *_fig("14_persona_heatmap.png", 9.5),
        Paragraph(
            "Z-scores within the food-insecure subset. Red = above the food-insecure mean.",
            s["caption"]),
    ]


def slide_persona_table(s):
    rows = [
        ["P", "Persona", "n", "Share", "Headline feature"],
        ["P0", "Working families just above poverty",          "266", "20%", "100% employed; pov. ratio 1.43"],
        ["P1", "Non-working low-income singles",                "391", "29%", "Pantry use 28% (3x avg); 16% emp."],
        ["P2", "Large low-income families with children",       "347", "26%", "Mean household 5.0; 2.6 kids; pov. 0.85"],
        ["P3", "Working educated, near boundary",               "214", "16%", "Pov. ratio 3.05 (above poverty!)"],
        ["P4", "Elderly low-income, high SNAP",                 "126", "9%",  "Mean head age 70; 42% on SNAP"],
    ]
    tbl = Table(rows, hAlign="CENTER",
                colWidths=[0.5*inch, 3.5*inch, 0.7*inch, 0.7*inch, 4.0*inch])
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("BOX",      (0, 0), (-1, -1), 0.5, colors.black),
        ("LINEBELOW",(0, 0), (-1, 0), 0.6, colors.black),
        ("BACKGROUND",(0,0), (-1, 0), colors.HexColor("#EEEEEE")),
        ("VALIGN",   (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",    (2, 1), (3, -1), "RIGHT"),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING",(0,0), (-1,-1), 5),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]))
    return [Paragraph("Track 2 &mdash; The Five Personas", s["h1"]), Spacer(1, 0.1*inch), tbl]


def slide_synthesis(s):
    return [
        Paragraph("Synthesis &mdash; Who Does the Model Flag?", s["h1"]),
        _two_col_image_text("15_persona_mix_at_cutoffs.png", 5.5, [
            *_fig("16_precision_recall_vs_cutoff.png", 4.0),
        ]),
        Paragraph(
            "<b>Top 10% predicted-risk: 88% of flags are P1 or P2.</b> &nbsp; "
            "P3 (working educated) and P4 (elderly on SNAP) get &lt;2% of flags.",
            s["body"]),
    ]


def slide_recommendations(s):
    return [
        Paragraph("Targeting Recommendations", s["h1"]),
        *_bullets([
            "<b>Top decile:</b> precision 67%, recall 24% &rarr; 97 of 966 households; 66 truly food-insecure",
            "<b>P1 (45%) &mdash; non-working low-income singles:</b> pantry partnerships, supplemental distribution",
            "<b>P2 (43%) &mdash; large families with children:</b> school-meal coordination, bulk-pack programmes",
            "<b>P0 (12%) &mdash; working families just above poverty:</b> light-touch SNAP-enrolment outreach",
            "<b>P4 &mdash; elderly on SNAP:</b> monitor via existing partnerships, do not disengage on low scores",
            "<b>P3 &mdash; working educated:</b> qualitative follow-up; not reachable by income-eligibility programmes",
        ], s["bullet"]),
        Spacer(1, 0.15 * inch),
        Paragraph(
            "<i>All findings are associations, not causal claims. Survey weights and"
            " external county-level data are obvious next steps.</i>", s["caption"]),
    ]


def slide_qa(s):
    return [
        Spacer(1, 1.2 * inch),
        Paragraph("Thank you. Questions?", s["title"]),
        Spacer(1, 0.4 * inch),
        Paragraph(
            "Code &amp; figures: <code>notebooks/</code>, <code>src/</code>, <code>report/</code>",
            s["subtitle"]),
        Spacer(1, 0.1 * inch),
        Paragraph(
            "Reproduction: <code>python -m venv venv &amp;&amp; pip install -r requirements.txt</code>",
            s["subtitle"]),
    ]


# --- driver ---------------------------------------------------------------

def render() -> None:
    s = styles()
    doc = SimpleDocTemplate(
        str(OUT), pagesize=PAGE_SIZE,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.5 * inch, bottomMargin=0.5 * inch,
        title="EE 5290 Final Project Slides",
    )
    slides = [
        slide_title,
        slide_problem,
        slide_data,
        slide_eda,
        slide_pred_setup,
        slide_pred_table,
        slide_pred_curves,
        slide_pred_features,
        slide_seg_overview,
        slide_seg_personas,
        slide_persona_table,
        slide_synthesis,
        slide_recommendations,
        slide_qa,
    ]
    story = []
    for i, sl in enumerate(slides):
        story.extend(sl(s))
        if i < len(slides) - 1:
            story.append(PageBreak())
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(f"wrote {OUT}")


def _footer(canv: canvas.Canvas, doc):
    canv.saveState()
    canv.setFont("Helvetica", 8)
    canv.setFillColor(colors.grey)
    page_w, page_h = PAGE_SIZE
    canv.drawRightString(page_w - 0.4 * inch, 0.25 * inch,
                          f"EE 5290 Final - Slide {doc.page}")
    canv.restoreState()


if __name__ == "__main__":
    render()
