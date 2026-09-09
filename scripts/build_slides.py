"""Build deliverables/Building-Age-Transfer.pptx from the presentation outline.

A designed 16:9 deck: single accent colour, one type scale, a hand-built flow
diagram, banded tables, slide numbers. Numbers come from
results/deliverable_table.csv so re-running notebook 5 then this script keeps the
deck in sync.

    python scripts/build_slides.py
"""

import csv
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
import sys
OUT = ROOT / "deliverables" / (sys.argv[1] if len(sys.argv) > 1 else "Building-Age-Transfer.pptx")
CURVE = ROOT / "results" / "transfer_curve.png"
TABLE_CSV = ROOT / "results" / "deliverable_table.csv"

# ── design tokens ────────────────────────────────────────────────────────
INK = RGBColor(0x1F, 0x24, 0x28)      # near-black text
MUT = RGBColor(0x5B, 0x66, 0x6B)      # muted grey text
ACC = RGBColor(0xC2, 0x4E, 0x00)      # single accent (burnt orange)
ACC_SOFT = RGBColor(0xF6, 0xE6, 0xDA)  # accent tint for fills
LINE = RGBColor(0xD8, 0xDE, 0xE0)     # hairlines / table grid
BG = RGBColor(0xFC, 0xFC, 0xFB)       # page
PANEL = RGBColor(0xF2, 0xF1, 0xEE)    # side panel / diagram box fill

TITLE_SZ, BODY_SZ, SMALL_SZ = Pt(30), Pt(17), Pt(13)
MARGIN = Inches(0.85)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]

_num = 0


def _no_line(shape):
    shape.line.fill.background()


def _fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color


def slide(section=None):
    global _num
    _num += 1
    s = prs.slides.add_slide(BLANK)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    _fill(bg, BG)
    _no_line(bg)
    bg.shadow.inherit = False
    # footer
    ft = s.shapes.add_textbox(MARGIN, SH - Inches(0.5), SW - 2 * MARGIN, Inches(0.32))
    p = ft.text_frame.paragraphs[0]
    p.text = f"Building-age transfer  ·  Madrid → Amsterdam" + (f"   |   {section}" if section else "")
    p.font.size = Pt(9)
    p.font.color.rgb = MUT
    num = s.shapes.add_textbox(SW - Inches(1.1), SH - Inches(0.5), Inches(0.7), Inches(0.32))
    q = num.text_frame.paragraphs[0]
    q.text = f"{_num:02d}"
    q.font.size = Pt(9)
    q.font.color.rgb = MUT
    q.alignment = PP_ALIGN.RIGHT
    return s


def heading(s, text, kicker=None):
    top = Inches(0.6)
    if kicker:
        kb = s.shapes.add_textbox(MARGIN, top, SW - 2 * MARGIN, Inches(0.32))
        kp = kb.text_frame.paragraphs[0]
        kp.text = kicker.upper()
        kp.font.size = Pt(11)
        kp.font.bold = True
        kp.font.color.rgb = ACC
        top = Inches(0.95)
    tb = s.shapes.add_textbox(MARGIN, top, SW - 2 * MARGIN, Inches(0.95))
    tp = tb.text_frame.paragraphs[0]
    tp.text = text
    tp.font.size = TITLE_SZ
    tp.font.bold = True
    tp.font.color.rgb = INK
    rule = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN, top + Inches(0.9),
                              Inches(0.9), Pt(3))
    _fill(rule, ACC)
    _no_line(rule)
    rule.shadow.inherit = False
    return top + Inches(1.25)


def bullets(s, items, left=None, top=Inches(2.15), width=None, size=BODY_SZ):
    left = left or MARGIN
    width = width or (SW - 2 * MARGIN)
    tb = s.shapes.add_textbox(left, top, width, SH - top - Inches(0.7))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        sub = it.startswith("- ")
        if sub:
            it = it[2:]
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        run = p.add_run()
        run.text = it
        p.font.size = Pt(size.pt - 3) if sub else size
        p.font.color.rgb = MUT if sub else INK
        p.level = 1 if sub else 0
        p.space_after = Pt(6 if sub else 11)
        p.line_spacing = 1.12
        # bullet glyph
        pPr = p._pPr if p._pPr is not None else p.get_or_add_pPr()
        for tag in ("a:buNone", "a:buChar", "a:buAutoNum"):
            for e in pPr.findall(qn(tag)):
                pPr.remove(e)
        bu = pPr.makeelement(qn("a:buChar"), {"char": "–" if sub else "▪"})
        pPr.append(bu)
    return tb


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


# ══ 1 — title ══════════════════════════════════════════════════════════
ABSTRACT = (
    "Predicting building construction era from 30 m Landsat imagery is hard, and "
    "harder across cities: a model trained on Madrid must work in Amsterdam, where "
    "building materials, climate and urban form all differ. We compress each "
    "pixel's 40-year, six-band reflectance series into 60 temporal statistics and "
    "train a class-balanced Random Forest on Madrid. Transfer rests on two "
    "unsupervised, label-free alignments: CORAL matches Madrid's feature "
    "covariance to Amsterdam's before training (zero-shot macro-F1 0.45 to 0.55), "
    "and budget-scaled ZCA whitening with a small local classifier, blended with "
    "the CORAL model, handles the few-shot regime. With 100 labelled Amsterdam "
    "pixels per class the transferred model matches Madrid's own in-city score "
    "(0.68 vs 0.63). Residual error concentrates on pre-1984 classes, which carry "
    "no construction event in the satellite record. Simple distribution alignment "
    "beat a learned embedding, ordinal loss, self-training and label-shift "
    "correction."
)
s = slide()
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.28), SH)
_fill(band, ACC)
_no_line(band)
band.shadow.inherit = False
kb = s.shapes.add_textbox(MARGIN, Inches(1.5), SW - 2 * MARGIN, Inches(0.4))
kp = kb.text_frame.paragraphs[0]
kp.text = "HACKATHON SUBMISSION"
kp.font.size = Pt(12)
kp.font.bold = True
kp.font.color.rgb = ACC
tb = s.shapes.add_textbox(MARGIN, Inches(2.0), SW - 2 * MARGIN, Inches(1.7))
tp = tb.text_frame.paragraphs[0]
tp.text = "A city-portable building-age classifier"
tp.font.size = Pt(40)
tp.font.bold = True
tp.font.color.rgb = INK
sp = tb.text_frame.add_paragraph()
sp.text = "from 40 years of Landsat  ·  Madrid → Amsterdam transfer"
sp.font.size = Pt(18)
sp.font.color.rgb = MUT
ab = s.shapes.add_textbox(MARGIN, Inches(3.9), SW - 2 * MARGIN, Inches(2.7))
ap = ab.text_frame
ap.word_wrap = True
ap.paragraphs[0].text = ABSTRACT
ap.paragraphs[0].font.size = Pt(13)
ap.paragraphs[0].font.color.rgb = INK
ap.paragraphs[0].line_spacing = 1.2
tm = s.shapes.add_textbox(MARGIN, SH - Inches(0.95), SW - 2 * MARGIN, Inches(0.4))
tm.text_frame.paragraphs[0].text = "Team <name>  ·  <member A>, <member B>, <member C>, <member D>"
tm.text_frame.paragraphs[0].font.size = Pt(12)
tm.text_frame.paragraphs[0].font.color.rgb = MUT
notes(s, "One line: a building-age classifier that ports to a new city on a few "
         "hundred labels. Don't read the abstract aloud.")

# ══ 2 — problem ══════════════════════════════════════════════════════
s = slide("Problem")
y = heading(s, "The problem", "why this is hard")
bullets(s, [
    "Task: classify the age class (1–4) of buildings in a 30 m Landsat pixel.",
    "It is possible because materials and weathering change a surface's spectral "
    "fingerprint over 40 years of imagery.",
    "The real challenge: generalise to a NEW city with few local labels.",
    "- Madrid → Amsterdam stands in for 'any city on Earth'.",
    "- Brick vs concrete, maritime vs Mediterranean, different urban form.",
    "The four classes split at the satellite record: 1 & 2 pre-date 1984 "
    "(finished before the cameras); 3 & 4 were built on-camera.",
], top=y)
notes(s, "Push the cross-city framing — that is what the rubric rewards. The "
         "1984 split returns on the results slide.")

# ══ 3 — features ═════════════════════════════════════════════════════
s = slide("Method")
y = heading(s, "Data → features", "one row per pixel")
bullets(s, [
    "Raw: one row per (pixel, year), up to 3 observations/year, ~40 years, 6 bands.",
    "Decision: collapse each pixel's series to 60 temporal statistics.",
    "- mean & std — overall, early (1984–2003), late (2004+), year-on-year change",
    "- 5 spectral indices (NDVI, NDBI, UI, MNDWI, BSI) + 2 coverage flags",
    "Per-year rows would leak temporal structure; summaries force the model to "
    "use trends and variability.",
    "The early/late split targets classes 3 & 4 — their construction event sits "
    "inside the record and shows as a contrast between the windows.",
], top=y)
notes(s, "One row per pixel is the single most important preprocessing choice.")

# ══ 4 — approach diagram ════════════════════════════════════════════
s = slide("Method")
y = heading(s, "Approach in one picture", "train once, transfer two ways")


def dbox(x, w, text, fill=PANEL, tcol=INK, y=y + Inches(0.15), h=Inches(0.95)):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    _fill(b, fill)
    b.line.color.rgb = LINE
    b.shadow.inherit = False
    tf = b.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(12)
    p.font.color.rgb = tcol
    return b


def arrow(a, b):
    cn = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                a.left + a.width, a.top + a.height // 2,
                                b.left, b.top + b.height // 2)
    cn.line.color.rgb = ACC
    cn.line.width = Pt(2)
    le = cn.line._get_or_add_ln()
    le.append(le.makeelement(qn("a:tailEnd"), {"type": "triangle"}))


row_y = y + Inches(0.15)
b1 = dbox(MARGIN, Inches(2.5), "Madrid features\n(60 per pixel)", y=row_y)
b2 = dbox(MARGIN + Inches(3.0), Inches(2.3), "CORAL align\nto Amsterdam", ACC_SOFT, y=row_y)
b3 = dbox(MARGIN + Inches(5.8), Inches(2.6), "Random Forest\nStage-1 model", y=row_y)
arrow(b1, b2)
arrow(b2, b3)

b4 = dbox(MARGIN + Inches(3.1), Inches(3.4),
          "0 labels  →  use the Stage-1 model directly  (zero-shot)",
          y=row_y + Inches(1.5), h=Inches(0.8))
b5 = dbox(MARGIN + Inches(3.1), Inches(7.0),
          "n labels  →  whiten Amsterdam (shrink ∝ n)  →  small RF on support "
          "→  blend with Stage-1",
          y=row_y + Inches(2.5), h=Inches(0.9))
for bb in (b4, b5):
    cn = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                b3.left + b3.width // 2, b3.top + b3.height,
                                bb.left + Inches(0.4), bb.top)
    cn.line.color.rgb = ACC
    cn.line.width = Pt(2)

cap = s.shapes.add_textbox(MARGIN, row_y + Inches(3.7), SW - 2 * MARGIN, Inches(0.6))
cp = cap.text_frame.paragraphs[0]
cp.text = ("Every alignment is unsupervised — covariances come from the UNLABELLED "
           "Amsterdam pool; the only target labels are the n-per-class support set.")
cp.font.size = Pt(12)
cp.font.italic = True
cp.font.color.rgb = MUT
cap.text_frame.word_wrap = True
notes(s, "The leakage point here pre-empts the biggest deduction on the rubric.")

# ══ 5 — CORAL ═══════════════════════════════════════════════════════
s = slide("Results")
y = heading(s, "Zero-shot: CORAL alignment", "no Amsterdam labels used")
bullets(s, [
    "Plain Madrid model on Amsterdam: macro-F1 0.45. The rules are right; the "
    "coordinates are Madrid's.",
    "CORAL: whiten Madrid's feature covariance, re-colour it with Amsterdam's, "
    "before training — boundaries are learned in the target's coordinates.",
    "Result:  0.45  →  0.55.  About 15 lines of linear algebra, zero labels.",
    "Recovers roughly half the Madrid–Amsterdam domain gap on its own.",
], top=y)
big = s.shapes.add_textbox(SW - Inches(4.4), Inches(2.4), Inches(3.6), Inches(2.4))
bp = big.text_frame.paragraphs[0]
bp.text = "0.45 → 0.55"
bp.font.size = Pt(40)
bp.font.bold = True
bp.font.color.rgb = ACC
bp.alignment = PP_ALIGN.CENTER
sp = big.text_frame.add_paragraph()
sp.text = "zero-shot macro-F1"
sp.font.size = Pt(13)
sp.font.color.rgb = MUT
sp.alignment = PP_ALIGN.CENTER
notes(s, "Analogy: same map, different grid reference. CORAL re-prints Madrid's "
         "data on Amsterdam's grid.")

# ══ 6 — few-shot mechanism ═════════════════════════════════════════
s = slide("Results")
y = heading(s, "Few-shot: whiten → local head → blend")
bullets(s, [
    "The 60 features are ~two-thirds redundant; nearest-prototype distance "
    "double-counts the redundant blocks.",
    "ZCA whitening untangles them — but needs data, so shrink toward plain "
    "scaling when labels are scarce (dial set by the budget n).",
    "A small Random Forest on the whitened support beats nearest-prototype (~+0.02).",
    "Blend its probabilities with the CORAL model, weight clip(n/50, 0.4, 0.95) "
    "toward the local head as n grows: +0.01 to +0.04, biggest at n = 5.",
], top=y)
notes(s, "Three cheap accounting fixes stacked — none of them a new model.")

# ══ 7 — results table ═════════════════════════════════════════════
s = slide("Results")
y = heading(s, "Results — macro-F1", "full 5×5 CV, fixed seed")
rows = list(csv.reader(TABLE_CSV.open()))
data = rows[1:]
tb = s.shapes.add_table(len(data) + 1, 3, MARGIN, y,
                        Inches(9.6), Inches(0.46) * (len(data) + 1)).table
tb.columns[0].width = Inches(5.8)
tb.columns[1].width = Inches(1.9)
tb.columns[2].width = Inches(1.9)
tb.first_row = False
for j, name in enumerate(["Setting", "macro-F1", "± std"]):
    c = tb.cell(0, j)
    c.text = name
    _fill(c, ACC)
    pr = c.text_frame.paragraphs[0]
    pr.font.size = Pt(13)
    pr.font.bold = True
    pr.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    if j:
        pr.alignment = PP_ALIGN.CENTER
for i, r in enumerate(data, start=1):
    key = ("few-shot" in r[0]) or ("CORAL" in r[0])
    for j, val in enumerate(r):
        c = tb.cell(i, j)
        c.text = val if val else "—"
        _fill(c, ACC_SOFT if key else (PANEL if i % 2 else BG))
        pr = c.text_frame.paragraphs[0]
        pr.font.size = Pt(12)
        pr.font.color.rgb = INK
        pr.font.bold = bool(key and j == 1)
        if j:
            pr.alignment = PP_ALIGN.CENTER
notes(s, "Say the numbers WITH their error bars. Walk zero-shot raw → CORAL, "
         "then the climb across the five budgets.")

# ══ 8 — plot ═════════════════════════════════════════════════════
s = slide("Results")
y = heading(s, "F1 vs. labels per class")
if CURVE.exists():
    s.shapes.add_picture(str(CURVE), MARGIN, y, height=Inches(4.5))
bullets(s, [
    "Steep rise to ~50 labels, then a plateau.",
    "The plateau sits at the Madrid in-city score — by 100 labels/class the "
    "transferred model is as good on Amsterdam as any model is at home.",
    "Error bars widest at n = 5 (±0.02): tiny support, unstable class means.",
    "Low-data point (25/class) is only ~0.04 below the plateau.",
], left=Inches(7.6), top=y, width=Inches(5.0), size=Pt(14))
notes(s, "Lead the whole talk with THIS story: few labels close the gap. "
         "Everything else is how.")

# ══ 9 — what didn't work ═════════════════════════════════════════
s = slide("Findings")
y = heading(s, "What did not work — and why", "and it is worth saying so")
fails = [
    ("Triplet-loss embedding", "features already linearly separable — nothing to learn"),
    ("Ordinal loss", "shrinks error size, not right-vs-wrong; macro-F1 unmoved"),
    ("Label-shift (EM) correction", "needs calibrated probabilities; domain gap breaks it (0.54→0.44)"),
    ("Self-training on 25k unlabelled", "~35% pseudo-label error at low n, compounds each round"),
    ("Gradient boosting", "ties RF in-city, collapses at 5 labels"),
]
tb = s.shapes.add_table(len(fails) + 1, 2, MARGIN, y, Inches(11.4),
                        Inches(0.5) * (len(fails) + 1)).table
tb.columns[0].width = Inches(3.9)
tb.columns[1].width = Inches(7.5)
for j, h in enumerate(["Tried", "Why it failed"]):
    c = tb.cell(0, j)
    c.text = h
    _fill(c, ACC)
    p = c.text_frame.paragraphs[0]
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
for i, (a, b) in enumerate(fails, start=1):
    for j, v in enumerate((a, b)):
        c = tb.cell(i, j)
        c.text = v
        _fill(c, PANEL if i % 2 else BG)
        p = c.text_frame.paragraphs[0]
        p.font.size = Pt(11.5)
        p.font.color.rgb = INK
        p.font.bold = j == 0
capt = s.shapes.add_textbox(MARGIN, y + Inches(3.4), Inches(11.4), Inches(0.5))
cp = capt.text_frame.paragraphs[0]
cp.text = "Theme: every attempt to be cleverer than the data lost; every accounting fix won."
cp.font.size = Pt(13)
cp.font.italic = True
cp.font.color.rgb = ACC
notes(s, "Own these confidently and briefly — the originality / insightful-"
         "failure points on the rubric.")

# ══ 10 — limitations ════════════════════════════════════════════
s = slide("Findings")
y = heading(s, "Limitations & honesty")
bullets(s, [
    "Classes 1 vs 2 (both pre-1984) are the main residual error — no "
    "construction event to separate them. A data limit, matching the pre-war "
    "building-age literature.",
    "Spatial-context features help in-city (+0.04 Madrid CV) but break raw "
    "transfer — deliberately left out of the final pipeline.",
    "All numbers: full 5×5 CV, fixed seed; dev-time quick runs agreed to ±0.01.",
    "Next: hyper-parameter tuning, per-decade features, a dedicated class-1/2 model.",
], top=y)
notes(s, "Showing you know the ceiling earns marks. Do not over-claim.")

# ══ 11 — team ═══════════════════════════════════════════════════
s = slide()
y = heading(s, "Team & contributions")
bullets(s, [
    "<member A> — feature engineering, data pipeline  (src/data.py)",
    "<member B> — domain alignment: CORAL, whitening  (src/adapt.py)",
    "<member C> — evaluation harness, few-shot curve, self-training  (src/evaluate.py)",
    "<member D> — results analysis, written justification, presentation",
    "All experiments reproducible: scripts/run_*.py, one fixed seed.",
], top=y)
notes(s, "Every member speaks to their own slide at least once — the rubric "
         "checks this explicitly.")

OUT.parent.mkdir(exist_ok=True)
prs.save(str(OUT))
print("wrote", OUT, "-", _num, "slides")
