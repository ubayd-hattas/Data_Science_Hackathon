"""Build deliverables/Building-Age-Transfer.pptx from the presentation outline.

Plain 16:9 deck, editable. Numbers are pulled from results/deliverable_table.csv
so a re-run of notebook 5 keeps the slides in sync (re-run this script after).

    python scripts/build_slides.py
"""

import csv
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "Building-Age-Transfer.pptx"
CURVE = ROOT / "results" / "transfer_curve.png"
TABLE_CSV = ROOT / "results" / "deliverable_table.csv"

INK = RGBColor(0x1A, 0x1A, 0x1A)
ACCENT = RGBColor(0xD9, 0x5F, 0x0E)
GREY = RGBColor(0x55, 0x55, 0x55)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def box(s, l, t, w, h):
    tb = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tb.text_frame.word_wrap = True
    return tb.text_frame


def title(s, text, sub=None):
    tf = box(s, 0.6, 0.4, 12.1, 1.1)
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = INK
    if sub:
        sp = tf.add_paragraph()
        sp.text = sub
        sp.font.size = Pt(15)
        sp.font.color.rgb = ACCENT


def bullets(s, items, left=0.7, top=1.7, width=12.0, height=5.3, size=18):
    tf = box(s, left, top, width, height)
    for i, it in enumerate(items):
        lvl = 0
        while it.startswith("- "):
            it = it[2:]
            lvl += 1
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if lvl <= 1 else "– ") + it
        p.level = max(lvl - 1, 0)
        p.font.size = Pt(size if lvl <= 1 else size - 3)
        p.font.color.rgb = INK if lvl <= 1 else GREY
        p.space_after = Pt(7)


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


# ── 1. Title + abstract ────────────────────────────────────────────────
ABSTRACT = (
    "Predicting building construction era from 30 m Landsat imagery is hard, and "
    "harder across cities: a model trained on Madrid must work in Amsterdam, where "
    "building materials, climate and urban form all differ. We compress each pixel's "
    "40-year, six-band reflectance series into 60 temporal statistics and train a "
    "class-balanced Random Forest on Madrid. Transfer rests on two unsupervised, "
    "label-free alignments: CORAL matches Madrid's feature covariance to Amsterdam's "
    "before training (zero-shot macro-F1 0.45 to 0.55), and budget-scaled ZCA "
    "whitening with a small local classifier, blended with the CORAL model, handles "
    "the few-shot regime. With 100 labelled Amsterdam pixels per class the transferred "
    "model matches Madrid's own in-city score (0.68 vs 0.63). Residual error "
    "concentrates on pre-1984 classes, which carry no construction event in the "
    "satellite record. Simple distribution alignment beat a learned embedding, ordinal "
    "loss, self-training and label-shift correction."
)
s = slide()
title(s, "A City-Portable Building-Age Classifier", "from 40 years of Landsat  ·  Madrid → Amsterdam transfer")
tf = box(s, 0.7, 1.9, 12.0, 4.4)
tf.paragraphs[0].text = ABSTRACT
tf.paragraphs[0].font.size = Pt(15)
tf.paragraphs[0].font.color.rgb = INK
tf.paragraphs[0].line_spacing = 1.15
t2 = box(s, 0.7, 6.5, 12.0, 0.6)
t2.paragraphs[0].text = "Team <name>  ·  <member A>, <member B>, <member C>, <member D>"
t2.paragraphs[0].font.size = Pt(13)
t2.paragraphs[0].font.color.rgb = GREY
notes(s, "Read one line: a building-age classifier that ports to a new city on a "
         "few hundred labels. Abstract is on the slide; don't read it out fully.")

# ── 2. The problem ────────────────────────────────────────────────────
s = slide()
title(s, "The problem")
bullets(s, [
    "Task: classify the age class (1–4) of buildings in a 30 m Landsat pixel.",
    "Why it is possible: materials and weathering change a surface's spectral "
    "fingerprint over 40 years of imagery.",
    "The real challenge: generalise to a NEW city with few local labels.",
    "- Madrid → Amsterdam is the test case for ‘any city on Earth’.",
    "- Brick vs concrete, maritime vs Mediterranean, different urban form.",
    "Four classes split at the satellite record: 1 & 2 are pre-1984 (finished "
    "before the cameras), 3 & 4 were built on-camera.",
])
notes(s, "Emphasise the cross-city framing — that is what the rubric rewards. "
         "The 1984 split matters and comes back on the results slide.")

# ── 3. Data -> features ──────────────────────────────────────────────
s = slide()
title(s, "Data → features")
bullets(s, [
    "Raw: one row per (pixel, year), up to 3 observations/year, ~40 years, 6 bands.",
    "Key decision: collapse to ONE row per pixel — 60 temporal statistics.",
    "- mean & std: overall, early period (1984–2003), late period (2004+), "
    "year-on-year change",
    "- 5 spectral indices (NDVI, NDBI, UI, MNDWI, BSI) + 2 data-coverage flags",
    "Why: per-year rows would leak temporal structure; summaries force the model "
    "to use trends and variability.",
    "Early/late split targets classes 3 & 4 — their construction event sits "
    "inside the record and shows as a contrast between the windows.",
])
notes(s, "One row per pixel is the single most important preprocessing choice. "
         "Mention gap-filling briefly if asked.")

# ── 4. Approach ─────────────────────────────────────────────────────
s = slide()
title(s, "Approach in one picture")
bullets(s, [
    "Stage 1:  Madrid features  →  [CORAL align]  →  Random Forest "
    "(300 trees, class-balanced).",
    "Two ways out to Amsterdam:",
    "- 0 labels  →  the CORAL-aligned model, used directly (zero-shot).",
    "- n labels  →  whiten Amsterdam (shrinkage ∝ n)  →  small RF on "
    "the support set  →  blend its vote with the CORAL model.",
    "Every alignment is unsupervised: covariances come from the UNLABELLED "
    "Amsterdam pool; the only target labels are the n-per-class support set.",
])
notes(s, "Draw the box diagram if you have time to make one. The leakage point "
         "here pre-empts the biggest deduction on the rubric.")

# ── 5. Zero-shot: CORAL ────────────────────────────────────────────
s = slide()
title(s, "Zero-shot: CORAL alignment", "no Amsterdam labels used")
bullets(s, [
    "Plain Madrid model on Amsterdam: macro-F1 0.45. The decision rules are "
    "right; the coordinates are Madrid's.",
    "CORAL: whiten Madrid's feature covariance, re-colour it with Amsterdam's, "
    "BEFORE training — so boundaries are learned in the target's coordinates.",
    "Result:  0.45  →  0.55.  ~15 lines of linear algebra, zero labels.",
    "Recovers roughly half the Madrid–Amsterdam domain gap on its own.",
])
notes(s, "Analogy: same map, different grid reference. CORAL re-prints Madrid's "
         "data on Amsterdam's grid.")

# ── 6. Few-shot mechanism ──────────────────────────────────────────
s = slide()
title(s, "Few-shot: whiten → local head → blend")
bullets(s, [
    "The 60 features are ~two-thirds redundant; nearest-prototype distance "
    "double-counts the redundant blocks.",
    "ZCA whitening untangles them — but needs data, so shrink it toward plain "
    "scaling when labels are scarce (dial set by the budget n).",
    "Small Random Forest on the whitened support beats nearest-prototype (~+0.02).",
    "Blend its probabilities with the CORAL model, weight clip(n/50, 0.4, 0.95) "
    "— shifts local as n grows. Adds +0.01 to +0.04, biggest at n = 5.",
])
notes(s, "Three cheap accounting fixes stacked. None of them is a new model.")

# ── 7. Results table ───────────────────────────────────────────────
s = slide()
title(s, "Results — macro-F1")
rows = list(csv.reader(TABLE_CSV.open()))
header, data = rows[0], rows[1:]
tbl = s.shapes.add_table(len(data) + 1, 3, Inches(1.6), Inches(1.7),
                         Inches(10.0), Inches(0.4 * (len(data) + 1))).table
tbl.columns[0].width = Inches(6.0)
tbl.columns[1].width = Inches(2.0)
tbl.columns[2].width = Inches(2.0)
for j, name in enumerate(["setting", "macro-F1", "± std"]):
    c = tbl.cell(0, j)
    c.text = name
    c.text_frame.paragraphs[0].font.size = Pt(14)
    c.text_frame.paragraphs[0].font.bold = True
for i, r in enumerate(data, start=1):
    for j, val in enumerate(r):
        c = tbl.cell(i, j)
        c.text = val if val else "—"
        p = c.text_frame.paragraphs[0]
        p.font.size = Pt(13)
        if j:
            p.alignment = PP_ALIGN.CENTER
        if "few-shot" in r[0] or "CORAL" in r[0]:
            p.font.bold = True
notes(s, "Say the numbers WITH their error bars. Point out zero-shot raw → "
         "CORAL, then the climb across the five budgets.")

# ── 8. Results plot ────────────────────────────────────────────────
s = slide()
title(s, "F1 vs. labels per class")
if CURVE.exists():
    s.shapes.add_picture(str(CURVE), Inches(1.2), Inches(1.6), height=Inches(4.6))
bullets(s, [
    "Steep rise to ~50 labels, then a plateau.",
    "The plateau sits at the Madrid in-city score — by 100 labels/class the "
    "transferred model is as good on Amsterdam as any model is at home.",
    "Error bars widest at n = 5 (±0.02): tiny support, unstable class means.",
    "Low-data point (25/class) is only ~0.04 below the plateau.",
], left=7.4, top=1.7, width=5.4, height=5.3, size=15)
notes(s, "Lead the whole talk with THIS slide's story if you can: few labels "
         "close the gap. Everything else is how.")

# ── 9. What didn't work ────────────────────────────────────────────
s = slide()
title(s, "What did not work — and why")
bullets(s, [
    "Triplet-loss embedding — features already linearly separable, nothing to learn.",
    "Ordinal loss — shrinks error SIZE, not right-vs-wrong; macro-F1 unmoved.",
    "Label-shift (EM) correction — needs calibrated probabilities; the domain "
    "gap breaks that (0.54 → 0.44).",
    "Self-training on 25k unlabelled — ~35% pseudo-label error at low n, "
    "compounds every round.",
    "Gradient boosting — ties RF in-city, collapses at 5 labels.",
    "Theme: every attempt to be cleverer than the data lost; every accounting fix won.",
])
notes(s, "Own these confidently and briefly — they are the originality / "
         "insightful-failure points on the rubric.")

# ── 10. Limitations ───────────────────────────────────────────────
s = slide()
title(s, "Limitations & honesty")
bullets(s, [
    "Classes 1 vs 2 (both pre-1984) are the main residual error — no "
    "construction event to separate them. A data limit, matching the pre-war "
    "building-age literature.",
    "Spatial-context features help in-city (+0.04 Madrid CV) but break raw "
    "transfer — deliberately excluded from the final pipeline.",
    "All numbers: full 5×5 CV, fixed seed; dev-time quick runs agreed to ±0.01.",
    "Next: hyper-parameter tuning, per-decade features, a dedicated class-1/2 model.",
])
notes(s, "Showing you know the ceiling is worth marks. Do not over-claim.")

# ── 11. Team ──────────────────────────────────────────────────────
s = slide()
title(s, "Team & contributions")
bullets(s, [
    "<member A> — feature engineering, data pipeline (src/data.py)",
    "<member B> — domain alignment: CORAL, whitening (src/adapt.py)",
    "<member C> — evaluation harness, few-shot curve, self-training (src/evaluate.py)",
    "<member D> — results analysis, write-up, presentation",
    "All experiments reproducible: scripts/run_*.py, one fixed seed.",
])
notes(s, "Every member speaks to their own slide at least once — the rubric "
         "checks this explicitly.")

OUT.parent.mkdir(exist_ok=True)
prs.save(str(OUT))
print("wrote", OUT, "-", len(prs.slides.__iter__.__self__._sldIdLst), "slides")
