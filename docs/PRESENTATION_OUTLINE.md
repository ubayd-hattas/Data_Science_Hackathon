# Presentation — Abstract + Slide Outline

Rubric expectations for this section: abstract (≤150 words) on slide 1; key
design decisions and what worked / didn't; the F1 table with error bars; the
F1-vs-log₂(sample size) plot. Judged on storytelling, interpretation of the
curves, and every team member speaking.

---

## Abstract (slide 1)

> Predicting building construction era from 30 m Landsat imagery is hard, and
> harder across cities: a model trained on Madrid must work in Amsterdam, where
> building materials, climate and urban form all differ. We compress each pixel's
> 40-year, six-band reflectance series into 60 temporal statistics and train a
> class-balanced Random Forest on Madrid. Transfer rests on two unsupervised,
> label-free alignments: CORAL matches Madrid's feature covariance to Amsterdam's
> before training (zero-shot macro-F1 0.45 → 0.55), and budget-scaled ZCA
> whitening with a small local classifier, blended with the CORAL model, handles
> the few-shot regime. With 100 labelled Amsterdam pixels per class the
> transferred model matches Madrid's own in-city score (0.68 vs 0.63). Residual
> error concentrates on pre-1984 classes, which carry no construction event in
> the satellite record. Simple distribution alignment beat a learned embedding,
> ordinal loss, self-training and label-shift correction.

*(148 words. Trim the last sentence if a hard 150 cap is enforced with the title.)*

---

## Slide outline (~11 slides, ~12 min + Q&A)

### 1 — Title + Abstract
Team name, members, one-line framing: *"A city-portable building-age classifier
from 40 years of Landsat."* Abstract text on the slide.

### 2 — The problem
- Task: age class (1–4) of buildings in a 30 m Landsat pixel.
- Why satellites can see it: materials and weathering change the spectral
  fingerprint over time.
- The real challenge: **generalise to a new city with few labels.** Madrid →
  Amsterdam is the test case for "any city on Earth."
- One visual: a Madrid pixel time series next to an Amsterdam one, same class,
  visibly different levels.

### 3 — Data → features
- One row per (pixel, year); up to 3 observations/year; ~40 years.
- Key decision: **collapse to one row per pixel** — 60 temporal statistics
  (mean/std overall, early 1984–2003, late 2004+, year-on-year; 5 spectral
  indices; coverage flags).
- Why: forces the model to use *trends*, not treat each year as independent.
- The early/late split targets classes 3–4 — their construction event sits
  inside the record.

### 4 — Approach in one picture
Diagram: Madrid features → [CORAL align] → Random Forest (Stage 1).
Two transfer routes out:
- 0 labels → CORAL model directly.
- n labels → whiten Amsterdam (shrinkage ∝ n) → small RF on support → blend with
  CORAL model.

### 5 — Zero-shot: CORAL
- Plain Madrid model on Amsterdam: 0.45 — the rules are right, the coordinates
  are wrong.
- CORAL: reshape Madrid's feature cloud to Amsterdam's covariance *before*
  training. No Amsterdam labels touched.
- Result: **0.45 → 0.55.** ~15 lines of linear algebra.

### 6 — Few-shot: whiten + local head + blend
- The 60 features are ~⅔ redundant; nearest-prototype distance double-counts
  them. Whitening untangles them.
- Whitening needs data → shrink it toward plain scaling when labels are scarce
  (dial set by budget).
- Small Random Forest on the whitened support beats nearest-prototype (~+0.02).
- Blend its vote with the CORAL model, weight shifting local as n grows: another
  +0.01–0.04, biggest at n = 5.

### 7 — Results table
| setting | macro-F1 |
|---|---|
| Madrid 5×5 CV | 0.626 ± 0.004 |
| Amsterdam zero-shot (raw) | 0.448 |
| Amsterdam zero-shot (CORAL) | 0.545 |
| Amsterdam few-shot, 5/class | 0.614 ± 0.020 |
| 25/class | 0.645 ± 0.009 |
| 50/class | 0.659 ± 0.009 |
| 100/class | 0.680 ± 0.009 |
| 200/class | 0.685 ± 0.008 |

### 8 — Results plot + interpretation
`results/transfer_curve.png`. Talking points:
- Steep rise to ~50 labels, then a plateau.
- The plateau sits **at the Madrid in-city score** — with 100 labels/class the
  transferred model is as good on Amsterdam as any model is at home.
- Error bars widest at n = 5 (±0.02): tiny support, unstable class means.
- The low-data prize point (25/class = 0.645) is only ~0.04 below the plateau.

### 9 — What didn't work (and why)
| tried | why it failed |
|---|---|
| Triplet-loss embedding | features already linearly separable — nothing to learn |
| Ordinal loss | improves error *size*, not right-vs-wrong; macro-F1 unmoved |
| Label-shift (EM) correction | needs calibrated probabilities; the domain gap breaks that |
| Self-training on 25 k unlabelled | ~35 % pseudo-label error at low n, compounds each round |
| Gradient boosting | ties RF in-city, collapses at 5 labels |
Theme: **every attempt to be cleverer than the data lost; every accounting fix won.**

### 10 — Limitations & honesty
- Classes 1 vs 2 (both pre-1984) are the main residual error — no construction
  event to separate them. Data limit, matches the literature on pre-war stock.
- Spatial-context features help in-city (+0.04) but break raw transfer — kept out
  of the final pipeline.
- Numbers from a full 5×5 CV, seed-fixed; quick-run estimates during development
  agreed to ±0.01.
- Next: hyper-parameter tuning, per-decade features, a class-1/2 specialist.

### 11 — Team contributions
One line per member: who owned features / alignment / evaluation / write-up.
Every member speaks to their slide.

---

## Delivery notes

- Lead with the **curve**, not the pipeline. The story is "few labels close the
  gap"; the methods are how.
- Say the numbers out loud with their error bars — the rubric rewards curve
  interpretation specifically.
- Own the dead ends briefly and confidently; they show rigour and hit the
  originality criterion.
- Keep one sentence ready for "why not deep learning?" → *30 m Landsat, ~100 k
  pixels, already-separable features; a forest plus alignment is the right size.*
