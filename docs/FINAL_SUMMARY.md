# Final summary — cross-city building-age transfer

One place that says what the challenge was, what we built, what the numbers are,
everything we tried, and what still needs doing. Plain language.

Branch: `israel-transfer-pipeline` on
<https://github.com/ubayd-hattas/Data_Science_Hackathon> · 43 commits.

---

## 1. The challenge

Estimate the **age group** of the buildings in a 30 m satellite patch — one of
four construction eras — from 40 years of how that patch reflects light. The
twist: the model has to work in a **city it was never trained on**. We train on
Madrid and transfer to Amsterdam, using between 5 and 200 labelled Amsterdam
patches per group.

Scored on **macro-F1** (0 = useless, 1 = perfect; all four groups weighted
equally, so the rare ones matter).

---

## 2. What we built

A three-part pipeline, all in `src/`, run end to end by
`notebooks/5-Transfer.ipynb` (one fixed seed, ~15 minutes):

1. **Stage 1** — a class-balanced Random Forest (500 trees) trained on 108
   per-pixel features: multi-period band statistics, five spectral indices, the
   size/direction/**timing** of the biggest single-year change, and 8-neighbour
   feature averages. One row per pixel, so the model reasons about trends.
2. **Zero labels** — **class-conditional CORAL**: predict Amsterdam once, then
   reshape each Madrid age group to match the covariance of the Amsterdam pixels
   the model assigned to that group, and refit. Two rounds. No Amsterdam labels.
3. **A few labels** — budget-scaled feature whitening → a small local Random
   Forest → probabilities blended with Stage 1 (weight shifts to local as
   labels grow) → inverse-distance smoothing over map-neighbours.

Every alignment/whitening/prior statistic comes from the **unlabelled** Amsterdam
pool; the only labelled target data that touches the model is the per-trial
support set. The smoothing is prediction-only.

---

## 3. Final numbers

Full 5×5 CV, 20 support draws, seed 42 (`results/deliverable_table.csv`):

| Setting | macro-F1 |
|---|---:|
| Madrid 5×5 CV (in-city ceiling) | 0.664 ± 0.004 |
| Amsterdam zero-shot, raw | 0.358 |
| Amsterdam zero-shot, pooled CORAL | 0.578 |
| **Amsterdam zero-shot, class-conditional CORAL** | **0.651** |
| Amsterdam few-shot, 5 / class | 0.662 ± 0.003 |
| Amsterdam few-shot, 25 / class | 0.677 ± 0.006 |
| Amsterdam few-shot, 50 / class | 0.704 ± 0.009 |
| Amsterdam few-shot, 100 / class | 0.721 ± 0.007 |
| Amsterdam few-shot, 200 / class | 0.737 ± 0.005 |

**From the provided-notebook baseline** (0.43 zero-shot; 0.42 / 0.55 / 0.61 /
0.64 / 0.67 few-shot): zero-shot **+0.22**, few-shot **+0.07 to +0.23**.

**Headline:** the few-shot curve crosses the Madrid in-city score at ~50 labels
per class — roughly 200 checked buildings — and keeps climbing. Transferring to a
new city, past that point, costs nothing.

---

## 4. What worked (biggest first)

| change | what it is | effect |
|---|---|---|
| Budget-scaled shrinkage whitening | untangle redundant features; strength dialled by label count | **+0.15 / +0.06 / +0.03** at 5 / 25 / 50 labels |
| Class-conditional CORAL | per-group covariance alignment via pseudo-labels | **+0.07** zero-shot (0.58 → 0.65); +0.02 at n=5 |
| Pooled CORAL (the base version) | align the whole feature cloud before training | +0.10 to +0.22 zero-shot |
| Feature untangling (whitening) | each idea counts once, not 3× | +0.06 at mid budgets |
| Two-model probability blend | local vote + Stage-1 vote, weight ∝ n | +0.04 at n=5 |
| Overnight 500-config search | feature set + hyper-parameters, held-out tuning half | +0.02–0.03 across the curve |
| Distance-weighted neighbour smoothing | closer neighbours weighted more | +0.011 at every budget |

## 5. What didn't work — and why

| tried | why it lost |
|---|---|
| Triplet-loss embedding network | features already linearly separable — nothing to learn; added variance |
| Ordinal training | shrinks error *size*, not *count*; macro-F1 doesn't care |
| Label-shift (EM) correction | needs calibrated probabilities; the domain gap breaks that |
| Self-training on unlabelled Amsterdam | ~35 % pseudo-label error at low n compounds each round |
| Gradient boosting instead of Random Forest | ties in-city, collapses at n=5 |
| Richer multi-scale neighbourhood features | +0.001 — the 8-NN mean already captures it |
| Support-set mixup augmentation | slightly negative; forests don't gain from interpolated points |
| Anchoring the smoothing to true support labels | +0.006, but that gain is same-block proximity, not skill — dropped after audit |
| Ubayd's logistic-regression head / score-product blend | −0.018 / −0.019 on our whitened pipeline; his edge was vs a weaker baseline |

**Pattern:** every attempt to out-*model* the data lost to better *use* of it.

---

## 6. Honest limitations (stated in the deck and write-up)

- **Age classes 1 and 2** (both pre-1984) can't be cleanly separated — no
  construction event in the satellite record. A data limit; matches the
  building-age literature. Most residual error is here (class-2 recall ≈ 0.52).
- **Spatial adjacency:** under the organisers' random per-class sampling, ~80 %
  of support pixels have an immediate map-neighbour in the query set (measured
  independently by two teammates). No rule is broken and query labels are never
  used, but part of every few-shot score is same-block proximity, not pure
  cross-location generalisation. Disclosed; smoothing kept prediction-only so it
  can't compound it.
- **± values are dispersion**, not confidence intervals — repeated resampling,
  not k-fold CV.

---

## 7. Repository map

| path | what |
|---|---|
| `notebooks/1–4` | organisers' notebooks, untouched |
| `notebooks/5-Transfer.ipynb` | our pipeline, one seeded run → table + plots |
| `src/data.py` | lean parquet loading + 108-feature engineering |
| `src/adapt.py` | CORAL, class-conditional CORAL, whitening, spatial smoother |
| `src/evaluate.py` | Madrid CV, zero-shot, few-shot curve variants |
| `src/embedding.py`, `src/ordinal.py` | kept for the record (both negative) |
| `scripts/run_*.py` | every experiment, re-runnable |
| `results/deliverable_table.csv`, `transfer_curve.png` | the required table + plot |
| `deliverables/Building-Age-Transfer-final.pptx` | 12-slide PowerPoint |
| `deliverables/slides-research.html`, `slides-preview.html` | clickable 7-slide preview decks (present / overview / PDF-for-Canva) |
| `docs/WRITTEN_JUSTIFICATION.md` | 441-word justification (3 pillars) |
| `docs/SLIDE_CONTENT.md` | 7-slide content table |
| `docs/TALK_SCRIPT.md` | timed spoken script, 4 speakers, Q&A crib |
| `docs/TEAM_BRIEF.md` | plain-language explainer for teammates |
| `docs/WHAT_I_TRIED.md` | full experiment diary |
| `docs/LESSONS.md` | consolidated understanding |
| `docs/RELATED_WORK.md` | published prior art (MapYourCity, CORAL, etc.) |
| `docs/AMSTERDAM_SPATIAL_ADJACENCY_AUDIT.md` | teammate audit of the adjacency issue |

---

## 8. Deliverables checklist

| requirement | status |
|---|---|
| Trained Madrid model + adaptation code, parameterised by `n`, fixed seed | ✅ `src/` + notebook 5 |
| Madrid CV macro-F1 (mean, SD) | ✅ 0.664 ± 0.004 |
| Amsterdam few-shot F1 at all five budgets, with error bars | ✅ table + plot |
| F1 vs log₂(sample size) plot with error bars | ✅ `results/transfer_curve.png` |
| Written justification (≤ 300–500 words) | ✅ 441 words — **cut to ≤300 if organisers hold to Notebook 1** |
| Abstract (≤ 150 words) on slide 1 | ✅ 149 words |
| PowerPoint: design decisions, worked/didn't, the table, the plot | ✅ `.pptx` + HTML decks |
| Team: everyone contributes and speaks | ⚠️ script assigns 4 speakers; only 2 of 4 have commits |

---

## 9. What still needs a human

1. **Fill in team names** — slides 1 & 7, `<Author A–D>`, `<team name>`.
2. **Two teammates each make one real commit** (you and 0geder have them).
3. **Rehearse** the 7-slide talk once, together — every member on their slides.
4. **Ask the organisers:** four or five Amsterdam F1 scores? (Notebook 1 says
   both.) Word cap 300 or 500? Report all five and write ≤300 until they reply.
5. **Optional:** generate the two figures the deck references — the
   before/after "cities line up" scatter (slide 3) and per-group score bars
   (slide 6).
