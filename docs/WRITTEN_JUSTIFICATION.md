# Written Justification

*Target: ≤300 words (Notebook 1 cap; the rubric states 500 — 300 satisfies both).
Covers the three required pillars: model design, transfer strategy, F1
interpretation. Word count of the justification body is noted at the end; confirm
the cap with the organisers before submitting.*

---

## Justification

**Model design.** Each pixel's 40-year, six-band Landsat series is collapsed to 60
temporal statistics — overall, early/late-period and year-on-year mean and
standard deviation per band, five spectral indices, coverage flags. One row per
pixel forces the classifier to reason about trends rather than treating each year
as an independent sample. Stage 1 is a class-balanced Random Forest (300 trees).
We kept it over gradient boosting, which matched it in-city but failed at small
few-shot budgets, and over a learned triplet embedding, which added variance
without gain because the 60 features are already linearly separable.

**Transfer strategy.** Two label-free alignments. Zero-shot: CORAL — whiten
Madrid's feature covariance, recolour with Amsterdam's — before training, so
boundaries are learned in the target's coordinate system. Few-shot: ZCA-whiten
the Amsterdam features with shrinkage scaled to the label budget (heavy when
support is thin, none at 200), removing the redundancy that otherwise dominates
prototype distance; a small Random Forest is fitted on the whitened support and
its probabilities blended with the CORAL Stage-1 model, weight
clip(n/50, 0.4, 0.95). All target statistics come from the unlabelled pool; the
only labelled target data is the support set.

**F1 interpretation.** Madrid CV 0.626 ± 0.004. Zero-shot 0.448 raw to 0.545 with
CORAL — alignment recovers about half the domain gap with no labels. Few-shot
0.61 / 0.65 / 0.66 / 0.68 / 0.69 at 5 / 25 / 50 / 100 / 200 labels per class: the
curve rises steeply to roughly 50 labels, then flattens at the Madrid in-city
score — with 100 labels the transferred model matches a home-city model. Residual
error concentrates on classes 1 and 2, both pre-1984, where no construction event
exists to separate them: a data limit, not a pipeline defect.

---

*Justification body: 288 words.*
