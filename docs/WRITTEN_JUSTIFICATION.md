# Written Justification

*Rubric cap: 500 words. Covers the three required pillars — model design, transfer
strategy, F1 interpretation. Word count of the justification body noted at the
end.*

---

## Justification

**Model design.** Each pixel carries up to 40 years of annual six-band Landsat
reflectance. We gap-fill each series (linear interpolation inside, edge-hold
outside) and collapse it to 60 temporal statistics: overall, early-period
(1984–2003), late-period (2004+) and year-on-year mean and standard deviation per
band, plus five spectral indices and two data-coverage flags. Reducing to one row
per pixel is deliberate — feeding raw per-year rows would treat each year as an
independent sample and leak temporal structure; the summary forces the model to
reason about trends and variability, which is what distinguishes construction
eras. The early/late split targets classes 3–4, whose construction event falls
inside the record and shows as a contrast between the two windows. Stage 1 is a
class-balanced Random Forest (300 trees, macro-F1 objective). We tested gradient
boosting (matched in-city, weaker at small few-shot budgets) and a triplet-loss
embedding (added variance without gain — the 60 features are already linearly
separable), and kept the simpler model.

**Transfer strategy.** Two unsupervised, label-free alignments. Zero-shot: CORAL —
whiten Madrid's feature covariance and recolour it with Amsterdam's — applied
before training, so the decision boundaries are learned directly in the target's
coordinate system. Few-shot: ZCA-whiten the Amsterdam features, with shrinkage
toward the diagonal scaled to the label budget (near-full at 5 labels/class,
none at 200) because a whitening rotation estimated against a noisy class mean
hurts when support is thin. A small Random Forest is fitted on the whitened
support set and its class probabilities are blended with the CORAL Stage-1
model, weight clip(n/50, 0.4, 0.95) toward the local head as labels accumulate.
Leakage discipline: every target statistic (CORAL covariance, whitening
covariance, Stage-1 prior) is computed from the unlabelled Amsterdam pool; the
only labelled target data entering the pipeline is the per-trial support set,
and Madrid CV folds split on pixels with one row per pixel.

**F1 interpretation.** Madrid 5×5 CV: 0.626 ± 0.004 — the in-city ceiling.
Zero-shot transfer: 0.448 raw, 0.545 with CORAL; a single covariance alignment
recovers roughly half the domain gap with no Amsterdam labels. Few-shot macro-F1:
0.61 / 0.65 / 0.66 / 0.68 / 0.69 at 5 / 25 / 50 / 100 / 200 labels per class. The
curve rises steeply to about 50 labels, then flattens at the Madrid in-city
score — by 100 labels/class the transferred model is as accurate on Amsterdam as
a model is on its home city. Residual error concentrates on classes 1 and 2,
both pre-1984: with no construction event in the satellite record to separate
them, this split is close to a data limit rather than a modelling shortfall,
consistent with the building-age literature on pre-war stock.

---

*Justification body: 439 words.*
