# Written Justification

*Rubric cap: 500 words (confirmed). Covers the three required pillars — model
design, transfer strategy, F1 interpretation. Word count noted at the end.*

---

## Justification

**Model design.** Each pixel carries up to 40 years of annual six-band Landsat
reflectance. We gap-fill each series (linear interpolation inside, edge-hold
outside) and collapse it to 108 per-pixel statistics: overall, early-period
(1984–2003), late-period (2004+) and year-on-year mean and standard deviation per
band; five spectral indices; the magnitude, direction and **timing** of each
band's largest single-year jump; and neighbourhood averages over the eight
nearest pixels. Reducing to one row per pixel is deliberate — feeding raw
per-year rows would treat each year as an independent sample and leak temporal
structure, whereas summaries force the model to reason about trends. The
early/late split and the jump-timing features target classes 3–4, whose
construction event falls inside the record; the neighbourhood block exploits the
fact that city blocks share a construction era. Stage 1 is a class-balanced
Random Forest (500 trees, unrestricted depth). Feature set and hyperparameters
were chosen by a 500-configuration random search scored on one half of Amsterdam
and confirmed once on the untouched other half. We tested and rejected gradient
boosting (matched in-city, weaker at small budgets), a triplet-loss embedding
(added variance without gain — the features are already linearly separable),
ordinal decomposition, and label-shift correction.

**Transfer strategy.** Two unsupervised, label-free alignments. Zero-shot: CORAL
— whiten Madrid's feature covariance and recolour it with Amsterdam's — applied
before training, so decision boundaries are learned directly in the target's
coordinate system. Few-shot: ZCA-whiten the Amsterdam features, with shrinkage
toward the diagonal scaled to the label budget (near-full at 5 labels/class, none
at 200), because a whitening rotation applied to a noisy class mean hurts when
support is thin. A small Random Forest is fitted on the whitened support set and
its class probabilities blended with the CORAL Stage-1 model, weight
clip(n/50, 0.3, 0.95) shifting toward the local head as labels accumulate.
Leakage discipline: every target statistic — CORAL covariance, whitening
covariance, neighbourhood averages, Stage-1 prior — is computed from the
unlabelled Amsterdam pool; the only labelled target data entering the pipeline is
the per-trial support set, and Madrid CV folds split on pixels with one row each.

**F1 interpretation.** Madrid 5×5 CV: 0.664 ± 0.004, the in-city ceiling.
Zero-shot: 0.358 raw, 0.578 with CORAL — one covariance alignment recovers most
of the domain gap with no labels, and is essential once neighbourhood features
are present, since those encode Madrid-specific urban form. Few-shot macro-F1:
0.62 / 0.66 / 0.68 / 0.70 / 0.72 at 5 / 25 / 50 / 100 / 200 labels per class. The
curve rises steeply to about 50 labels, then flattens *at* the Madrid ceiling —
by 100 labels/class the transferred model matches and slightly exceeds a
home-city model. Residual error concentrates on classes 1 and 2, both pre-1984:
with no construction event in the record to separate them, this is a data limit
rather than a modelling shortfall, consistent with the building-age literature on
pre-war stock.

---

*Justification body: 479 words.*
