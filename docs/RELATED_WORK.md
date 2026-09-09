# Related Work — Has Anyone Solved This Before?

**Short answer: yes.** This is an active research area with a directly comparable public
competition, published cross-city transferability numbers, and a mature toolkit of
statistical domain-adaptation methods. Several findings contradict or improve on the
baseline in Notebook 4.

Research date: September 2026. Confidence is flagged per item — some publisher sites
(Taylor & Francis, MDPI, ScienceDirect) returned HTTP 403, so those entries rest on
search-result summaries rather than a direct read of the paper.

---

## 1. The closest match: the ESA #MapYourCity Challenge (2024)

This is almost the same problem, run as a public competition, with results published.

| | MapYourCity | Our hackathon |
|---|---|---|
| Task | Predict building construction **epoch** | Predict building **age class** |
| Classes | **7** ordinal epochs | **4** ordinal classes |
| Region | European cities (EUBUCCO labels) | Madrid, Amsterdam |
| Satellite data | Sentinel-2 (10–20 m) + VHR Pléiades + street view | Landsat 5/7/8/9 (30 m), 40-year time series |
| **Test protocol** | **4 cities entirely held out from training** | **Amsterdam held out, few-shot adaptation** |
| Scale | 15 training cities (~20,000 samples), 4 test cities | 2 cities |

Their seven epochs: `before 1930`, `1930–1945`, `1946–1960`, `1961–1976`, `1977–1992`,
`1993–2006`, `after 2006`. Note the similarity to our class boundaries — mid-century
breaks driven by war and reconstruction, then ~15–20 year intervals.

**Headline results** (higher confidence — read from the arXiv full text):

| Setting | Score |
|---|---|
| Full modality, private leaderboard (winner) | **76.0%** accuracy |
| Full modality, 2nd place | 75.8% |
| **Top-view satellite only** (no street view) | **~58.0%** |
| Generalisation drop to unseen cities | **> 10 points** MPA |

**Why this matters to us:** the satellite-only number (~58% on 7 classes, vs. ~14% random)
is the honest reference point for our task — and MapYourCity had Sentinel-2 at 10 m plus
VHR Pléiades. We have 30 m Landsat. **We are doing the harder version of an already-hard
problem.** If our numbers look modest, this is the citation that says so.

### The winning trick — and our baseline doesn't do it

Team **Creamble** (1st place) won on a single insight:

> Age classes are **ordinal**, not nominal. "After 2006" is much closer to "1993–2006"
> than to "before 1930". A standard classifier throws that structure away.

Their method: convert hard labels into **soft labels shaped as a Gaussian over adjacent
classes**, train with **KL-divergence loss**, with label smoothing factor **0.3**.
Backbone EVA-02, 15-model ensemble over 5-fold CV.

The other podium methods:

| Team | Key idea |
|---|---|
| **TelePIX AI** (2nd) | Country-specific models; **city-fold cross-validation** (hold out whole cities, not random rows); randomly hide street-view images with p=0.5 during training so the model survives missing modalities |
| **xmb** (3rd) | **Focal loss** for class imbalance; distribution-alignment loss; geometric fusion |
| **The AI Buzzard** (4th) | SwinV2 encoders, late fusion, **stratified *grouped*** CV |

Two of the four independently used **whole-city / grouped cross-validation** — a much
stricter protocol than the `RepeatedStratifiedKFold` in Notebook 4. Worth noting in our
write-up.

**Overall conclusion of the challenge:** estimating construction epoch in *unseen* cities
is feasible, and EO-only (no street view) remains practical for global-scale deployment.

- Paper: [Building Age Estimation: A New Multi-Modal Benchmark Dataset and Community Challenge (arXiv 2502.13818)](https://arxiv.org/abs/2502.13818)
- Challenge page: [ESA Φ-lab #MapYourCity](https://philab.esa.int/revealing-urban-secrets-with-the-mapyourcity-challenge/)

---

## 2. Cross-city transferability — the published penalty

The urban-form literature has measured exactly the drop we see in our zero-shot step.

**Headline number: transferring a building-age model between cities costs > 10% accuracy
and > 20% sensitivity.** (Moderate confidence — from search summaries of the Bristol
group's work; publisher pages returned 403.)

Their protocol: train on 6 of 7 cities, test on the 7th, rotate. Result — sensitivity
drops sharply for every held-out city.

**The per-class breakdown is the most interesting part**, because it matches our own
confusion matrix:

| Building period | Cross-city sensitivity |
|---|---|
| 1949–1978 | **> 80%** (best) |
| 1919–1948 | ~40% |
| Post-1995 | 28–61% |
| **Pre-1919** | **7–19%** (worst) |

**The oldest class transfers worst.** That is precisely what Notebook 4 reports — Class 1
pixels getting swallowed by Class 2. This is not a bug in our pipeline; it is a documented
property of the problem. Old building stock is heterogeneous (centuries of accumulated
renovation, infill and material replacement) and its "signature" is the most
city-specific. Saying this in the write-up, with the citation, is worth real rubric points
under *Insightful Failure* and *Low-Data Mechanics*.

Counterpoint for balance: at least one two-city geographic-transfer study reports kappa
values of 0.93 and 0.91 — so transfer is not hopeless; it depends heavily on how similar
the two cities' building stocks are.

- [Predicting building age from urban form at large scale](https://www.sciencedirect.com/science/article/abs/pii/S019897152300073X)
- [Spatial factors influencing building age prediction and implications for urban residential energy modelling](https://www.sciencedirect.com/science/article/pii/S0198971521000442)

---

## 3. A completely different family: Landsat change-point detection

There is a whole literature that dates buildings **not** by classifying their spectral
appearance, but by **detecting the moment the ground changed** from non-building to
building. This is the mainstream approach for Landsat specifically.

| Method | Idea |
|---|---|
| **BATSCCD** | Change detection over Landsat time series for building age in rapidly changing cities |
| **LandTrendr-based** | Temporal segmentation of monthly Landsat series to pull out the construction breakpoint |
| **Logistic curve fitting** | Fit a logistic function to the **NDVI** and **NDBI** time series; the inflection point is the date the surface converted to built-up |

Reported performance (moderate confidence — search summaries; Taylor & Francis returned
403): **87.8% overall accuracy for detecting building change, 77.4% for getting the year
of change right.** One Shenzhen study beat a monthly-LandTrendr baseline by ~1.8×.

**The critical limitation, and why it matters for us:**

> Change detection can only date buildings constructed **during** the satellite record.

Landsat starts in 1984. So this family of methods can, in principle, do very well on our
**Classes 3 and 4** (built 1984–2004 and 2004–2024) — and is structurally **incapable** of
distinguishing our **Classes 1 and 2** (both pre-1984), which it can only lump together as
"already built".

That is a genuinely strong framing for our project:

- **Classes 3/4** → a *change-point / event-detection* problem (when did the surface flip?)
- **Classes 1/2** → a *material-and-weathering signature* problem (what does aged stock
  look like?)

These are two different statistical problems wearing one label. A **hybrid model** that
routes them differently — change-point features for the recent classes, spectral-signature
features for the old ones — would be defensible, novel for this hackathown, and directly
targets the *Methodological Innovation* line of the rubric. Notebook 4's own "ideas for
improvement" gestures at "changepoint detection features: year of largest single-year
jump" without noting that this is an entire published subfield.

> **What actually happened:** we built the change-point block *alongside* the
> spectral features (not a routed hybrid), and it was enough — the final pipeline
> separates classes 1 and 2, with class 1 our *strongest* at ≈ 0.78 and 2–4 near
> 0.70. The "structurally incapable" limit above is real for *change-detection
> alone*; combined with weathering-signature features and cross-city alignment,
> the pre-1984 split is not a dead end. See `docs/FINAL_SUMMARY.md` §6.

- [BATSCCD (Int. J. Digital Earth)](https://www.tandfonline.com/doi/full/10.1080/17538947.2024.2358859)
- [Mapping Building Construction Year from Landsat in Data-Scarce, Cloud-Prone Regions](https://doi.org/10.3390/rs18132135)
- [Time-Series Landsat Data for 3D Reconstruction of Urban History](https://www.mdpi.com/2072-4292/13/21/4339)

---

## 4. The statistical domain-adaptation toolkit

This is the "solved statistically" part of the question. There is a mature, mostly
*unsupervised* set of methods for exactly our Madrid→Amsterdam shift.

### CORAL — Correlation Alignment (the best value-for-effort option here)

Aligns the **second-order statistics** (the covariance matrix) of source and target
feature distributions:

1. **Whiten** the source features (remove their correlation structure).
2. **Re-colour** them with the *target* domain's covariance.

Properties that make it near-ideal for our constraints:

- **Requires no target labels** — it uses only the *unlabelled* Amsterdam feature matrix,
  which we are given in full. **No leakage**, since no Amsterdam label is touched.
- Famously simple — the original paper notes it is ~4 lines of MATLAB.
- Has a differentiable deep variant (**Deep CORAL**) if we go the neural route.

Applied to our pipeline it slots in immediately after Notebook 3's standardisation and
before the classifier — a genuine domain-adaptation step that the current baseline lacks
entirely. Under the rubric this converts "standard off-the-shelf model" into "applied
domain adaptation tricks", and it is cheap to implement.

Related family members worth naming in a presentation: **MMD** (Maximum Mean Discrepancy)
alignment, **domain-adversarial training / DANN** (train an encoder to *defeat* a
city-classifier so the representation carries no city identity), and simple
**histogram matching** of per-band reflectance distributions.

- [Correlation Alignment for Unsupervised Domain Adaptation (arXiv 1612.01939)](https://ar5iv.labs.arxiv.org/html/1612.01939)
- [Domain Adaptation for Time-Series Classification to Mitigate Covariate Shift](https://arxiv.org/pdf/2204.03342)
- [A Survey of Unsupervised Domain Adaptation for Visual Recognition](https://arxiv.org/pdf/2112.06745)

### Ordinal regression — the statistically correct loss for age classes

Our four classes have a natural order, and standard classification ignores it. The
published fix is **soft ordinal labels**: replace the one-hot target with a probability
vector whose mass decays with distance from the true class, then train with cross-entropy
or KL-divergence. This encodes both the *ordering* and the *metric* between classes at zero
architectural cost.

This is exactly what won MapYourCity (§1). There is even a maintained Python package,
`dlordinal`, for deep ordinal classification.

- [Soft Labels for Ordinal Regression (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.pdf)
- [Deep Ordinal Regression with Label Diversity](https://arxiv.org/pdf/2006.15864)
- [dlordinal: A Python package for deep ordinal classification](https://www.sciencedirect.com/science/article/pii/S0925231224020769)
- [Deep Ordinal Regression using Optimal Transport Loss and Unimodal Output Probabilities](https://arxiv.org/pdf/2011.07607)

### Few-shot metric learning — our Notebook 4 approach, in context

Prototypical Networks, Siamese and Triplet networks are the standard metric-learning
family, and there is a substantial remote-sensing-specific literature: **HCPNet**
(discriminative prototypes for few-shot RS scene classification), **global–local
prototype-based few-shot learning for cross-domain hyperspectral classification**, and
survey work on few-shot RS scene classification.

So the prototype classifier in Notebook 4 is a legitimate, published technique — but it is
the **plain baseline** of that family. The literature's improvements over vanilla
prototypes (contrastive prototype learning, global–local prototypes, distribution
alignment before prototype computation) are the natural place to look for our originality
points.

- [Few-shot RS image scene classification: recent advances, new baselines, future trends](https://www.sciencedirect.com/science/article/pii/S0924271624000509)
- [HCPNet: discriminative prototypes for few-shot RS scene classification](https://www.sciencedirect.com/science/article/pii/S1569843223002716)
- [Global–local prototype-based few-shot learning for cross-domain hyperspectral classification](https://www.sciencedirect.com/science/article/abs/pii/S0950705125002461)
- [Cross-City Matters: multimodal benchmark + HighDAN adversarial cross-city segmentation](https://arxiv.org/abs/2309.16499)

---

## 5. Where the labels come from — EUBUCCO

Worth knowing, because it is almost certainly the ultimate source of our Amsterdam and
Madrid construction years.

**EUBUCCO v0.1** — ~202 million individual building footprints across the EU-27 plus
Switzerland, assembled from 50 open government datasets and OpenStreetMap. Attribute
coverage: height 73%, **construction year only 24%**, type 46%.

That 24% figure *is* the motivation for this entire hackathon: three quarters of European
buildings have no recorded construction year, and the gap is wildly uneven between
countries. EUBUCCO explicitly tags attributes as Ground Truth / Merged / **ML Estimated**,
i.e. the project anticipates models like ours filling it in.

Open code: [`ai4up/ufo-prediction`](https://github.com/ai4up/ufo-prediction) — "using urban
form information to predict building attributes". Directly reusable prior art.

- [EUBUCCO v0.1 (Nature Scientific Data)](https://www.nature.com/articles/s41597-023-02040-2)
- [eubucco.com](https://eubucco.com/)

---

## 6. What the wider literature uses that we don't have

Most published building-age work leans on signals absent from our dataset. Useful for
honest scoping in the presentation:

| Signal | Typical use | Available to us? |
|---|---|---|
| Building **footprint geometry** (area, shape, adjacency) | Strong age predictor | ✗ (only `coverage`) |
| **Building height** | Strong age predictor | ✗ |
| **Street-block morphology** | Strong age predictor | ✗ (but `px_key`/`py_key` allow neighbourhood context) |
| **Street-view façade imagery** | Best single modality in MapYourCity | ✗ |
| **VHR / Sentinel-2 imagery** (≤10 m) | Texture, individual roofs | ✗ (30 m Landsat only) |
| **40-year annual spectral time series** | Change-point dating | ✓ **our main asset** |

Random-Forest models using height + footprint + block metrics report **MAE of 12–18 years**.
Those numbers are *not* comparable to ours — different inputs, different task.

Our comparative advantage is **temporal depth**: 40 years of annual observations per pixel.
Most of the cited work uses a single snapshot. Any originality claim should be built on
the time series, because that is the thing we have and they mostly don't.

---

## 7. Actionable shortlist

Ranked by (rubric points earned) ÷ (effort), given 40 points ride on originality.

1. **Ordinal-aware soft labels + KL-divergence loss.** Won MapYourCity. Our baseline
   treats classes 1–4 as unordered, so confusing Class 1 with Class 4 is penalised the same
   as confusing it with Class 2 — statistically wrong. Cheap, citable, immediately
   defensible. Report **MAE in class-steps** alongside macro F1.
2. **CORAL alignment of Madrid → Amsterdam features.** Unsupervised, leakage-free (uses
   unlabelled Amsterdam only), a few lines of code, and it turns "off-the-shelf model" into
   "domain adaptation" on the rubric.
3. **Split the problem by mechanism.** Change-point features for Classes 3/4 (which have
   their construction event inside the Landsat record) + weathering/signature features for
   Classes 1/2 (which don't). Novel framing, physically motivated, exploits our one real
   advantage.
4. **Whole-city / grouped CV discipline.** Two of four MapYourCity podium teams did this.
   Cheap credibility, and it guards against the leakage deduction.
5. **Cite the > 10% cross-city drop and the pre-1919 sensitivity collapse** when
   interpreting the zero-shot result. Reframes a weak number as a predicted, understood
   outcome — this is what *Insightful Failure* rewards.
6. **Focal loss or class weighting** for the Class 1/2 imbalance the notebook flags but
   never fixes (3rd place used focal loss for exactly this).

---

## Sources

- [Building Age Estimation: A New Multi-Modal Benchmark Dataset and Community Challenge (arXiv 2502.13818)](https://arxiv.org/abs/2502.13818)
- [ESA Φ-lab — Revealing urban secrets with the #MapYourCity Challenge](https://philab.esa.int/revealing-urban-secrets-with-the-mapyourcity-challenge/)
- [BATSCCD: change detection for mapping building age from Landsat time series](https://www.tandfonline.com/doi/full/10.1080/17538947.2024.2358859)
- [Mapping Building Construction Year from Landsat in Data-Scarce, Cloud-Prone Regions](https://doi.org/10.3390/rs18132135)
- [Time-Series Landsat Data for 3D Reconstruction of Urban History](https://www.mdpi.com/2072-4292/13/21/4339)
- [Predicting building age from urban form at large scale](https://www.sciencedirect.com/science/article/abs/pii/S019897152300073X)
- [Spatial factors influencing building age prediction and implications for urban residential energy modelling](https://www.sciencedirect.com/science/article/pii/S0198971521000442)
- [Deep learning and remote sensing for scalable building age prediction in urban energy modeling](https://www.sciencedirect.com/science/article/abs/pii/S0378778825010333)
- [Correlation Alignment for Unsupervised Domain Adaptation (CORAL)](https://ar5iv.labs.arxiv.org/html/1612.01939)
- [A Survey of Unsupervised Domain Adaptation for Visual Recognition](https://arxiv.org/pdf/2112.06745)
- [Domain Adaptation for Time-Series Classification to Mitigate Covariate Shift](https://arxiv.org/pdf/2204.03342)
- [Soft Labels for Ordinal Regression (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.pdf)
- [Deep Ordinal Regression with Label Diversity](https://arxiv.org/pdf/2006.15864)
- [dlordinal: A Python package for deep ordinal classification](https://www.sciencedirect.com/science/article/pii/S0925231224020769)
- [Few-shot RS image scene classification: recent advances, new baselines, future trends](https://www.sciencedirect.com/science/article/pii/S0924271624000509)
- [HCPNet: discriminative prototypes for few-shot RS scene classification](https://www.sciencedirect.com/science/article/pii/S1569843223002716)
- [Global–local prototype-based few-shot learning for cross-domain hyperspectral classification](https://www.sciencedirect.com/science/article/abs/pii/S0950705125002461)
- [Cross-City Matters: multimodal RS benchmark, HighDAN](https://arxiv.org/abs/2309.16499)
- [EUBUCCO v0.1 (Nature Scientific Data)](https://www.nature.com/articles/s41597-023-02040-2)
- [ai4up/ufo-prediction (GitHub)](https://github.com/ai4up/ufo-prediction)
