# Slide content — research version (8 slides, ~10 min)

Four columns: the plain heading, what goes *on* the slide (kept short), the
jargon the speaker keeps in reserve for Q&A, and the rubric item each slide
serves. Numbers are current (few-shot 0.66–0.74, zero-shot 0.36→0.65).

Two figures still to generate: the before/after domain-shift scatter (slide 4)
and per-class F1 / confusion (slide 7).

| # | Heading (plain) | On the slide (short) | Speaker notes carry (jargon in reserve) | Rubric |
|---|---|---|---|---|
| 1 | Building age from space — transferred to a city we've never studied | Title · authors line · team name · abstract panel (149 words, verbatim) | The one-liner: "a model that fails in a new city is mis-tuned, not broken." | §0 · abstract on slide 1 |
| 2 | The same model doesn't work in a new city | • sort each 30 m satellite patch into 1 of 4 age groups; score = macro-F1<br>• train on Madrid, use in Amsterdam with 5–200 examples per group<br>• Amsterdam looks different — brick vs concrete, wetter climate, different satellites across the years<br>• Madrid → Amsterdam stands in for *any* city | covariate shift **and** label shift (class-1 prior 1.6× higher in Amsterdam); 108 features; macro-F1 weights all four classes equally; the three contributions | §1 · Challenge understanding (domain shift) |
| 3 | We describe each patch by how it changed, not how it looks | • 40 years × 6 colours → 108 summary numbers per patch<br>• one row per patch → the model must use trends, not single snapshots<br>• the key group: *timing* of the biggest year-to-year jump — the construction event<br>• features picked for physics that travels, not to fit Madrid | gap-filling; overall / early / late / year-on-year mean & SD per band; 5 spectral indices (NDVI, NDBI…); 8-nearest-neighbour feature averages; 76,263 Madrid / 25,992 Amsterdam pixels | §2 · Model design (pillar 1) |
| 4 | Our fix: reshape Madrid's data to match Amsterdam — one age group at a time | • pipeline: Madrid features → Random Forest (500 trees) → align → predict<br>• 4-line algorithm box for the alignment<br>• no Amsterdam answers used — only the model's own first guesses<br>• zero-label score **0.36 → 0.65**<br>• leakage box: only the support labels ever touch the model | class-conditional CORAL; pseudo-labels, 2 rounds; per-class covariance match; plain CORAL as the baseline it improves on; 500-config hyperparameter search scored on a held-out Amsterdam half | §3 · Model design + Transfer strategy (pillars 1–2); "did the model address domain shift" |
| 5 | A few local examples do the rest — and "few" needs a different recipe than "many" | • 5–25 examples → mostly trust the Madrid model<br>• 200 examples → mostly trust the local model<br>• the blend shifts automatically as the number grows<br>• untangling is turned down when data is thin<br>• setup: 5×5 CV, 5 budgets, 20 draws each, seed 42, ~15 min end-to-end | budget-scaled ZCA shrinkage whitening; blend weight ∝ n; inverse-distance neighbour smoothing; repeated resampling ≠ k-fold (± is spread); support-selection sensitivity ±0.02 at n=5, ±0.007 at n=100 | §4 · Low-data mechanics + reproducibility |
| 6 | With 50 examples per group, it matches a home-city model | • Table 1: zero-shot 0.36 → 0.65; few-shot 0.66 / 0.68 / 0.70 / 0.72 / 0.74 (± err); Madrid ceiling 0.66<br>• Fig. 2: F1 vs log₂(examples per group), error bars ±1 SD, dashed line = Madrid ceiling<br>• baseline row: provided notebook 0.42 → 0.67<br>• reading: the curve crosses the ceiling near 50 examples | ~200 checked buildings total to reach home-city accuracy; error bars ±0.003–0.009; steep-then-flat means the alignment does the work and labels buy the last few points | §5 · F1 table + F1-vs-log₂(n) plot + interpretation (pillar 3) |
| 7 | What each piece is worth, what didn't work, and what we won't claim | • **worked** (Δ macro-F1): alignment +0.07 · whitening +0.15 @ n=5 · blend +0.04 · smoothing +0.01 · search +0.02<br>• **didn't**: neural net · ordinal loss · self-training · gradient boosting · class-mix fix — one line each on *why*<br>• **limit**: classes 1 & 2 (both pre-1984) can't be separated — class-2 recall 0.52; ~26% of class-1 predicted as class 2<br>• small 4-bar per-class F1 or a 4×4 confusion matrix | ablation deltas; "every attempt to out-model the data lost to using it more carefully"; ~80% of support pixels touch a query pixel — disclosed, smoothing kept prediction-only | §6 · design decisions + what worked/didn't; insightful failure; scientific soundness |
| 8 | It's a calibration problem, not a hard one | 3 takeaways:<br>1 · a new-city model is mis-tuned, not incapable<br>2 · ~50 labels buy home-city accuracy; the low-data case needs its own recipe, not a shrunk full pipeline<br>3 · careful use of the data beat every fancier model<br>• originality box · next steps (block-disjoint evaluation, more cities) · who did what — each name | originality claim = iterative pseudo-label-driven per-class alignment + the framing that transfer is a spectrum needing different machinery at different label budgets | §7 · Originality + Team collaboration |

## Timing (target ≈ 9:50)

| slide | § | time | speaker |
|---|---|---|---|
| 1 | Title | 0:30 | A |
| 2 | Problem & contributions | 1:15 | A |
| 3 | Data & representation | 1:00 | B |
| 4 | Method | 1:45 | B |
| 5 | Experimental setup | 1:15 | C |
| 6 | Results | 2:00 | C |
| 7 | Analysis | 1:30 | D |
| 8 | Conclusion | 0:35 | D |

Full spoken script: `docs/TALK_SCRIPT.md`. Deck previews:
`deliverables/slides-research.html`, `deliverables/slides-preview.html`.
