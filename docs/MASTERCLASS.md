# Masterclass — Building Age Classification, Read Against the Rubric

This document does three things:

1. Decodes the **evaluation rubric** (`docs/Evaluation Rubric Overview.docx`) and what
   scores each point.
2. Gives a **deep technical masterclass** on the project — every stage, every important
   variable explained in brackets the first time it appears.
3. Maps concrete **moves that earn rubric points** and **mistakes that lose them**.

Companion docs: [PROJECT_GUIDE.md](PROJECT_GUIDE.md) (plain-language overview),
[METRIC_LEARNING_APPROACH.md](METRIC_LEARNING_APPROACH.md) (the suggested transfer method).

---

## Part 1 — The rubric, decoded

Total **100 points**, three scored dimensions plus deductions. Performance bands are
roughly: **< 50% = Poor/Minimal**, **50–75% = Satisfactory/Good**, **> 75% = Exceptional**.

| # | Dimension | Points | One-line summary |
|---|-----------|:-----:|------------------|
| 1 | Challenge Understanding & Transfer Strategy | **30** | Do you actually understand cross-city domain shift, and does your method address it? |
| 2 | Originality & Solution Creativity | **40** | Is the approach novel and well-reasoned — not an off-the-shelf classifier? |
| 3 | Presentation & Team Collaboration | **30** | Clear story, honest interpretation of the F1 curve, everyone contributes and speaks. |
| — | Deductions | up to −25 or DQ | Word-count breach, missing pillar, data leakage, or "not actually transfer learning". |

### Dimension 1 — Challenge Understanding & Transfer Strategy (30)

Three sub-questions the judges ask:

- **Model Design & Adaptation** — does the method *explicitly* target the mismatch between
  the *source distribution* (Madrid: `P_madrid(features, label)` — the joint distribution
  of pixel features and age classes in the training city) and the *target distribution*
  (Amsterdam: `P_amsterdam(features, label)` — the same joint distribution in the transfer
  city)? These two differ because Amsterdam brick/maritime climate produces different
  reflectance than Madrid concrete/Mediterranean climate. That difference is the
  **domain shift** (also called *covariate shift* when only the feature distribution moves,
  *concept shift* when the feature→label mapping itself moves — here you have both).
- **Spectral Representation** — are your features *city-invariant* (they describe *ageing*
  in a way that holds anywhere) rather than *city-specific* (they memorise "this is what
  Madrid looks like")? A feature like "absolute mean Blue reflectance" is city-specific;
  a feature like "slope of NDVI over 40 years" is closer to city-invariant.
- **Low-Data Mechanics** — can you explain *why* the 25-labels-per-class regime needs a
  different mechanism than full-data training? (Answer: with 25 examples you cannot fit a
  full model without overfitting; you can only *nudge* an existing representation — e.g.
  recompute class centres, fit a 1-layer head, or shift a decision boundary. This is the
  whole point of the "Best low-data F1" prize.)

**Exceptional looks like:** a written dissection of *which* features transferred and which
didn't, backed by a plot (e.g. embedding of both cities coloured by class), plus a
reproducible pipeline with a fixed seed (`random_state` — the integer that makes every
random split, shuffle, and model init repeatable).

### Dimension 2 — Originality & Solution Creativity (40) — the biggest lever

Explicit judge note: **a team with lower F1 can still win if the method is novel,
well-reasoned, or opens a research direction.** So this is where a strong idea, cleanly
argued, beats a marginally higher score.

- **Methodological Innovation** — unconventional architecture, *loss function* (the
  quantity the model minimises during training — e.g. cross-entropy vs. triplet loss vs. a
  domain-adversarial loss), or domain-adaptation trick.
- **Problem Framing** — a genuinely different angle on the cross-city problem.
- **Insightful Failure** — if a bold approach scores lower, extract *why* and state the
  scientific lesson.

### Dimension 3 — Presentation & Team Collaboration (30)

- **Abstract & Pitch** — abstract ≤ 150 words (per Notebook 1), concise and matching the
  talk.
- **Cohesion** — visibly shared work; **every member speaks** during the presentation.
- **Communication** — explain hard ideas within the time limit; "masterful interpretation
  of performance curves" means you can read the **F1-vs-sample-size curve** (macro F1 on
  the y-axis, labels-per-class `n` ∈ {5, 25, 50, 100, 200} on the x-axis, usually plotted
  against `log2(n)`) and say what its shape *means*.

### Deductions — do not lose free points

| Trigger | Penalty |
|---------|---------|
| Written justification exceeds **500 words** (Notebook 1 says 300 — **use the stricter 300 unless organisers confirm 500**) | up to −5 |
| Presentation or justification **omits any of the three pillars** (Model Design, Transfer Strategy, F1 Interpretation) | up to −5 |
| **Data leakage** — method not scientifically sound | up to −20 or **disqualification** |
| **Off-topic** — e.g. no transfer learning actually performed | up to −20 or **disqualification** |
| Unprofessional conduct | up to −20 or DQ |

**Data leakage** (the single most dangerous item) = letting information from the test set,
or from Amsterdam labels you're not supposed to have, influence training. Concrete leaks
to avoid:

- Fitting the `StandardScaler` (the transformer that subtracts each feature's mean and
  divides by its standard deviation) on Madrid **+** Amsterdam together. Fit on the
  training portion only. Notebook 3 already does this correctly: `scaler.fit(X_madrid)`
  then `scaler.transform(X_amsterdam)`.
- Selecting features or tuning hyper-parameters using the full Amsterdam labelled set
  instead of only the `n`-per-class few-shot sample.
- Computing gap-fill interpolation or temporal statistics across the CV boundary in a way
  that mixes train and test pixels. (Per-pixel aggregation is safe here because each pixel
  is one independent row; just keep whole pixels on one side of each split.)
- Cross-validation that splits *rows* before aggregation so the same pixel appears in both
  train and test — always split on `px_key`/`py_key` (the geographic grid indices that
  identify a physical location) *after* you're down to one row per pixel.

---

## Part 2 — Technical masterclass

### 2.1 The problem, stated precisely

Given a 30 m × 30 m ground pixel that contains buildings, predict its **age class**
`y ∈ {1, 2, 3, 4}` (an integer bucket of building construction era) from a **feature
vector** `x ∈ ℝ⁶⁰` (60 real numbers summarising ~40 years of satellite measurements —
built in Notebook 3).

- Train on **Madrid** (`X_madrid` — an `(n_madrid, 60)` array of features;
  `y_madrid` — length-`n_madrid` array of class labels).
- Adapt to **Amsterdam** (`X_amsterdam`, `y_amsterdam`) using only `n` labelled pixels
  per class, for `n ∈ {5, 25, 50, 100, 200}`.
- Report **macro F1** (the F1 score computed per class then averaged with equal weight, so
  the rare classes count as much as the common ones — F1 itself is the harmonic mean of
  *precision* = of the pixels I called class k, how many were; and *recall* = of the true
  class-k pixels, how many I found).

### 2.2 Why age is even visible from space

Landsat measures **reflectance** (fraction of incoming sunlight bounced back, per
wavelength band) in 6 bands: **Blue, Green, Red, NIR** (near-infrared),
**SWIR1, SWIR2** (short-wave infrared). Construction materials and their ageing —
weathering, soiling, roof replacement, vegetation growing around older buildings — change
that reflectance. The signal per year is weak and noisy; the **40-year trajectory** is
where the information lives.

Two physical anchors the notebooks lean on:

- **The satellite record starts in 1984.** Classes 3 (built 1984–2004) and 4 (built
  2004–2024) have their construction event *inside* the record, so a step-change in
  reflectance can be directly visible. Classes 1 and 2 (pre-1984) show only slow ageing.
- **`weighted_mean_year`** (the area-weighted mean construction year of every building
  footprint overlapping the pixel — "area-weighted" means a footprint covering 60% of the
  pixel counts three times as much as one covering 20%) is the continuous label from which
  the discrete `age_class` is cut.

### 2.3 The data object

Wide format: **one row = one `(pixel, year)`**. Key columns:

| Variable | Meaning (bracketed detail) |
|----------|---------------------------|
| `pixel_id` | string `"{city}_{px_key}_{py_key}_{year}"` — unique per row |
| `px_key`, `py_key` | integer grid indices (UTM easting/northing ÷ 30 m, rounded — stable identifier for a physical 30 m cell across all years and satellite passes) |
| `pixel_x`, `pixel_y` | float UTM coordinates in metres of the pixel centre (for plotting maps) |
| `coverage` | fraction of the pixel's area covered by building footprints (0–1; dataset pre-filtered to ≥ 0.15; higher ⇒ the age label is more trustworthy) |
| `weighted_mean_year` | the continuous construction-year label (see 2.2) |
| `Blue_1 … SWIR2_3` | reflectance for observation slot 1/2/3 (up to 3 summer scenes per year, sorted earliest-to-latest by day-of-year) |
| `qa_valid_1 … 3` | boolean — was that slot a clean, non-cloud, non-shadow, non-fill pixel |
| `doy_N`, `scene_id_N`, `dataset_N` | day-of-year, Landsat scene ID, and collection (`landsat_tm_c2_l2` = Landsat 5; `landsat_ot_c2_l2` = Landsat 8/9) for slot N |

Harmonisation note: bands from different sensor generations are put on one scale using the
**Roy et al. (2016) linear correction** (per-band `slope · x + intercept` that maps
Landsat 5 TM reflectance onto the Landsat 8 OLI scale), so a 1990 value and a 2020 value
are comparable.

### 2.4 Notebook 2 — exploration (what it establishes)

- **`age_class`** is derived by `pd.cut(weighted_mean_year, bins)` with **city-specific
  bins** held in `CLASS_EVENTS` (a dict `{'Amsterdam': 1945, 'Madrid': 1960}` — the year
  that splits class 1 from class 2 in each city; classes 3/4 always split at 1984/2004/2024).
  Madrid 1960 = the *Planes de Desarrollo* industrialisation; Amsterdam 1945 = end of
  occupation and the *wederopbouw* reconstruction. Different construction typologies ⇒
  different spectra.
- **`df_flat`** (also `df_ams_flat` for Amsterdam) — the "flat view": one reflectance value
  per band per row, taken from the first `qa_valid` slot (obs_1, else obs_2, else obs_3).
  Rows with no valid slot are dropped.
- **`BLUE_MAX = 15_000`** — an atmospheric-outlier threshold. A batch of 2003 Landsat 7
  Amsterdam scenes (just before the Scan Line Corrector failure) has haze that the standard
  `qa_valid` flag misses, pushing Blue reflectance as high as ~64,000 (normal urban Blue
  peaks near 10,000). `make_flat_view()` (the helper that builds `df_flat`) removes every
  row whose best-valid Blue exceeds `BLUE_MAX`. **You must apply the identical filter in
  your own preprocessing** or a single spike will distort per-pixel means.
- **Class imbalance** — the notebook text claims Class 1 dominates. **The actual counts
  disagree**: Class 2 is the largest in both cities, and Class 1 is the *smallest* in
  Madrid.

  | Class | Madrid | Amsterdam |
  |:-----:|-------:|----------:|
  | 1 | 13,794 (18.1%) | 7,568 (**29.1%**) |
  | 2 | **27,152 (35.6%)** | **8,802 (33.9%)** |
  | 3 | 18,672 (24.5%) | 6,745 (25.9%) |
  | 4 | 16,645 (21.8%) | 2,877 (**11.1%**) |

  This explains Notebook 4's Class 2 over-prediction exactly. **Verify this yourself and
  say it in the presentation** — catching an error in the provided material is cheap
  evidence of genuine understanding.

- **Label shift is a second domain shift.** Beyond the *feature* distributions differing
  between cities (covariate shift), the **class priors differ too**: Class 1 is 1.6× more
  common in Amsterdam, Class 4 less than half as common. This is *prior shift* (a.k.a.
  label shift), and it is a distinct problem with its own literature and its own fix
  (prior correction / re-weighting predicted probabilities by the target prior ratio).
  Naming it separately — and noting that a prototype classifier is implicitly
  prior-agnostic, which is part of why few-shot transfer helps so much — is precisely the
  "deep insight into cross-city domain shift" the rubric's top band asks for.
- **What the plots say about features** (`BASE_BANDS = ['Blue','Green','Red','NIR','SWIR1','SWIR2']`):
  - per-pixel **mean reflectance** rises monotonically Class 1 → Class 4 in every band —
    the strongest single signal;
  - per-pixel **std of reflectance** follows the same ordering;
  - per-pixel **std of the year-on-year difference** (value in year `t` minus year `t−1`)
    is narrow for Classes 1–2, broad for Classes 3–4 — the split lands exactly at 1984;
  - Blue/Green/Red are mutually highly correlated (redundant); NIR/SWIR add independent
    information — so use all six, not RGB.

### 2.5 Notebook 3 — feature engineering (the heart of the pipeline)

**Design decision: one row per pixel, not per `(pixel, year)`.** Feeding yearly rows makes
the model treat each year as an independent sample, hides the trajectory, and leaks
temporal structure across a train/test split. Collapsing to per-pixel temporal statistics
forces the model to reason about *change over time*.

Pipeline:

1. **Load** `madrid_train.parquet`, `amsterdam_data.parquet` from `DATA_DIR` (the string
   `'../data'` — correct when the notebook runs from `notebooks/`).
2. **`assign_age_class`** — the `pd.cut` bucketing from `CLASS_EVENTS`.
3. **`get_best_obs_band` / `make_flat_view`** — flatten to one clean obs per band-year;
   drop `Blue > 15_000` haze rows.
4. **`has_early_data`, `has_late_data`** — booleans computed from the *observed* years
   *before* gap-filling: did this pixel have at least one real scene in 1984–2003 /
   in 2004+? (Almost always true; flags the rare all-cloud pixel so its early/late stats
   aren't silently fabricated.)
5. **`fill_time_series_gaps`** — reindex each pixel to every year from `year_min` to
   `year_max`; interior missing years → **linear interpolation** (draw a straight line
   between the two surrounding observed values); years before the first observation →
   **back-fill** (copy the first observed value); after the last → **forward-fill**. After
   this, every `(pixel, year)` cell is populated, so downstream statistics are always
   defined and no imputation flag is needed on the feature matrix.
6. **`compute_spectral_indices`** — five normalised band ratios, each in roughly `[-1, 1]`
   and robust to scene-to-scene brightness differences:

   | Index | Formula | Captures |
   |-------|---------|----------|
   | **NDVI** | `(NIR − Red)/(NIR + Red)` | vegetation (low ⇒ built-up) |
   | **NDBI** | `(SWIR1 − NIR)/(SWIR1 + NIR)` | built-up intensity |
   | **UI** | `(SWIR2 − NIR)/(SWIR2 + NIR)` | urban-ness |
   | **MNDWI** | `(Green − SWIR1)/(Green + SWIR1)` | water (negative on land) |
   | **BSI** | `((SWIR1+Red) − (NIR+Blue)) / ((SWIR1+Red) + (NIR+Blue))` | bare soil vs vegetation |

   (`eps = 1e-6` is added to denominators to avoid divide-by-zero.)
7. **`build_pixel_features`** — aggregate each pixel's gap-filled series into **60
   features** (`FEATURE_COLS` — the ordered list of column names; `PID = ['px_key','py_key']`
   is the per-pixel group key):

   | Group | Features | Count |
   |-------|----------|:-----:|
   | Overall mean, per band | `{band}_mean` | 6 |
   | Overall std, per band | `{band}_std` | 6 |
   | Overall mean, per index | `{index}_mean` | 5 |
   | Overall std, per index | `{index}_std` | 5 |
   | Early-period (1984–2003) mean, per band | `{band}_early_mean` | 6 |
   | Early-period std, per band | `{band}_early_std` | 6 |
   | Late-period (2004+) mean, per band | `{band}_late_mean` | 6 |
   | Late-period std, per band | `{band}_late_std` | 6 |
   | Year-on-year change mean, per band | `d_{band}_mean` | 6 |
   | Year-on-year change std, per band | `d_{band}_std` | 6 |
   | Coverage indicators | `has_early_data`, `has_late_data` | 2 |
   | **Total** | | **60** |

   **Why 2004 as the early/late split:** 1984–2003 captures the *pre-existing* land use at
   that location; 2004+ captures the *current* building. For Classes 1–2 both windows
   describe the same structure (signals stable). For Classes 3–4 the windows straddle the
   construction event (early = field / demolished building / construction site; late =
   finished building), so the *early-vs-late contrast* is maximally class-discriminative.
   **Why year-on-year change:** its std cleanly separates pre-1984 (settled) from
   during-record (transient) construction — a direct time-series signal.
8. **Standardise** — `X_madrid = scaler.fit_transform(features_madrid[FEATURE_COLS])`,
   `X_amsterdam = scaler.transform(features_amsterdam[FEATURE_COLS])`. Fit on Madrid only.
   Essential for the distance-based transfer step (2.6.7); a feature on a larger numeric
   scale would otherwise dominate Euclidean distance.
9. **Save** `preprocessed_data.pkl` — a dict with `X_madrid, y_madrid, X_amsterdam,
   y_amsterdam, feature_cols, scaler`, plus `pixel_ids_*` for map plots.

### 2.6 Notebook 4 — modelling & evaluation

Approach label: **metric learning** — learn a feature space where same-age buildings sit
close together regardless of city, so a new city only needs its class centres located.
Baseline model: an untuned **`RandomForestClassifier`** (an ensemble of decision trees,
each trained on a bootstrap sample with random feature subsets; prediction = majority vote;
`RF_PARAMS` holds the constructor arguments, e.g. `n_estimators` = number of trees).

1. **Load** `preprocessed_data.pkl`.
2. **Cross-validation on Madrid** — `RepeatedStratifiedKFold(n_splits=N_FOLDS,
   n_repeats=N_REPEATS, random_state=…)`:
   - **fold** = one of `k` equal parts; train on `k−1`, score the held-out 1; rotate so
     every pixel is tested once ⇒ `k` scores.
   - **stratified** = each fold keeps the same class proportions as the whole set (so a
     rare class isn't missing from a fold).
   - **repeat** = redo the whole `k`-fold split with a fresh shuffle. The `k` scores within
     one repeat are *correlated* (any two folds share `(k−2)/k` of their training data), so
     their spread *understates* true run-to-run variance. Independent repeats fix the error
     bar. `k·r` scores total (`all_f1` — the list of macro-F1 values; `total_fits` = `k·r`).
   - Outputs: mean macro-F1 ± std, and a **confusion matrix** (`cm[i, j]` = count of true
     class `i+1` predicted as class `j+1`; row-normalised gives per-class **recall** on the
     diagonal).
   - **Finding:** F1 stable to ≈ ±0.01; **Class 2 over-predicted** — many true Class 1
     pixels labelled Class 2 because imbalance isn't corrected. Fixing this is left to you.
3. **Spatial uncertainty mapping** — `rf.predict_proba(X)` returns `(p1, p2, p3, p4)` per
   pixel (fraction of trees voting each class; sums to 1; `predict` = `argmax`). Mapped
   geographically via two measures:
   - **Shannon entropy** `H = −Σ pₖ log₂ pₖ` (bits; 0 = certain, 2 = uniform guess over
     4 classes) — how unsure the model is, label-free.
   - **Negative log-prob of the true class** `−log₂ p_true` (bits; 0 = confident *and*
     correct; large = wrong or unsure) — needs labels; equals the KL divergence
     (a measure of how far the predicted distribution is from the true one-hot
     distribution) from the true class.
   - **Finding:** even at modest accuracy the model gets the dominant-class *neighbourhoods*
     right ⇒ a second-stage model over the probability maps could improve things.
4. **Final Madrid model** — refit a Random Forest on *all* Madrid data (`rf_final`); this
   is what transfers. In-sample F1 ≈ 1.0 is expected (the forest memorises training rows)
   and is *not* a performance estimate — the CV number from step 2 is.
5. **Feature importance** — `rf_final.feature_importances_` (mean decrease in Gini impurity
   contributed by splits on each feature). Result: many features nearly tied, partly
   because they're correlated, partly because the forest exploits complex interactions.
6. **Zero-shot transfer** — `rf_final.predict(X_amsterdam)` with **no** Amsterdam labels.
   `f1_zero` = macro F1 of that. This is the **baseline domain gap**. **Finding:** poor,
   Class 2 wildly over-predicted — expected, and the motivation for few-shot.
7. **Few-shot prototype transfer** — the payoff:
   - `prototype_predict(X_support, y_support, X_query)`: for each class `c`, the
     **prototype** `prototypes[c] = X_support[y_support == c].mean(axis=0)` (mean feature
     vector of the labelled examples — `X_support`/`y_support` are the `n`-per-class sample;
     `X_query` is everything else). Predict `argmin` Euclidean distance to a prototype.
     This is 1-NN against a *smoothed* class representative; averaging beats raw k-NN when
     labels are scarce because it cancels noise.
   - `SHOTS_PER_CLASS` = the list `[5, 25, 50, 100, 200]` (labels per class, `n`). Each `n`
     is repeated ~10× with different random support draws (`proto_results[n]` = list of
     F1s) to get error bars.
   - `prototype_predict_proba` — soft version: `softmax(−distances)` (turns the negative
     distances into a probability vector) for uncertainty maps.
   - **Findings:** 5/class already crushes zero-shot; 25/class adds ~10 points (this is the
     "Best low-data F1" operating point); ~100/class locates the centres well, beyond that
     is marginal; Amsterdam per-class recall ends up ≈ Madrid's.
8. **Summary + improvement ideas** (from the notebook itself — explicitly unverified):
   - *Features:* per-decade stats instead of two periods; linear-trend **slope** per band;
     year of max/min reflectance; change-point year (largest single-year jump).
   - *Metric learning:* a real embedding network trained with **triplet loss**
     (`max(0, d(a,p) − d(a,n) + margin)` over triplets of anchor `a`, same-class positive
     `p`, different-class negative `n` — pulls same-age together, pushes different-age
     apart); Random-Forest **leaf-node co-occurrence** as a similarity kernel
     (`rf.apply(X)` returns each sample's leaf id per tree; two samples landing in the same
     leaf often are "similar"); **domain-adversarial** training (add a branch that tries to
     predict *city* from the features and train the encoder to *defeat* it, so the
     representation carries no city information).
   - *Transfer:* fine-tune on the Amsterdam labels instead of prototypes; self-train with
     pseudo-labels (trust the zero-shot model's confident Amsterdam predictions as
     provisional labels and retrain).

### 2.7 Reading the F1-vs-sample-size curve (Dimension 3 asks for this)

Plot macro F1 (mean ± std over the ~10 repeats) against `log2(n)` for
`n ∈ {5, 25, 50, 100, 200}`, with the **Madrid CV F1** drawn as a horizontal reference
line (the practical ceiling — same-city performance).

What the shape tells you:

| Observation | Interpretation |
|-------------|----------------|
| Big jump from zero-shot to `n = 5` | The Madrid representation *is* reused; only the class *locations* were wrong, and a handful of labels fixes that. Evidence the features transfer. |
| Curve flattens by `n ≈ 100` and sits **below** the Madrid line | Residual gap = genuinely city-specific structure the shared representation can't capture. Name it (which classes? see the confusion matrix). |
| Curve reaches the Madrid line | Representation transferred essentially fully; the only cost was locating centres. |
| Wide error bars at small `n` | Few-shot result is draw-dependent — report the std honestly; it's expected and not a flaw. |
| `n = 25` point unusually high or low | This is the judged low-data operating point — comment on it specifically. |

---

## Part 3 — Turning the project into points

### Move the score up

| Rubric target | Concrete action |
|---------------|-----------------|
| **D1 · domain shift** | Show a 2-D embedding (PCA / t-SNE / UMAP) of Madrid+Amsterdam features coloured by class *and* by city. If cities separate more than classes do, you've *shown* the shift — then show your method reduces it. |
| **D1 · city-invariant features** | Prefer shape-of-trajectory features (slopes, change-point timing, ratios, within-pixel deltas) over absolute levels. Report a before/after: F1 with vs. without the level features. |
| **D1 · low-data mechanics** | State plainly why 25/class ⇒ "adjust representation, don't refit it", and tie each transfer variant to a label budget. |
| **D2 · innovation (40 pts!)** | Pick *one* non-baseline idea and execute it cleanly: domain-adversarial encoder, triplet-loss embedding, RF-leaf kernel + few-shot, CORAL/MMD feature alignment, or a probability-map second stage. Argue the reasoning even if F1 doesn't beat the baseline. |
| **D2 · insightful failure** | If the bold method underperforms, dedicate a slide to *why* (e.g. "adversarial term collapsed Class 3/4 separation because construction-era signal *is* partly city-specific"). |
| **D3 · curve interpretation** | Use the table in 2.7. Say what the plateau height and the zero-shot→5 jump *mean*, not just "it goes up". |
| **D3 · cohesion** | Split ownership (data/features, model, transfer, presentation) and make sure each person presents their part. |

### Don't lose points

- **Word cap:** keep the written justification to **300 words** (Notebook 1's number;
  the rubric says 500 — follow the stricter one unless organisers say otherwise). Cover
  all three pillars explicitly: *Model Design*, *Transfer Strategy*, *F1 Interpretation*.
- **Fixed seed:** one `random_state` constant threaded through every split, shuffle,
  sampler, and model. Code must run end-to-end.
- **No leakage:**
  - `scaler` and any feature selection / hyper-parameter tuning fit on **training data
    only** (Madrid for Stage 1; the `n`-per-class support set for Stage 2 — never the full
    Amsterdam labels).
  - CV splits on **pixels** (`px_key`,`py_key`), after aggregation, never on raw
    `(pixel, year)` rows.
  - Keep the `Blue > 15_000` haze filter — an un-filtered spike silently corrupts a
    pixel's mean/std features.
  - Report the CV estimate as the headline number, never the ≈1.0 in-sample F1.
- **Stay on topic:** you must actually *perform transfer* — a model trained directly on
  Amsterdam labels is disqualifiable. The Stage 1 model must be Madrid-only; Stage 2 must
  adapt it with the small labelled sample.
- **Deliverables:** Stage 1 trained model, parameterised end-to-end adaptation
  code/notebook (sample size `n` a single variable), slide deck (≤150-word abstract on
  slide 1, results table of the Madrid F1 + five Amsterdam F1s with error bars, the
  F1-vs-`log2(n)` plot), and the written justification.
