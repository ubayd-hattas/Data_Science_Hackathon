# Experiment log

## A003 final critical audit

- Completed the independent audit in `Docs/FINAL_CRITICAL_AUDIT.md`, including the teammate's two-tier RF, score-product and LDA files. Historical experiment entries below are preserved.
- Ran `src/final_audit_verify.py`: all 184 saved summary-vector arithmetic checks passed, all 50 saved support/query sets passed integrity checks, and all 100 raw prototype/logistic trial scores reproduced exactly from cached features. Recomputed group shift and 25-shot logistic aggregate class metrics matched saved outputs. Evidence: `outputs/final_audit_verification.json`; complete tables: `Docs/VERIFIED_RESULT_TABLES.md`.
- Revised submission verdict: all-60 logistic is category C as a primary transfer method because it uses only Madrid scaling. The existing two-tier RF is the provisional primary, pending a bounded reproduction and saved-model/unlabelled-inference packaging gate.
- Broad modelling remains frozen. No new model family or targeted candidate was trained during the audit. Re-evaluating existing raw controls was verification, not model exploration.
- Important qualifications: query scores informed model selection; metric-learning failure causes remain hypotheses; feature groups overlap; the support normalisation test primarily changes scaling rather than establishing a general alignment result; teammate RF provenance is incomplete and its model artifact is missing locally.
- Team ownership and 24 judge questions are in `Docs/TEAM_HANDOFF.md`. No PowerPoint, polished abstract or final written justification was created.

## B000 organiser baseline reproduction

- **Status:** completed 2026-09-08.
- **Purpose:** reproduce the supplied Notebook 3 preprocessing and Notebook 4 Random Forest/prototype baseline without changing the organiser notebooks or data.
- **Command:** `.venv/Scripts/python.exe src/organiser_baseline.py --folds 5 --repeats 5 --trees 500 --trials 10 --seed 42 --n-jobs -1`
- **Runtime:** 2,704.4 seconds.
- **Environment:** Python 3.13.13, NumPy 2.5.3, pandas 3.0.5, Matplotlib 3.11.1, scikit-learn 1.9.0, PyArrow 25.0.1.
- **Organiser notebook kernel metadata:** Python 3.11.15 in a kernel named `landsat`; package versions are not recorded in the notebooks.
- **Data hashes:** Madrid `0E106A07044D867F8C68E77DE8C36826C491D0F26692B8CDE753D1046CE24C9C`; Amsterdam `32388F3A9792B01D9831CA4D0FC4BFADBA132A1C23C06E372D7A82E985BE0719` (SHA-256).

### Observation construction and features

One model observation is one geographic 30 m pixel represented by its complete 1984-2025 annual spectral history. For each pixel-year the first QA-valid slot among observations 1, 2, and 3 is selected. Rows with no complete six-band valid observation are dropped, and rows whose selected Blue value exceeds 15,000 are removed. Missing years are linearly interpolated within a pixel, with nearest-value backward/forward filling at the edges.

The model uses 60 features:

- Six bands: Blue, Green, Red, NIR, SWIR1, SWIR2.
- Five per-year indices: NDVI, NDBI, UI, MNDWI, and BSI.
- Overall mean and standard deviation for all six bands and five indices: 22.
- 1984-2003 band means and standard deviations: 12.
- 2004-2025 band means and standard deviations: 12.
- Mean and standard deviation of annual first differences for each band: 12.
- `has_early_data` and `has_late_data`: 2.

Preprocessing produced 76,263 Madrid pixel observations and 25,992 Amsterdam observations. It removed 34,845 Madrid and 12,917 Amsterdam selected observations above the Blue threshold, then produced 3,203,046 and 1,091,664 gap-filled pixel-year rows respectively.

### Model and evaluation procedure

- **Scaler:** `StandardScaler`, fitted once on all Madrid pixel features, then applied to both cities. This mirrors Notebook 3 but leaks Madrid validation-fold distribution statistics into CV.
- **Source model:** `RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)`.
- **Madrid split:** `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)` at one-row-per-pixel level.
- **Madrid metric:** for each class, `F1 = 2 * precision * recall / (precision + recall)`; macro F1 is the unweighted mean of the four class F1 values. It is calculated per fold. Reported error is population SD across the 25 fold scores.
- **Final source fit:** Random Forest fitted on all scaled Madrid pixels.
- **Zero-shot transfer:** apply that fitted forest directly to all Amsterdam pixels.
- **Few-shot transfer:** for each Amsterdam budget and each of 10 sequential seeded draws, sample `n` support pixels per class without replacement, calculate each class mean in the scaled 60-feature space, classify every non-support Amsterdam pixel by nearest Euclidean prototype, and report macro F1. Support and query sets are disjoint within each trial.

### Results

| Evaluation | Fresh reproduction macro F1 | Organiser notebook stored macro F1 |
|---|---:|---:|
| Madrid 5-fold x 5-repeat CV | 0.6281 +/- 0.0043 | 0.6179 +/- 0.0043 |
| Amsterdam zero-shot | 0.4433 | 0.3427 |
| Amsterdam 5/class | 0.5437 +/- 0.0324 | 0.5437 +/- 0.0324 |
| Amsterdam 10/class (extra, not a stated required budget) | 0.5649 +/- 0.0312 | 0.5649 +/- 0.0312 |
| Amsterdam 25/class | 0.5986 +/- 0.0176 | 0.5986 +/- 0.0176 |
| Amsterdam 50/class | 0.6079 +/- 0.0101 | 0.6079 +/- 0.0101 |
| Amsterdam 100/class | 0.6129 +/- 0.0097 | 0.6129 +/- 0.0097 |
| Amsterdam 200/class | 0.6150 +/- 0.0036 | 0.6150 +/- 0.0036 |

Fresh Madrid in-sample accuracy was 0.99995. The full fold vector and accumulated confusion-matrix counts are in `outputs/baseline_results.json`.

Fresh Madrid CV row-normalised confusion-matrix proportions (rows are actual classes; columns are predicted classes):

| Actual / predicted | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| C1 | 0.5307 | 0.3614 | 0.0623 | 0.0456 |
| C2 | 0.1412 | 0.6819 | 0.1142 | 0.0627 |
| C3 | 0.0436 | 0.2424 | 0.5885 | 0.1255 |
| C4 | 0.0217 | 0.1179 | 0.1694 | 0.6911 |

### Interpretation and reproducibility notes

- All Amsterdam prototype means and SDs reproduce the stored notebook values exactly to four decimals. Dataset shapes, class counts, haze-removal counts, gap-filled row counts, and feature dimensions also match. This validates the organiser feature and prototype pipeline.
- Fresh Random Forest results do not reproduce the stored notebook values: Madrid CV is +0.0102 macro F1 and Amsterdam zero-shot is +0.1006. The notebook was run under Python 3.11.15 but does not record NumPy, pandas, or scikit-learn versions. Sequential execution counts show its outputs were produced in order. The discrepancy is therefore an unresolved environment/implementation-version sensitivity, not random-seed drift.
- The prototype stage does not use the fitted Madrid Random Forest or a Madrid-learned embedding/metric. It reuses only the Madrid-fitted scaler, so its claim to be a transferred learned representation is weak under the rubric.
- The same Amsterdam dataset supplies support samples and internal query samples. These are index-disjoint within a trial, but nearby pixels can cross the boundary and repeated trials reuse pixels in different roles.
- The 25 repeated-CV fold scores are not independent: folds share training data, and repeats reuse the same pixels. Their SD is a dispersion summary, not a standard error or independent uncertainty estimate.
- The organiser split is non-spatial. In the audited 5-fold split, 55,796 of 69,725 orthogonally adjacent Madrid pixel pairs (80.0%) cross fold boundaries.
- No exact raw-row duplicates, pixel-year duplicates, or varying per-pixel targets were found in either parquet file.

## M001 Madrid to Amsterdam triplet metric learning

- **Status:** completed 2026-09-08.
- **Purpose:** test whether a Madrid-supervised triplet embedding improves few-shot Amsterdam classification over raw organiser features.
- **Command (default):** `.venv/Scripts/python.exe src/metric_learning_experiment.py`
- **Direction:** Madrid labelled features -> train embedding -> embed Amsterdam -> adapt using Amsterdam support labels -> evaluate on disjoint Amsterdam query pixels.
- **Shared evaluation:** 10 deterministic trials (seed 42) at 5, 25, 50, 100, and 200 support pixels per class. Every method uses the exact same saved support/query indices in `outputs/amsterdam_fixed_splits.npz`.
- **Metric:** four-class macro F1 on the query set; reported uncertainty is population SD across the 10 fixed trials.
- **Features:** the exact organiser 60-feature matrices, scaled with a `StandardScaler` fitted on Madrid only, cached in `outputs/features_60.npz`.
- **Raw logistic:** Amsterdam-support-only `LogisticRegression(C=1.0, max_iter=2000)`.
- **Embedding:** 60 -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(16) -> L2 normalization. NumPy implementation, Adam (learning rate 0.001), squared-Euclidean triplet loss, margin 0.2, 100 epochs, 32 batches/epoch, 256 triplets/batch.
- **Default triplets:** uniformly sampled Madrid anchor classes, random same-class positive, and random different-class negative. Amsterdam labels are never used in embedding training; query labels are used only for evaluation.

### Fair-split Amsterdam results

| Labels/class | Raw prototype | Raw logistic | Embedding prototype | Embedding logistic |
|---:|---:|---:|---:|---:|
| 5 | 0.5437 +/- 0.0324 | **0.5457 +/- 0.0446** | 0.5167 +/- 0.0585 | 0.4618 +/- 0.0572 |
| 25 | 0.5960 +/- 0.0222 | **0.6187 +/- 0.0227** | 0.5804 +/- 0.0253 | 0.5584 +/- 0.0393 |
| 50 | 0.6052 +/- 0.0133 | **0.6467 +/- 0.0092** | 0.5826 +/- 0.0215 | 0.5788 +/- 0.0139 |
| 100 | 0.6111 +/- 0.0103 | **0.6679 +/- 0.0095** | 0.5873 +/- 0.0129 | 0.5974 +/- 0.0086 |
| 200 | 0.6152 +/- 0.0050 | **0.6867 +/- 0.0056** | 0.5911 +/- 0.0037 | 0.6074 +/- 0.0042 |

The organiser-style raw prototype values differ slightly from B000 at budgets above 5 because this experiment creates and persists a new fixed split set containing exactly the five required budgets. B000's sequential random stream also drew the extra 10/class condition before drawing its 25/class and larger conditions.

### Minimal adjustment

Because the default embedding lost to the raw representation, one permitted adjustment was tested: online selection of the closest negative from eight randomly drawn different-class Madrid candidates, with every other setting unchanged. It collapsed toward the margin (final loss 0.2001) and degraded embedding-prototype F1 to 0.4003, 0.4731, 0.4804, 0.4800, and 0.4914 across the five budgets. It was rejected. The full result is retained in `outputs/metric_learning_results_hard_negative_8.json`; no further tuning was performed.

### Interpretation

- The Madrid-trained embedding did **not** improve Amsterdam at any budget. Against the central raw-prototype comparison, its deficits were 0.0270, 0.0156, 0.0226, 0.0238, and 0.0241 macro F1 from 5 through 200 labels/class.
- Metric learning did not help most in the low-data regime: at 5/class, embedding prototype scored 0.5167 versus 0.5437 raw, and embedding logistic scored 0.4618 versus 0.5457 raw.
- At 25/class, the default embedding-prototype aggregate confusion matrix gives approximate per-class F1 values of C1 0.677, C2 0.490, C3 0.593, and C4 0.563. Class 2 remains the clearest failure, with substantial confusion into class 1 and class 3.
- The PCA view shows an age-related continuum but extensive class overlap and a different Amsterdam density along that continuum. A Madrid-only class-separation objective has no direct pressure to remove city-specific effects. The city-dependent first boundary (Madrid 1960 versus Amsterdam 1945), random triplet supervision, spectral domain shift, and information loss from compressing 60 scaled features to a unit-normalized 16-vector are plausible causes.
- Raw Amsterdam-support logistic regression is the strongest tested method at 25/class (0.6187 +/- 0.0227) and overall (0.6867 +/- 0.0056 at 200/class).

### Artifacts

- `src/metric_learning_experiment.py`: minimal end-to-end experiment runner and shared macro-F1 evaluator.
- `outputs/features_60.npz`: reusable Madrid and Amsterdam scaled feature matrices, targets, pixel keys, feature names, and scaler parameters.
- `outputs/amsterdam_fixed_splits.npz`: reusable fixed support/query indices.
- `outputs/metric_learning_results.json`: complete default scores, configuration, loss trace, and embedding-prototype confusion counts.
- `outputs/metric_learning_results_random_negative.json`: preserved copy of the default result.
- `outputs/metric_learning_results_hard_negative_8.json`: the single rejected adjustment.
- `outputs/figures/macro_f1_vs_budget.png`: presentation-ready budget curve.
- `outputs/figures/embedding_pca_by_city.png`: Madrid and Amsterdam embedding PCA coloured by age class.

## F002 feature-group transfer and domain-shift ablation

- **Status:** completed 2026-09-08; broad modelling concluded.
- **Command:** `.venv/Scripts/python.exe src/transfer_ablation_experiment.py`
- **Purpose:** test whether temporal/change information transfers better than absolute spectral appearance, quantify Madrid-Amsterdam shift, and test one evidence-led correction.
- **Reused inputs:** `outputs/features_60.npz` and `outputs/amsterdam_fixed_splits.npz`; no Landsat preprocessing was rerun.
- **Evaluation:** the same ten support/query trials and five required budgets as M001. Each cell below is prototype mean +/- population SD, then logistic mean +/- population SD.

### Feature-transfer results

| Representation | 5/class P / L | 25/class P / L | 50/class P / L | 100/class P / L | 200/class P / L |
|---|---:|---:|---:|---:|---:|
| All 60 | 0.5437 +/- 0.0324 / 0.5457 +/- 0.0446 | 0.5960 +/- 0.0222 / **0.6187 +/- 0.0227** | 0.6052 +/- 0.0133 / **0.6467 +/- 0.0092** | 0.6111 +/- 0.0103 / **0.6679 +/- 0.0095** | 0.6152 +/- 0.0050 / **0.6867 +/- 0.0056** |
| Absolute spectral means (6) | 0.3874 +/- 0.0422 / 0.3810 +/- 0.0423 | 0.4022 +/- 0.0301 / 0.4225 +/- 0.0305 | 0.4035 +/- 0.0146 / 0.4360 +/- 0.0201 | 0.4044 +/- 0.0104 / 0.4264 +/- 0.0208 | 0.4056 +/- 0.0106 / 0.4450 +/- 0.0144 |
| Spectral indices (10) | 0.3891 +/- 0.0458 / 0.4097 +/- 0.0432 | 0.4189 +/- 0.0298 / 0.4713 +/- 0.0219 | 0.4219 +/- 0.0123 / 0.4987 +/- 0.0103 | 0.4243 +/- 0.0073 / 0.5092 +/- 0.0064 | 0.4254 +/- 0.0055 / 0.5201 +/- 0.0069 |
| Temporal variability / SDs (29) | 0.5355 +/- 0.0392 / **0.5476 +/- 0.0358** | 0.5745 +/- 0.0226 / 0.6047 +/- 0.0258 | 0.5813 +/- 0.0147 / 0.6315 +/- 0.0097 | 0.5809 +/- 0.0126 / 0.6449 +/- 0.0112 | 0.5857 +/- 0.0069 / 0.6575 +/- 0.0071 |
| Early-period statistics (12) | 0.4739 +/- 0.0478 / 0.4640 +/- 0.0422 | 0.5090 +/- 0.0199 / 0.5252 +/- 0.0179 | 0.5199 +/- 0.0171 / 0.5418 +/- 0.0146 | 0.5230 +/- 0.0072 / 0.5572 +/- 0.0108 | 0.5220 +/- 0.0073 / 0.5709 +/- 0.0056 |
| Late-period statistics (12) | 0.4303 +/- 0.0546 / 0.4375 +/- 0.0464 | 0.4603 +/- 0.0265 / 0.4864 +/- 0.0241 | 0.4610 +/- 0.0175 / 0.5014 +/- 0.0211 | 0.4533 +/- 0.0163 / 0.5172 +/- 0.0130 | 0.4632 +/- 0.0089 / 0.5296 +/- 0.0094 |
| Year-to-year change (12) | 0.4070 +/- 0.0300 / 0.4026 +/- 0.0251 | 0.4580 +/- 0.0190 / 0.4618 +/- 0.0145 | 0.4560 +/- 0.0180 / 0.4718 +/- 0.0147 | 0.4552 +/- 0.0131 / 0.4755 +/- 0.0139 | 0.4611 +/- 0.0051 / 0.4911 +/- 0.0096 |

The two data-presence flags were characterized but not tested as a standalone classifier: their mean absolute SMD is only 0.027 and two nearly constant binary inputs are not a scientifically credible representation by themselves.

### Madrid-Amsterdam shift

Shift is the absolute difference between the city means divided by pooled within-city SD, averaged across features in each group.

| Feature group | Mean absolute SMD |
|---|---:|
| Absolute spectral means | 1.849 |
| Early-period statistics | 1.366 |
| Late-period statistics | 0.977 |
| Year-to-year change | 0.849 |
| Temporal variability / SDs | 0.674 |
| Spectral indices | 0.593 |
| Data-presence flags | 0.027 |

Absolute means have both the largest measured shift and weak transfer, while variability has lower shift and is the strongest reduced subset. The pattern is not general: year-to-year change and indices still transfer poorly. Across the six predictive groups, shift versus 25/class F1 correlations are Pearson -0.364 / Spearman -0.200 for prototype and Pearson -0.476 / Spearman -0.314 for logistic. This is weak-to-moderate directional evidence, not support for a reliable monotonic relationship. The stated change-transfer hypothesis is rejected.

### Targeted correction

Because large marginal location/scale shifts were observed, one correction was tested: standardise each feature using only the labelled Amsterdam support rows in that trial, then apply those statistics to its query rows. This is inductive and uses no unlabeled query/test features. For logistic regression, corrected scores were 0.5374 +/- 0.0490, 0.6149 +/- 0.0230, 0.6435 +/- 0.0092, 0.6660 +/- 0.0090, and 0.6861 +/- 0.0061 from 5 through 200/class. It did not beat raw logistic at any budget and was rejected.

### Decision

- **Strongest 25/class:** all-60 Amsterdam-support logistic, 0.6187 +/- 0.0227.
- **Strongest overall:** the same method; it is best at four of five budgets and reaches 0.6867 +/- 0.0056 at 200/class.
- **Madrid contribution:** no Madrid-trained classifier or embedding improved Amsterdam-support logistic. The selected pipeline retains only the Madrid-fitted feature scaler.
- **Metric retry:** skipped. No reduced representation beat all 60 features at 25/class, and explicit support normalization did not improve them.
- **Conclusion:** domain shift is substantial, especially in absolute levels, but discriminative utility is complementary across groups and shift magnitude alone does not predict transfer. Freeze broad modelling and proceed to reproducibility validation and packaging.

### Artifacts

- `src/transfer_ablation_experiment.py`
- `outputs/feature_transfer_results.json`
- `outputs/consolidated_results.json`
- `outputs/figures/feature_group_domain_shift.png`
- `outputs/figures/feature_group_shift_vs_transfer.png`
- `outputs/figures/final_comparison_f1_vs_budget.png`
- `outputs/figures/strongest_25_confusion_matrix.png`
- `FINAL_FINDINGS.md`
