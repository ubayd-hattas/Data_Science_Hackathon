# Experiment log

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
