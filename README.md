# Madrid to Amsterdam Building Age Transfer

This repository contains the frozen final submission for the UCT building-age transfer challenge. A 500-tree balanced Random Forest is trained on Madrid labels and reused in Amsterdam through a fixed two-tier adaptation rule. All Amsterdam scores reported here are internal development results on the supplied data, not organiser held-out scores.

## Final reproduced results

| Labels per Amsterdam class | Frozen branch | Macro F1 mean | Population SD |
|---:|---|---:|---:|
| 5 | Score-product | 0.5475 | 0.0371 |
| 25 | Score-product | 0.5872 | 0.0310 |
| 50 | Score-product | 0.5943 | 0.0134 |
| 100 | Modal-leaf agreement | 0.6145 | 0.0075 |
| 200 | Modal-leaf agreement | 0.6188 | 0.0043 |

Madrid Stage 1 random 5-fold x 5-repeat CV, with the scaler fitted inside every fold, is **0.6280 +/- 0.0043 macro F1**. The saved full-Madrid model is `artifacts/stage1_madrid_rf.pkl`.

## Environment

Python 3.13.13 was used for the final run. Install the pinned environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The reproduced run used NumPy 2.5.3, pandas 3.0.5, Matplotlib 3.11.1, scikit-learn 1.9.0, and PyArrow 25.0.1.

## Repository structure

- `src/final_pipeline.py`: frozen feature extraction and train/load/adapt/predict API.
- `src/train_stage1.py`: Stage 1-only training entry point.
- `src/submission_cli.py`: feature extraction and adaptation/prediction CLI.
- `src/reproduce_submission.py`: full fixed evaluation and artifact reproduction.
- `src/smoke_test.py`: held-out organiser-style smoke test.
- `artifacts/stage1_madrid_rf.pkl`: serialized forest, scaler, priors, feature order, configuration, versions, seeds, and hashes.
- `outputs/final_submission/final_results.json`: authoritative machine-readable results.
- `outputs/final_submission/amsterdam_trial_details.npz`: all support/query indices and IDs plus predictions for 50 trials.
- `Docs/FINAL_SUBMISSION_METHOD.md`: final method and interpretation.
- `Docs/FINAL_CLAIMS.md`: presentation-safe claims.
- `Docs/SUBMISSION_CHECKLIST.md`: requirement-to-artifact map.
- `Docs/FROZEN_CONFIGURATION.md`: exact modelling freeze.

## Train and load Stage 1

The final artifact was trained once from the verified cache:

```powershell
$env:PYTHONPATH = "src"
python src/train_stage1.py --cache outputs/features_60.npz --output artifacts/stage1_madrid_rf.pkl
```

Load it in Python without retraining:

```python
from final_pipeline import load_stage1
package = load_stage1("artifacts/stage1_madrid_rf.pkl")
```

## Extract features without labels

Raw input may be Parquet or CSV and must contain complete available annual histories for each included pixel: `city`, `year`, `px_key`, `py_key`, and the available `qa_valid_1..3` plus six spectral `band_1..3` fields. `weighted_mean_year` and `age_class` are not required or accessed.

```powershell
python src/submission_cli.py extract --raw amsterdam_query.parquet --output query_features.parquet
```

The output contains `px_key`, `py_key`, `city`, and the frozen 60 features in exact order.

## Adapt and predict

Support input must contain the 60 features, pixel IDs, and `age_class`. Query input must contain the same 60 features and pixel IDs, with no target field. Exactly `n` support rows are required for each class 1, 2, 3, and 4.

```powershell
python src/submission_cli.py predict `
  --model artifacts/stage1_madrid_rf.pkl `
  --support amsterdam_support_features.parquet `
  --query amsterdam_query_features.parquet `
  --n 25 `
  --output predictions.csv
```

Add `--support-raw` and/or `--query-raw` when passing the supplied raw schema. The output preserves query order and contains `px_key`, `py_key`, and `predicted_age_class`. The code rejects insufficient class counts, duplicate IDs, support/query overlap, malformed features, and labels in query feature tables.

## Reproduce evaluations

Place the supplied parquet files under `Data/` and the verified caches under `outputs/`, then run:

```powershell
$env:PYTHONPATH = "src"
python src/reproduce_submission.py
```

This fits one final full-Madrid forest, runs the fixed Madrid CV, evaluates the exact saved Amsterdam splits, writes all trial evidence, creates the figures, and runs unlabelled and deterministic smoke tests. It does not search parameters or choose methods from Amsterdam query results.

## Reproducibility notes and limitations

- Seeds are 42 for the forest, Madrid split generator, and Amsterdam saved splits.
- Class order is `[1, 2, 3, 4]`; features are float32 at forest fit and inference.
- Score-product is selected for `n <= 50`; modal-leaf agreement for `n > 50`.
- The fresh results do not exactly reproduce the teammate's historical values because the teammate's serialized forest and `features_cache.pkl` are absent. The final run uses the verified float32 `features_60.npz`; no parameter was changed to close the gap.
- Madrid random CV is organiser-comparable but spatially interleaved. The historical spatial result is only a development stress test and is not presented as an exact leakage estimate.
- Target-data trials reuse many query pixels, and development results have influenced prior method selection. Only organiser held-out scoring can provide an official result.
- Raw feature extraction assumes each submitted pixel includes its complete available time history. The organiser's exact external inference schema was not specified.
