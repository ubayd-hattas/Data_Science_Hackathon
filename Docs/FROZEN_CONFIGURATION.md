# Frozen final configuration

This file records the modelling choices frozen before any antes before the before the final reproduction. The executable authority is `src/final_pipeline.py`; no result-dependent parameter changes were made.

| Component | Frozen value |
|---|---|
| Transfer direction | Madrid to Amsterdam |
| Stage 1 model | `RandomForestClassifier` |
| Trees | 500 |
| Class weighting | `balanced` |
| Forest seed | 42 |
| Parallelism | `n_jobs=-1` |
| Other RF parameters | scikit-learn defaults in version 1.9.0 |
| Feature count | | 60 |
| Training dtype | float32 |
| Class order | 1, 2, 3, 4 |
| Feature order | Exact `feature_names` list in the saved model metadata |
| Scaling | Madrid `StandardScaler`; no Amsterdam refit |
| Madrid prior | Empirical full-Madrid class frequencies saved in the package |
| Product branch | Adjusted RF scores multiplied by prototype softmax scores |
| Prototype distance | Euclidean on Madrid-scaled features |
| Temperature | 1.0 |
| Product epsilon | 1e-6 after multiplication, then renormalise |
| Leaf branch | Agreement with the per-class modal leaf in each of 500 trees |
| Branch threshold | Product for n less than or equal to 50; modal leaf for n greater than 50 |
| Evaluation seed | 42 |
| Required budgets | 5, 25, 50, 100, 200 labels per class |

## Feature order

The order is: six overall band means; six overall band SDs; five index means; five index SDs; six early means; six early SDs; six late means; six late SDs; six year-on-year difference means; six year-on-year difference SDs; and two coverage indicators. The complete ordered names are embedded in `artifacts/stage1_madrid_rf.pkl` and `outputs/final_submission/final_results.json`.

## Preprocessing

For each band independently, use the first QA-valid observation among slots 1, 2, and 3. Drop a pixel-year row if any selected band is missing. Remove rows with Blue greater than 15,000. Record whether each pixel had observations at or before 2003 and at or after 2004. Create a complete annual grid from the minimum through maximum year present in the supplied dataset, interpolate internal gaps linearly, then backward-fill and forward-fill edge gaps. Compute NDVI, NDBI, UI, MNDWI, and BSI with epsilon 1e-6. Aggregate with pandas sample SD (`ddof=1`).

## Resolved naming ambiguity

Historical code called the low low-budget multiplication Bayesian. The computation is preserved exactly, but the final submission calls it a **score-product heuristic** because calibration and conditional-independence assumptions were not established. Historical comments around the threshold were inconsistent; executable branching was unambiguous: `n > 50` uses modal leaves, otherwise score-product.
