# Final findings

> Historical modelling summary. The [final critical audit](Docs/FINAL_CRITICAL_AUDIT.md) supersedes the submission recommendation below: all-60 logistic is a benchmark, not the primary transfer submission. The pulled two-tier RF pipeline is the provisional primary pending reproduction and packaging. Use the [team handoff](Docs/TEAM_HANDOFF.md) for current explanations and [verified tables](Docs/VERIFIED_RESULT_TABLES.md) for comparisons.

## Challenge in plain English

Use Madrid-labelled 30 m Landsat pixel histories to support building-age classification in Amsterdam, where only 5, 25, 50, 100, or 200 labelled pixels per class are available. There are four city-specific age classes, and the official metric is macro F1. The evaluation must keep Amsterdam support pixels separate from query pixels and must not learn from query labels or unavailable test-set statistics.

## Verified experimental progression

1. Reproduced the organiser's 60-feature preprocessing and Random Forest/prototype baseline.
2. Trained one Madrid-supervised triplet embedding and evaluated both prototype and logistic classifiers on ten saved Amsterdam support/query splits.
3. Ablated six scientific feature groups on those exact splits and measured Madrid-Amsterdam shift with absolute standardized mean difference (SMD).
4. Tested one inductive correction: within each trial, standardise all 60 features using only the labelled Amsterdam support set. No query features or labels were used to fit it.
5. Skipped a second metric-learning run because no reduced representation beat the full representation and the correction did not improve it.

## Organiser baseline

- Madrid balanced Random Forest, 5-fold x 5-repeat CV: **0.6281 +/- 0.0043** macro F1.
- Madrid Random Forest applied directly to Amsterdam: **0.4433** macro F1.
- Organiser-style Amsterdam prototype at 25/class on the later fixed splits: **0.5960 +/- 0.0222**.

## Final chosen method and exact scores

The strongest tested method is logistic regression trained on the labelled Amsterdam support pixels in the complete organiser 60-feature space, with the original Madrid-fitted scaler. This is the numerical choice for submission; importantly, the Madrid contribution is only the scaler, not a beneficial Madrid-trained classifier or embedding.

| Amsterdam labels/class | Macro F1 mean | SD |
|---:|---:|---:|
| 5 | 0.5457 | 0.0446 |
| 25 | **0.6187** | 0.0227 |
| 50 | 0.6467 | 0.0092 |
| 100 | 0.6679 | 0.0095 |
| 200 | 0.6867 | 0.0056 |

The strongest 25/class score is **0.6187 +/- 0.0227**. This method is also strongest overall: it wins at 25, 50, 100, and 200/class and has the best mean trend across the required budgets. Variability-only logistic is marginally higher at 5/class (0.5476 versus 0.5457), well within trial dispersion, but is worse at every larger budget.

## Feature and domain-shift finding

| Feature group | Features | Mean absolute SMD | 25/class prototype | 25/class logistic |
|---|---:|---:|---:|---:|
| Absolute spectral means | 6 | 1.849 | 0.4022 +/- 0.0301 | 0.4225 +/- 0.0305 |
| Spectral indices | 10 | 0.593 | 0.4189 +/- 0.0298 | 0.4713 +/- 0.0219 |
| Temporal variability / SDs | 29 | 0.674 | **0.5745 +/- 0.0226** | **0.6047 +/- 0.0258** |
| Early-period statistics | 12 | 1.366 | 0.5090 +/- 0.0199 | 0.5252 +/- 0.0179 |
| Late-period statistics | 12 | 0.977 | 0.4603 +/- 0.0265 | 0.4864 +/- 0.0241 |
| Year-to-year change | 12 | 0.849 | 0.4580 +/- 0.0190 | 0.4618 +/- 0.0145 |
| Early/late presence flags | 2 | 0.027 | Not tested alone | Not tested alone |

Absolute means shift most and transfer poorly; variability shifts less and is the strongest reduced representation. However, low shift is not sufficient for strong transfer: indices and year-to-year changes both perform poorly. Across the six predictive groups, shift versus 25/class F1 has only a modest negative association (Pearson -0.364 for prototype and -0.476 for logistic; Spearman -0.200 and -0.314). Therefore the evidence does **not** support the broad claim that higher-shift groups reliably transfer worse, and it rejects the specific hypothesis that change-only information transfers best.

The full 60 features outperform every isolated group. Complementary information matters more than removing all high-shift features.

## Alignment and metric-learning outcomes

- Support-fitted target standardisation changed raw logistic from 0.6187 to **0.6149** at 25/class and did not improve any required budget. Explicit alignment did not help.
- The Madrid triplet embedding was worse than raw features at all budgets. At 25/class, embedding prototype scored **0.5804 +/- 0.0253** and embedding logistic **0.5584 +/- 0.0393**.
- Madrid-derived information did not improve on Amsterdam-support logistic regression. The likely failure is objective mismatch: Madrid age separation does not remove city-specific spectral differences, the city class boundaries differ, and 60 features are compressed into a 16-dimensional unit-normalised embedding.
- A metric-learning retry was not performed because the prerequisite evidence was absent.

## Class-level weakness at 25/class

For the chosen logistic method, aggregate per-class F1 across the ten query sets is C1 0.668, **C2 0.556**, C3 0.650, and C4 0.600. Class 2 is weakest, with recall 0.517; 26.0% of actual C2 predictions go to C1 and 15.2% go to C3. This is consistent with overlap between adjacent construction-age intervals.

## Limitations

- Results use ten random, non-spatial Amsterdam support/query trials; nearby pixels may occur on opposite sides of a split.
- Trials reuse pixels in different roles, so their SD is dispersion across fixed trials, not an independent standard error.
- The final evaluation is internal; organiser-held-out Madrid and Amsterdam test distributions are unavailable.
- The supplied city class boundaries differ (Madrid 1960 versus Amsterdam 1945), limiting direct label-semantic transfer.
- The final numerical winner uses only Madrid-fitted scaling as its transfer component. This is a rubric risk if judges require a substantively Madrid-learned representation.

## What was original about the investigation

The investigation held Amsterdam splits constant while linking scientific feature ablation to an interpretable city-shift statistic, then used that evidence to gate one leakage-safe correction and the decision not to retry metric learning. The useful result is diagnostic: domain shift is real and concentrated in absolute spectral levels, but shift magnitude alone does not predict transfer utility.

## Likely technical judge questions

1. **How did you prevent Amsterdam leakage?** Each of ten saved trials has disjoint, class-balanced support indices and query indices. Models and support standardisation use only support rows; query labels are used only for scoring.
2. **Why is macro F1 appropriate?** It gives equal weight to all four age classes despite unequal city class counts, matching the challenge specification.
3. **What is the best 25/class result?** Raw 60-feature support logistic: 0.6187 mean macro F1 with 0.0227 population SD.
4. **Did Madrid supervision improve Amsterdam?** No. The Madrid triplet embedding and its hard-negative variant were worse than the raw representation; only the Madrid-fitted scaler remains in the winning pipeline.
5. **What proves domain shift?** Absolute spectral means differ by 1.849 pooled SD units on average between cities; early statistics differ by 1.366. The Madrid Random Forest also falls to 0.4433 zero-shot Amsterdam F1.
6. **Why did alignment fail?** Support-only normalization safely removes marginal location/scale differences but cannot correct class-conditional mismatch or changed class boundaries; at 25/class it reduced logistic F1 by 0.0038.
7. **Why not retry metric learning on variability features?** Variability-only logistic remained 0.0140 below all-60 logistic at 25/class, so it did not meet the predeclared justification for another retry.
8. **What should happen next?** Freeze broad modelling. Run final reproducibility validation from cached inputs, verify deterministic outputs and packaging, then prepare the presentation and written justification.

## Recommendation

**Freeze modelling now.** No result from this phase indicates a realistic material gain from another broad experiment. Proceed to final reproducibility validation and submission packaging.
