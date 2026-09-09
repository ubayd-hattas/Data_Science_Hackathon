# Challenge specification

This specification follows the evaluation rubric as the judging authority and uses the organiser notebooks as the detailed challenge instructions. The separate metric-learning note is advisory and conflicts with those sources on transfer direction.

## Authoritative task

- **Source city:** Madrid.
- **Target city:** Amsterdam.
- **Task:** classify the majority building-age class of each 30 m Landsat pixel from its spectral time series.
- **Prediction:** one integer class in `{1, 2, 3, 4}` per pixel.
- **Target source field:** `weighted_mean_year`, an area-weighted construction year. It is used to derive labels and must not be a model feature.
- **Transfer requirement:** train the Stage 1 model on Madrid, then adapt the learned representation/model using small labelled Amsterdam samples. The rubric treats a method that does not perform transfer learning as off-topic.
- **Allowed Amsterdam labelled sample sizes:** 5, 25, 50, 100, and 200 pixels **per class**, according to Notebook 1 sections 1.5 and 1.7. Notebook 4 additionally evaluates 10 per class, but 10 is not listed as a required challenge budget.
- **Evaluation metric:** macro-averaged F1. Madrid reporting requires cross-validation mean and standard deviation plus row-normalised confusion-matrix proportions. Amsterdam reporting requires cross-validation/resampling mean and standard deviation at the required label budgets.
- **Official test evaluation:** organisers evaluate held-out Madrid and Amsterdam test sets that are not in the supplied files.
- **Numeric F1 thresholds:** none are specified. Prizes reward (1) mean across the five Amsterdam models, (2) best single Amsterdam F1, and (3) the 25-samples-per-class Amsterdam F1. Originality is judged qualitatively.

## Class definitions

| Class | Madrid | Amsterdam |
|---|---|---|
| 1 | Before the 1960 event | Before the 1945 event |
| 2 | 1960 event to 1984 | 1945 event to 1984 |
| 3 | 1984 to 2004 | 1984 to 2004 |
| 4 | 2004 to 2024 | 2004 to 2024 |

The supplied code uses `pd.cut(..., bins=[-inf, event, 1984, 2004, inf])` with the default right-closed intervals. Therefore exact boundary years are assigned as follows: the city event year is Class 1, 1984 is Class 2, 2004 is Class 3, and years greater than 2004 are Class 4. This does not exactly match labels such as "Pre-1960" and "1960-1984"; organisers should clarify inclusivity before final submission.

## Submission requirements

1. The trained Stage 1 Madrid model.
2. Runnable training/adaptation code, parameterised by Amsterdam samples per class and executable end-to-end with a fixed random seed.
3. A PowerPoint presentation.
4. A written justification.

No prediction-file schema, model serialization format, file naming convention, packaging convention, runtime limit, hardware limit, or submission portal is specified in the supplied repository.

## Presentation requirements

- First slide: abstract of at most 150 words.
- Explain key design decisions and what worked and did not work.
- Show Madrid F1 and Amsterdam label-budget F1 results with error bars.
- Plot F1 with error bars against `log2(sample size)`.
- Interpret performance trends, including low-data behavior.
- All four team members must contribute meaningfully and speak.
- Schedule workbook: 10-minute team presentation plus 5-minute Q&A on Thursday 10 September 2026.

## Written justification

The rubric requires at most **500 words** and requires three pillars: model design and parameter rationale, transfer strategy, and interpretation of F1 trends across sample sizes. Notebook 1 says at most **300 words** for the same deliverable. Use **at most 300 words** to satisfy both; do not assume that the rubric's larger cap overrides the submission instruction without organiser clarification.

## Restrictions and judging priorities

- The workflow must be scientifically sound, leakage-free, reproducible, and genuinely cross-city.
- Data leakage can cost up to 20 points or lead to disqualification.
- A non-transfer method can be treated as off-topic and similarly penalised.
- Metric learning is optional; the notebooks and approach note explicitly allow alternatives.
- The rubric weights challenge understanding/transfer strategy 30 points, originality/creativity 40 points, and presentation/team collaboration 30 points.
- Omitting any written/presentation pillar or exceeding the rubric word cap can cost up to 5 points.
- Unprofessional behavior can cost up to 20 points or cause disqualification.

## Explicit leakage and validity risks

- Split by geographic pixel before any row-level modelling; all years and observation slots from a pixel must stay together.
- Keep Amsterdam support/adaptation pixels disjoint from Amsterdam evaluation pixels within every trial.
- Fit all learned preprocessing, imputers, scalers, selectors, representations, and models on the relevant training/support partition only. Do not fit them on Madrid validation folds or Amsterdam query/test data unless a transductive step is explicitly permitted.
- Never include `weighted_mean_year`, derived `age_class`, or label-dependent aggregates in model inputs.
- Guard against exact pixel duplicates and near-neighbour spatial autocorrelation across folds. Random pixel CV can substantially overstate geographic generalisation even without exact duplicates.
- Do not use future imagery, target statistics, scene-specific cleaning thresholds, or metadata that will be unavailable when organiser test predictions are generated.
- Keep target-city label-budget selection independent from evaluation. Hyperparameter selection on the same Amsterdam query labels would leak evaluation information.

## Ambiguities and contradictions in supplied files

1. **Transfer direction:** `METRIC_LEARNING_APPROACH - Copy.md` describes Amsterdam training followed by Madrid transfer. Notebook 1, Notebook 3, Notebook 4, and the rubric's phrase "adapt learned representations to Amsterdam" specify Madrid source to Amsterdam target. The authoritative direction is **Madrid to Amsterdam**.
2. **Amsterdam result count:** Notebook 1 requires five budgets (5, 25, 50, 100, 200 per class) and defines average F1 across five Amsterdam models, but its presentation deliverable says "four Amsterdam F1 scores." The rubric also refers to "four F1 scores / sample sizes" without listing them. Report all five required budgets pending organiser clarification.
3. **Written word cap:** rubric 500 words; Notebook 1 300 words. The rubric governs, while 300 words satisfies both.
4. **Date range:** notebooks describe 1984-2024 and 41 years, but both parquet files contain 42 distinct years from 1984 through 2025. The supplied preprocessing includes 2025 automatically.
5. **Class-boundary inclusivity:** prose labels say "Pre-1960" and "Pre-1945," while `pd.cut` assigns 1960 and 1945 to Class 1 and assigns 1984 and 2004 to the earlier interval.
6. **Baseline transfer claim:** Notebook 4 calls prototypes in the 60 engineered-feature space "metric learning," but no embedding or metric is learned from Madrid for that step; only the Madrid-fitted scaler is reused. This may not satisfy the rubric's requirement to adapt learned representations.
7. **Schedule date formatting:** the schedule labels the event "Mon 7th Sept" through "Thurs 10th Sep"; in 2026 these dates align with Monday through Thursday. It contains presentation timing but no additional modelling or submission specification.

## Source precedence used in this audit

1. `Docs/Evaluation Rubric Overview - Copy.docx` - judging authority.
2. `Docs/1-Introduction.ipynb` - detailed official task and submission instructions.
3. `Docs/2-Reading_Data.ipynb`, `Docs/3-Preprocessing.ipynb`, and `Docs/4-Modelling.ipynb` - organiser data and baseline implementation.
4. `Docs/Hackathon Final Schedule - Copy.xlsx` - event and presentation timing.
5. `Docs/METRIC_LEARNING_APPROACH - Copy.md` - non-binding advisory note.
