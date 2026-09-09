# Final critical audit of the UCT building age project

Audit date: 8 September 2026. Reviewed repository HEAD: `d275929`, including the teammate's `4c1d1a3` transfer pipeline. This audit supersedes the submission recommendation in `FINAL_FINDINGS.md`; historical experiment outputs are preserved.

The all-60 Amsterdam logistic regression is a verified numerical benchmark, but the verdict is **C: should not be submitted as the primary transfer method**. It does not use Madrid-supervised parameters. The existing two-tier Random Forest transfer is the provisional primary method because both of its branches substantively use the Madrid-trained forest. Its implementation and locally missing artifacts must pass a reproducibility and inference gate before it can be called final.

**Keep broad modelling frozen. No additional model family is recommended.** The newly pulled code already implements the RF transfer direction that would otherwise justify one final experiment. The necessary next action is one bounded reproduction and comparison of that existing pipeline, followed by packaging. This audit did not train a new candidate or build the PowerPoint.

## Evidence and verification boundaries

Read all seven supplied files in `Docs`, the specification, experiment log, findings, requirements, README, all nine existing Python source files, result JSONs, console logs, and the archive inventory. Inspected saved comparison, shift, confusion and embedding figures. `experiments` contains only `.gitkeep`; no separate war book, leaf-transfer experiment or hybrid-transfer experiment is present. No applicable `AGENTS.md` was found in the repository.

Primary authorities are `Docs/1-Introduction.ipynb` section 1.5 for deliverables and the evaluation rubric DOCX, especially Detailed Dimension Breakdown and Penalties. Notebook 3 supplies executable feature/label conventions; Notebook 4 supplies the baseline implementation. The metric-learning note is advisory, and several of its claims are hypotheses rather than established science.

The audit script `src/final_audit_verify.py` checks saved arithmetic and recomputes the existing raw prototype/logistic evaluations from cached features. `outputs/final_audit_verification.json` records data/source hashes, 184 checked summary vectors, 50 checked support/query sets, exact reproduction of all 100 raw-method trial scores, shift recomputation and aggregate class metrics. All these checks passed. The data SHA-256 hashes match the experiment log. This is not a fresh raw-parquet feature rebuild or a forest/neural-network retraining. Saved forest zero-shot F1 cannot be independently reconstructed from predictions because neither those predictions nor the forest are present. LDA/Bayes-only JSON files omit individual scores; their SDs cannot be independently recalculated from those files.

The complete five-budget tables, including hard-negative logistic and every ablation, are in [VERIFIED_RESULT_TABLES.md](VERIFIED_RESULT_TABLES.md). Repeated copies in consolidated JSONs are not independent replications. `consolidated_results.json` also hard-codes the Madrid numbers; the original baseline result is their evidence source.

## Official requirements and submission checklist

| Requirement | Safest exact interpretation | Current status |
|---|---|---|
| Madrid Stage 1 | Train a transferable age-class prediction model entirely on supplied Madrid data; deliver the final full-Madrid fit | Implemented RF training; no saved Stage 1 model in this checkout |
| Madrid CV | Report macro-F1 mean, SD and confusion proportions from held-out folds | Random 5-fold × 5-repeat results present; pulled five spatial folds present with weaker provenance |
| Amsterdam adaptation | Take the Madrid model and adapt its learned representation/predictions using the allowed labelled support sample | Raw logistic fails the substantive-transfer test; two-tier RF uses the learned forest |
| Budgets | 5, 25, 50, 100 and 200 labelled pixels per class; totals 20, 100, 200, 400 and 800 | All evaluated internally |
| Amsterdam CV | Mean and SD at each budget; Notebook 4 operationalises this as ten repeated support/complement-query draws | Ten saved non-spatial development trials available; describe as repeated resampling, not conventional k-fold CV |
| Macro F1 | Compute each of four class F1 scores, then their unweighted mean; explicitly keep labels 1–4 | Shared evaluator does this |
| Held-out evaluation | Organisers score unseen Madrid and Amsterdam test sets outside the supplied files | No official test results available; prediction interface still needed |
| Trained model submission | Include a usable Madrid Stage 1 artifact and required preprocessing metadata | Missing locally; `.gitignore` excludes pickle/joblib/NPZ artifacts |
| Runnable code | Load Stage 1, accept external support and unlabelled query data, vary `n`, return predictions; include training/rebuild route | Present scripts are experimental runners; external-data adaptation/inference is incomplete |
| Random seeds | Fixed seed must permit an end-to-end rerun; no organiser-mandated numerical seed or cross-version bitwise guarantee | Seed 42 convention; split caches and stage-specific seeds need explicit metadata |
| PowerPoint | First-slide abstract ≤150 words; design decisions, successes/failures; Madrid plus all five Amsterdam mean/SD values; F1 error bars versus log2(n) | Not created; intentionally deferred |
| Written justification | Model/parameter rationale, transfer strategy and F1 interpretation; ≤300 words satisfies both official caps | Not present |
| Originality | Explain an actual adaptation choice or defensible scientific insight; metric learning is optional | Diagnostic investigation offers moderate originality; most components are supplied or standard |
| Low-data learning | Explain 25/class, source contribution and sensitivity to support selection | Important transfer-performance tradeoff unresolved by reproduction |
| Team explanation | All four members contribute and speak; schedule says 10-minute talk + 5-minute Q&A, Thursday 10 September | Handoff provided; no rehearsal evidence |

No supplied source specifies a prediction CSV schema, serialization format, portal, submission deadline, runtime/hardware ceiling or exact held-out input schema. Resolve these logistical items with organisers while preparing a portable local package; do not invent requirements.

### Contradictions and cautions in the supplied material

1. **Transfer direction and classes:** the advisory metric note reverses the cities, suggests five different age bins and mentions MAE. Follow Notebook 1/3: Madrid → Amsterdam, four classes, macro F1.
2. **Five versus four scores:** Notebook 1 explicitly requires five budgets and averages five models, but its deliverables and the rubric say four scores. Include all five; Notebook 4's extra 10/class is optional and must not alter the primary comparison splits.
3. **Word caps:** rubric says 500, Notebook 1 says 300. Use at most 300, not the earlier specification's potentially misleading statement that 500 alone is the formal safe cap.
4. **Age target:** introductory prose says majority building-age class; executable labels bin the area-weighted mean construction year. These are not mathematically equivalent for mixed-age pixels. Follow supplied label construction, describe it accurately, and seek clarification before changing it.
5. **Boundary inclusivity:** supplied `pd.cut` is right-closed. Madrid C1 ≤1960; Amsterdam C1 ≤1945; C2 event < year ≤1984; C3 1984 < year ≤2004; C4 >2004. Do not silently change to left-closed bins to match “pre-” labels.
6. **Years:** data contain 1984–2025, 42 annual positions, despite prose referring to 1984–2024. Preserve the current convention pending the official test specification; imagery after construction is not automatically leakage in this retrospective task.
7. **Baseline transfer:** Notebook 4 trains an RF, but its raw prototype step never uses that forest. Calling this a learned metric does not make it one. The explicit transfer requirement and rubric penalty take precedence over that example's terminology.
8. **CV uncertainty:** Notebook 4 incorrectly says fresh repeat splits make the scores independent. Repeats reuse the same observations; SD is descriptive dispersion. Its prose says defaults 5×1 whereas the actual configured run is 5×5.
9. **Class weighting:** Notebook 4 prose says imbalance was not addressed, while executable RF parameters say `class_weight='balanced'`. Notebook 2 says Class 1 dominates, while supplied data show Class 2 is largest in both cities. Trust the code/counts.
10. **Prize descriptions:** introduction lists four independent prizes; rubric also mentions Best Overall. Prepare for the three weighted judging dimensions and four stated metric/originality prizes; do not assume they are the same ranking.

## Verified central results

All ± values below are population SD across the stated folds/trials. They are not standard errors or 95% confidence intervals.

| Internal evaluation | Macro F1 | Evidence limitation |
|---|---:|---|
| Local Madrid balanced RF, random 5×5 CV | 0.6281 ± 0.0043 | Saved fold arithmetic verified; scaler fitted before CV; random pixels |
| Local RF Madrid → Amsterdam zero-shot | 0.4433 | Saved scalar; source forest/predictions absent |
| Organiser notebook stored Madrid CV | 0.6179 ± 0.0043 | Different execution, unresolved discrepancy |
| Organiser notebook stored zero-shot | 0.3427 | Do not substitute for the local 0.4433 |
| Pulled RF spatial CV, five 17-pixel-block folds | 0.5884 ± 0.0158 | Fold arithmetic verified, model/environment/cache not reproduced |
| Pulled RF after uniform-prior score adjustment, zero-shot | 0.4732 | Uncalibrated heuristic, not a verified improvement over the local forest |

| Method | 5/class | 25/class | 50/class | 100/class | 200/class |
|---|---:|---:|---:|---:|---:|
| Fixed-split raw prototype | .5437 ± .0324 | .5960 ± .0222 | .6052 ± .0133 | .6111 ± .0103 | .6152 ± .0050 |
| Fixed-split raw logistic | .5457 ± .0446 | .6187 ± .0227 | .6467 ± .0092 | .6679 ± .0095 | .6867 ± .0056 |
| Madrid triplet → prototype | .5167 ± .0585 | .5804 ± .0253 | .5826 ± .0215 | .5873 ± .0129 | .5911 ± .0037 |
| Madrid triplet → logistic | .4618 ± .0572 | .5584 ± .0393 | .5788 ± .0139 | .5974 ± .0086 | .6074 ± .0042 |
| Pulled two-tier RF transfer | .5592 ± .0316 | .5976 ± .0235 | .6086 ± .0095 | .6200 ± .0064 | .6215 ± .0047 |

Raw logistic's mean across five budget means is **0.6331**, peak mean **0.6867**, and 25/class mean **0.6187**. The pulled transfer pipeline's corresponding numbers are **0.6014**, **0.6215**, and **0.5976**. These are internal summaries, not official prize scores. Do not take the best trial as “peak F1,” or average SDs to manufacture uncertainty on an average.

The six-budget organiser-style reproduction has prototype means .5437, .5986, .6079, .6129, .6150 at the five required budgets. Its 10/class condition consumes RNG draws before 25/class. This explains the changed prototype results; it is not a model improvement. The fixed-split logistic advantage at 25/class is **0.02263**, with SD of paired differences **0.02005**, winning 9/10 trials. At 5/class it is only **0.00198**, winning 4/10; no convincing 5-shot advantage is established. At 50/100/200 it wins 10/10 development draws. These counts are descriptive, not independent significance tests.

### Feature ablation and shift

| Subset | Feature count | Mean absolute SMD | 25/class prototype | 25/class logistic |
|---|---:|---:|---:|---:|
| Absolute band means | 6 | 1.8487 | .4022 ± .0301 | .4225 ± .0305 |
| Spectral indices | 10 | .5925 | .4189 ± .0298 | .4713 ± .0219 |
| All temporal SD features | 29 | .6740 | .5745 ± .0226 | .6047 ± .0258 |
| Early-period statistics | 12 | 1.3658 | .5090 ± .0199 | .5252 ± .0179 |
| Late-period statistics | 12 | .9773 | .4603 ± .0265 | .4864 ± .0241 |
| Annual change statistics | 12 | .8494 | .4580 ± .0190 | .4618 ± .0145 |
| Presence indicators | 2 | .0270 | Not tested alone | Not tested alone |

The old prose rounds spectral-index SMD to .593; the actual value is .5924597, which rounds to .592 at three decimals. Other headline rounding checks passed.

SMD is `abs(mean_M - mean_A) / sqrt((var_M + var_A)/2)`, then averaged over the subset. It uses an equal-city average of population variances, not sample-size-weighted pooling. It measures marginal mean separation only: it cannot see covariance changes or identify class-conditional shift. Different class proportions can contribute to it. Groups overlap: variability includes index, early/late and annual-difference SDs. The JSON label “atomic_feature_groups” is inaccurate. Six overlapping groups of unequal size are not six independent experiments establishing causation.

Pearson shift–25-shot correlations are −.364 for prototypes and −.476 for logistic; Spearman −.200 and −.314. These describe this small set only. Lower shift is not sufficient for useful classification. All-60 logistic beats every isolated group at 25–200/class; variability-only is slightly higher at 5/class (.5476 versus .5457). The experiments did not remove each group from the full model, so they do **not** prove that every group, every high-shift feature, or either nearly constant flag adds incremental value. The claim “the full features always outperform every group” is false at 5/class.

### Class 2 and error bars

For all-60 logistic at 25/class, pooled query confusion counts give C1/C2/C3/C4 F1 **.6682/.5564/.6501/.6002**. C2 recall is **.5166**; roughly 26.0% of actual C2 pixels go to C1 and 15.2% to C3. The default embedding-prototype confusion counts independently give class F1 **.6768/.4899/.5928/.5627**. C2 is the weakest class in these two 25-shot summaries, not a universal property of every model or every budget. Pooled macro F1 for raw logistic is about .6187 but is not exactly the mean of per-trial macro F1. Counts represent repeated appearances, not ten times as many unique pixels.

Budgets have different, non-nested random support sets; queries are all remaining pixels and also change with budget. The growing support removes equal counts from unequal class populations, so the query mix changes slightly. All ten trials reuse most query pixels. Neural results share one Madrid-trained network per configuration, so error bars omit source-training seed variability. Nothing here establishes third-city uncertainty, spatial independence or official-test confidence intervals.

## Audit of the experimental decisions

### 1 Organiser baseline reproduction

Purpose/hypothesis: establish a trustworthy starting point and test whether the local pipeline reproduces the supplied implementation. The feature/prototype path was reproduced closely, including pixel aggregation and seeded draws. Raw arrays use labels to identify/group labelled rows, but labels are not among the 60 model features; per-pixel targets are constant in the supplied data. The individual band picker can select different slots if a band is missing, so “first complete six-band observation” is stronger than the implementation guarantees.

Established: feature sizes, cleaning counts and prototype results match the supplied reference; local RF results differ. Not established: exact equality of all engineered floating-point values to the organiser's original cache, or the cause of forest disagreement. Package-version sensitivity is plausible, but the log's assertion “therefore environment sensitivity, not random-seed drift” is unjustified without the original environment/model. Stored execution counts do not prove that no code/settings changed. Baseline reproduction was the right next step, but its numerical discrepancy remains open.

### 2 Baseline audit and leakage findings

Purpose/hypothesis: determine whether apparent performance reflects repeated-pixel leakage or optimistic validation. The audit found no duplicate pixel-year rows or varying per-pixel labels. Pixel aggregation prevents the obvious same-pixel-across-years split error. The local Madrid scaler was fitted on all Madrid before CV: a protocol violation, although RF split decisions are largely invariant to positive affine scaling, so a material inflation cannot be inferred from that fact alone.

55,796/69,725 adjacent Madrid pixel pairs cross folds in the diagnostic random split (80.0%). The Amsterdam number 79.7% is from a separate five-fold diagnostic, **not** the actual few-shot support/query trial. These counts establish spatial interleaving, not a quantified F1 bias. Block CV is useful, but adjacent blocks can still touch without a buffer. Neither random nor block CV should be called universally “leakage-safe.” Auditing before model changes was sound; treating all ensuing scores as independent tests was not.

### 3 Amsterdam logistic baseline

Purpose/hypothesis: assess whether learning class-specific linear boundaries from support labels is better than equal-weight nearest-centroid distances. With C=1, 2000 maximum iterations and the same source scaler, the comparison is fair at the trial-fitting level. Reproduction now confirms the numbers exactly. It establishes a strong target-supervised benchmark at 25–200/class; it does not establish transfer from Madrid labels or the optimality of logistic regression. At 5/class the apparent mean gain is weak.

Next decision: use it as a control for Madrid-supervised representations, not automatically as the submission. Model selection uses Amsterdam query scores repeatedly, so “query labels used only for scoring” must not be interpreted as “query labels never influenced decisions.” A fresh reshuffle of these already inspected pixels would not restore an untouched test set.

### 4 Madrid triplet metric learning

Purpose/hypothesis: learn an age-similarity representation on Madrid that makes Amsterdam support classes easier to identify. A 60→64→32→16 ReLU network with unit-length output, class-balanced random triplets, margin .2 and fixed NumPy Adam training was a legitimate Madrid-only experiment. The source network never used Amsterdam labels during its fit. Its loss fell from .1969 to .1323; all Amsterdam budget means lost to their raw counterparts.

Established: this particular training configuration and its two fixed adaptation classifiers underperformed. Not established: that metric learning is intrinsically unsuitable, that class-boundary shift caused the loss, that compression is the cause, or that the network learned a city-invariant metric. There is no source embedding CV, source validation loss, random/untrained embedding control, compression-only control, unnormalised embedding control, or multiple source seeds. Identical C values in raw and unit-normalised spaces do not imply equivalent regularisation, so the logistic comparison does not isolate representation quality perfectly; the prototype comparison helps.

Code-level nuance: backprop divides by the concatenated 3B embedding rows while logged triplet loss averages B triplets, so gradients are scaled by 1/3 relative to that logged objective. This uniform factor is largely absorbed by Adam but is still an implementation discrepancy; it is not evidence of the cause of failure. No embedding weights are serialized. The mixed-city PCA was fit only for visualisation, not prediction. Two projected dimensions cannot diagnose all 16 dimensions or show why a method fails.

Decision: one small mining adjustment was a reasonable bounded check. Broad architecture search was not justified by the evidence or time budget.

### 5 Hard-negative attempt

Purpose/hypothesis: harder Madrid negatives might create more useful separation than random negatives. Selecting the nearest of eight sampled different-class negatives used source data only and otherwise retained the setup. Prototype means became .4003/.4731/.4804/.4800/.4914 and final loss .200065, close to the margin. The configuration clearly lost and was reasonably rejected.

Not established: literal embedding collapse. A loss near the margin and the much smaller coordinate spread in the saved hard-negative PCA support a near-collapse interpretation, but no full embedding variance, distance distribution, active-triplet fraction or source validation diagnostic was saved. Say “training stalled near the margin, the PCA contracted and transfer degraded,” not “we proved hard negatives caused collapse.” One source seed and altered random draws limit attribution to mining alone.

### 6 Feature-group ablations

Purpose/hypothesis: temporal/change summaries may provide a more portable signal than absolute reflectance. Reusing fixed splits and fixed classifiers makes the within-run comparisons controlled. Variability is the strongest reduced subset; pure annual-change features are much weaker. This rejects the operational claim “change-only is the best tested representation,” not all temporal representations or a physical aging hypothesis.

Limitations: target-only classifiers assess Amsterdam utility under source scaling, not supervised knowledge transfer; group size, overlap, regularisation and information loss are confounded. No full-minus-group or reduced-feature embedding experiment was run. Rejecting every possible metric retry because reduced raw logistic lost is not a logical proof; freezing was defensible as a resource decision. The source signal's usefulness cannot be inferred solely from these target-only ablations.

### 7 Madrid–Amsterdam shift analysis

Purpose/hypothesis: larger feature mean shift might predict weaker target utility. All supplied city features were inspected, including those later acting as query rows; class-coloured PCA also inspected query labels for interpretation. This is permissible exploratory analysis of supplied development data, but it is information that would not exist for an unseen city/test set without permission. The deployed transform must not require these full-target statistics.

Established: large marginal shifts, especially absolute means, with only a modest descriptive association to F1. Not established: causal mechanism, class-conditional mismatch, reliable shift-based feature ranking or a rule generalising to other cities. No target statistics were fed into the evaluated predictive scaler; nevertheless this analysis informed subsequent design. Selecting a support-only correction was reasonable; reporting a pristine unseen-target experiment would not be.

### 8 Support-only standardisation

Purpose/hypothesis: replacing source feature scales with those estimated on target support might improve adaptation. Fit statistics used only each trial's support; this is valid inductive preprocessing. Logistic means became .5374/.6149/.6435/.6660/.6861, lower at every tested budget; the 25/class difference is −.0038.

Crucial interpretation: applying the same translation to support and query does not change prototype distances, and an intercept makes logistic boundaries translation-invariant in the ideal fitted solution. Rescaling changes distance weights and logistic regularisation. The experiment is therefore primarily a **support-based rescaling test**, not a general test of domain alignment. For nonconstant features, source scaling algebraically cancels under a second support standardisation. It does not show that all alignment fails or that the observed marginal mean gap causes errors. Reject this configuration; do not claim statistical proof of harm or a diagnosed class-conditional cause.

### 9 Freezing broad modelling

Purpose: stop low-yield tuning and preserve time for a usable submission and explanation. Supported by repeated loss of the bounded alternatives and strong raw benchmark. Not supported: calling scaler-only logistic the “final numerical choice for submission,” claiming no Madrid information can help, or declaring technical readiness without model serialization/inference checks. The freeze decision is retained; the submission selection is revised.

## Audit of the teammate's pulled work

The new files materially change the compliance options. Do not ignore them, and do not accept their docstrings as experimental evidence.

**LDA:** a supervised Madrid projection to three discriminant dimensions is genuine source learning. Saved prototype means are .5154/.5881/.5973/.6087/.6081. It loses to our raw prototype means at every budget, assuming matching pixels. The .5383 “Madrid self” score uses the first 5,000 training rows and full-training centroids; it is neither CV nor an upper bound on Amsterdam performance. The “too much compression” explanation remains untested. `BASELINE_RAW` contains values inconsistent with the verified raw baseline, including .6200/.6215 that coincide with the RF leaf results.

**RF/prototype score product:** the classifier multiplies adjusted RF class scores by softmax negative prototype distances. This substantively uses Madrid labels through the forest. It is a defensible heuristic combination, but not an established Bayesian posterior: the RF is already conditional on the query features, the softmax is not a validated class likelihood, and both derive from the same features. Normalisation alone proves neither calibration nor conditional independence. The formula has fixed temperature and no explicit n-dependent relative weight, so a claimed automatic switch to target evidence as n grows is not established.

Its source-prior adjustment assumes a uniform target prior and divides by raw Madrid prevalence although the forest was trained with balanced class weights. The source probabilities need not represent the raw prevalence posterior that this correction assumes. Uniform is a modelling choice motivated by equal-class scoring; it is not the measured Amsterdam prior. Do not call this a theoretically justified label-shift correction without further assumptions. It may still be empirically useful.

**Leaf branch:** for each tree and support class, take the most common leaf ID. Predict by the fraction of trees in which a query matches that class's modal leaf. This genuinely reuses supervised Madrid partitions. It is **modal-leaf agreement**, not the usual average pairwise RF proximity to all support pixels or a learned continuous kernel embedding. Ties choose the smallest leaf ID in each mode and the first class in a class-score tie; this should be documented. The branch can discard multimodal class structure.

**Two-tier choice:** n≤50 uses the product; n>50 uses modal leaves. The code comment says the reverse boundary ambiguously; actual code is authoritative. The saved full sweep has arithmetic-consistent vectors. It does not beat the stored organiser baseline at 25/class (.5976 < .5986), despite the commit title/docstring. It is only .00159 above the fixed-split raw prototype mean at 25, far below raw logistic's .6187. Its 5-shot mean is above raw logistic's, so the old global statement “logistic is strongest among all tested methods except variability at 5” is now stale.

**Provenance:** the pulled scripts reconstruct the same five-budget RNG index sets as our saved splits if input labels and row ordering are identical; this index-set property was verified, including the harmless support sorting difference. They do not load the saved split file, and their `features_cache.pkl` is missing, so pixel-identity equivalence for the historical run is not verified. Float64 raw features there versus float32 before scaling in `features_60.npz` also matter for forest reproducibility. The output path names another machine. No package versions, data/cache hashes, source-model hash or complete intermediate predictions were saved. Missing sibling scripts are referenced only in comments, so `final_pipeline.py` can still rebuild using `organiser_baseline.py`; standalone Bayes/LDA scripts fail without their required cache.

**Madrid spatial claim:** .5884 is saved, but the referenced controlled `.588 versus .618` spatial-leakage comparison and war book are absent. It is invalid to subtract local .6281 from teammate .5884 and attribute the entire gap to spatial splitting. Even 510m blocks are not a guarantee of independence; no buffer was imposed. Treat it as a reported stress-test result pending reproduction.

**Submission functionality:** the main runner always loads labelled Madrid and Amsterdam and retrains CV plus a forest even for `--n 25`; it has no load-model/external-query CLI. The raw preprocessor requires `weighted_mean_year` and filters/group-bys `age_class`, so it cannot process a genuinely unlabelled test file as written. Serialized state omits `madrid_prior`, required by the product branch. No Stage 1 artifact is present. This is not yet an end-to-end held-out submission package.

## Final method and experiment verdict

**All-60 logistic: C.** Its only Madrid-derived fitted state is an unsupervised scaler. All class coefficients/intercepts are learned anew from Amsterdam. Appending a separately trained Madrid RF to a ZIP does not turn that logistic procedure into adaptation of the RF. The organiser example provides an argument for lenient interpretation, but the explicit Stage 1→Stage 2 wording and up-to-20-point/disqualification rule make it unsafe to rely on that argument. It could only become a primary option after explicit organiser acceptance of this exact procedure; no such acceptance is in the repository.

**Provisional primary: the existing two-tier RF transfer, after reproduction and packaging.** It is structurally a genuine transfer procedure and reportedly stronger than the triplet alternatives, while retaining useful 5-shot performance. Select it for its actual source dependence, not for an unsupported Bayesian claim. It remains at verdict B for overall submission readiness until provenance and inference gates pass; its transfer mechanism itself is substantially safer than scaler-only logistic. No deployable final method is approved by this audit yet.

**One final new model experiment is not necessary.** RF leaf assignments were a sensible possible compliance repair before these pulled files were considered; that direction already exists. Another architecture, anchoring scheme, kernel encoding, ensemble-weight search or fine-tuning sweep would now primarily seek extra F1 rather than repair an absent transfer mechanism. That does not justify reopening modelling at this handoff stage. This is a resource judgement, not proof that no better method exists.

The one bounded validation required next is precise:

1. Recover the teammate's original cache/model/environment if available; otherwise train exactly the existing 500-tree balanced RF on the verified local features, recording the float precision and source hashes. Do not pretend reconstructed float32-derived features are bit-identical to the missing float64 cache.
2. Freeze both branches and the n=50 threshold before scoring. Evaluate the unchanged product branch at 5/25/50 and unchanged modal-leaf branch at 100/200 on `amsterdam_fixed_splits.npz`, with the existing raw logistic/prototype controls. Save predictions, per-trial scores and confusions plus model hashes. No temperature, threshold or weight search.
3. Use the already implemented five-fold spatial Madrid evaluation with fold-fitted scaling to report the actual submitted source forest configuration, or a documented random-CV reference and spatial stress test. Reproduce once; do not repeat a broad RF sweep. The imported full run took about 514 seconds on the teammate's machine, which is a feasibility clue, not a local runtime promise.
4. Verify reload, external unlabelled queries, support-budget enforcement, class order and repeated-run determinism. These are submission checks, not a new method search.
5. If the RF pathway reproduces and works, freeze it as primary and disclose the raw logistic benchmark's higher internal F1. If reproduction fails materially, diagnose provenance/implementation first; do not silently pick a best seed or submit the old JSON. If its performance is genuinely poor, revisit the final choice explicitly using compliant existing methods rather than resurrecting scaler-only logistic by wording alone.

This single validation has a realistic chance of delivering a runnable, defensible transfer entry cheaply because the algorithm and complete sweep already exist. It does not need to beat raw logistic to fix the rubric weakness. No extra experiment was implemented during this audit.

## Demanding judge assessment

These are subjective estimates from repository evidence, not calibrated winning probabilities. Penalties below are separate from category marks.

| Dimension | Likely now | Strengths | Weaknesses and remaining opportunities |
|---|---:|---|---|
| Challenge understanding and transfer strategy /30 | 17; plausible 14–20 | Correct cities/budgets; pixel-level protocol; useful low-shot controls and diagnostic failures | Conflicting final methods, missing model/inference, overclaims about transfer and alignment. Reproduce genuine RF transfer and state selection/spatial limitations clearly; roughly 22–25 becomes plausible |
| Originality and creativity /40 | 18; plausible 14–22 | Controlled subset comparison linked to shift; willingness to report failed metric learning; combined source/target mechanism | Sixty features, prototype and metric ideas largely supplied; RF leaf idea is explicitly suggested. No established new general method or causal failure diagnosis. Precise insight and a source-contribution comparison can improve this toward 22–27; more experiments alone will not |
| Presentation and collaboration /30 | Not assessable from a delivered talk; readiness roughly 3–8 | Handoff and supporting figures now exist | No deck, abstract, justification or rehearsal. All speakers must explain limitations; a prepared team could earn 22–26 |

The first two categories therefore suggest around **35/70 before penalties**, not an excellent finished submission. Using scaler-only logistic as primary risks up to 20 points or disqualification for being off-topic. Unsupported “leakage-free” or Bayesian claims create additional scientific-rigor exposure. Exceeding caps or omitting one of the three pillars can cost up to five points. Do not add these deductions mechanically as independent guaranteed penalties.

| Prize | Assessment from our own evidence |
|---|---|
| Best average F1 | Raw logistic .6331 is a useful local benchmark, but currently unsuitable as primary. RF .6014 is only modestly above raw prototype's trend and has incomplete provenance. No evidence supports claiming front-runner status |
| Best peak F1 | Raw logistic .6867 at 200/class is our strongest verified mean at a budget. Genuine RF candidate reportedly peaks at .6215. Limited evidence of exceptional source-transfer performance |
| Best 25/class | Raw logistic .6187 versus raw prototype .5960 is a real development improvement; compliant RF .5976 is nearly flat versus that control. This is a substantive competitive weakness |
| Best originality | The diagnostic finding is more promising than claiming a new architecture. Moderate prospects for a thoughtful explanation of failure; neither a standard triplet attempt nor copied RF-leaf advice is novel by itself |

## Strongest truthful scientific story

We are trying to use a data-rich city's satellite histories to classify age in a city with very few local labels. Each observation is a mixed 30m pixel rather than a clean individual building. The cities differ in spectral distributions, class frequencies and even the historical boundary between their first two classes. A Madrid decision rule therefore need not apply directly in Amsterdam.

We reproduced the organiser's time-series summaries, RF source model and target prototypes, then audited what those stages actually transfer. The first useful classifier change was fitting flexible linear boundaries on Amsterdam support: it improved 25–200-shot development scores. We investigated triplet learning because Madrid labels might teach a reusable similarity. This fixed embedding lost, and harder negatives made it worse. The lesson is limited but useful: source-supervised separation is not itself evidence of portability.

We then tested whether simpler, less shifted temporal subsets would be better. Annual change alone did not win; low-shift indices were weak; retaining the complete feature vector worked best from 25/class upward. The surprising finding is **that low marginal city shift was not sufficient to identify the useful features, while source-label training did not automatically improve our best target classifier**. This is supported by controlled comparisons, not a general impossibility theorem about transfer.

The teammate's RF method supplies an actual transfer pathway, with a reported low-shot benefit and weaker larger-budget F1 than the target-only benchmark. Our final task is to validate that tradeoff and deliver it honestly. We cannot claim successful metric learning, proven city invariance, a Bayesian posterior, or generalisation to a third city. The final source-transfer performance must be tied to the artifact actually submitted.

## Exact presentation claims and figure guidance

Allowed, with the named conditions: “On ten fixed internal Amsterdam trials, all-60 logistic scored .6187 ± .0227 at 25 labels per class”; “our tested Madrid triplet embeddings underperformed the raw controls”; “annual-change-only features did not win”; “mean shift alone did not reliably rank the tested subsets”; “Class 2 was weakest for the 25-shot raw logistic benchmark”; “the RF transfer procedure uses source-trained tree structure/predictions.” Label imported scores as reported until reproduced.

Avoid: “logistic is our winning transfer model”; “Madrid supervision never helps”; “we discovered why metric learning fails”; “the network definitely collapsed”; “all domain alignment fails”; “every high-shift feature is necessary”; “all 60 always win”; “the two-tier pipeline beats baseline at all budgets”; “repeated CV gives independent trials”; “.6867 is official test performance”; “the spatial gap is exactly .030/.040”; “Bayesian optimal/calibrated posterior”; “the Madrid self score is an upper bound”; “we invented the supplied feature engineering.”

`final_comparison_f1_vs_budget.png` is readable and uses log2 correctly, but it omits the pulled RF method and needs a development/SD caption and benchmark naming. `strongest_25_confusion_matrix.png` is readable but must be titled with all-60 logistic and “pooled over ten query sets”; it is not the provisional primary model's confusion. `feature_group_shift_vs_transfer.png` and `feature_group_domain_shift.png` need overlap and development-data caveats; “transfer” there means target performance with source scaling. Generic metric budget plots use a log axis without explicitly setting base 2; use the final log2 plotting route for submission. Embedding PCA figures are illustrations only, with no variance-explained or causal evidence. Do not display obsolete generic/random/hard-negative files interchangeably: runner outputs reuse generic filenames. No saved figure establishes held-out official performance.

## Remaining work before presentation creation

| Priority | Task and acceptance criterion | Suggested owner |
|---|---|---|
| 1 | Resolve the primary method using the single RF reproduction gate above; record exact source-model/configuration and all five budget results | Ubayd + Alex |
| 2 | Separate label derivation from spectral feature extraction. Query feature code must accept data without age/year labels, group only by pixel, preserve identities and use a fixed temporal schema | Alex |
| 3 | Create train/save/load/adapt/predict entry points. Persist RF, scaler, feature order, class order, source priors, year/cleaning rules, adaptation settings and package versions | Alex + Samson |
| 4 | Validate support has exactly n examples of each class; fail on shortages rather than `min(n, count)` silently changing the task. Validate no overlap/malformed features, and no query labels are needed | Samson |
| 5 | Cache validation by source hash, feature schema and pixel-key order; split manifest keyed to pixels. Explain that `--n 25` and `--sweep` currently draw different 25-shot sets because their RNG histories differ. Avoid silent cache overwrites on changed seeds | Samson |
| 6 | Clean-machine or isolated-environment smoke run from saved Stage 1 through unlabelled predictions, followed by repeat-run equality checks on identical inputs. Pin resolved dependencies and document Python version | Alex + Samson |
| 7 | Save source CV fold scores/confusion counts and per-trial target predictions/metrics; rebuild final figures from those outputs with SD labels and log2 axis | Ubayd |
| 8 | Replace the two-line README with commands, input/output contracts, data location, runtime, model artifact location and reproducibility notes. Make a fresh ZIP with code/model/metadata/docs, no virtual environment or stale results | Samson |
| 9 | Resolve organiser logistics and any boundary/word-count ambiguity; freeze a claims sheet and ≤300-word justification outline, then create ≤150-word abstract and 10-minute slide plan | Israel + Ubayd |
| 10 | Use TEAM_HANDOFF.md to confirm everyone can explain source dependence, F1, failure evidence and limitations before building/rehearsing the deck | All four |

The existing ZIP contains an old baseline and `.venv` but no current transfer source, official Docs, raw Data or trained model; it is not a submission. Build a fresh explicit package. A raw-data-independent Stage 1 load path should make the evaluator's adaptation run quick; do not require all source CV fits just to predict a query set.

Estimated full submission completion after this audit: **about 60%**, an effort-based estimate rather than a count of files. Research/audit is substantially complete; the highest-risk missing work is deployable reproducibility and the primary-model evidence. Roughly 20 percentage points remain in validation/packaging and 20 in deck/abstract/justification/rehearsal. The next stage should be that bounded validation and packaging, then presentation creation.
