# Team handoff for the UCT building age challenge

Everyone should read this before preparing slides. The current decision is that all-60 Amsterdam logistic is our strongest verified numerical benchmark, but its only Madrid-derived state is a scaler. It is not safe as our primary transfer submission. The existing two-tier Madrid Random Forest procedure is the provisional primary, pending a local reproduction and a working saved-model prediction interface. Do not describe provisional results as final.

The full evidence and allowed claims are in [FINAL_CRITICAL_AUDIT.md](FINAL_CRITICAL_AUDIT.md); all five-budget numbers are in [VERIFIED_RESULT_TABLES.md](VERIFIED_RESULT_TABLES.md).

## Four natural presentation roles

| Member | Talk segment | What to own in Q&A | Handoff sentence |
|---|---|---|---|
| Ubayd | Problem, official task and evaluation; about 2 minutes | Madrid → Amsterdam, source/target, four classes, macro F1, 25/class | “Alex will explain what information goes into one pixel prediction.” |
| Alex | Features, baseline and genuine transfer mechanism; about 2.5 minutes | The 60 features, scaler versus supervised model, RF predictions/leaves, saved-model interface | “Samson will show how we checked whether the methods actually worked.” |
| Samson | Controlled experiments, metric-learning failure and validation; about 2.5 minutes | Fixed splits, leakage, error bars, ablations and shift | “Israel will connect that evidence to our final choice and its limitations.” |
| Israel | Results, honest final-method tradeoff, limitations; about 2 minutes | Which numbers are final, originality, prize tradeoffs, what we cannot claim | “Our strongest finding is that source training and low feature shift did not automatically give better target predictions.” |

Leave roughly one minute total for transitions. The official schedule allows 10 minutes plus 5 minutes Q&A. These are proposed responsibilities, not claims about who historically performed each experiment. Each person must understand all core concepts and be able to cover another speaker's main answer.

## Core concepts everyone must understand

**Landsat pixel and time series.** A pixel covers a 30m by 30m patch. It may contain several buildings, roads, trees and bare ground. Landsat records reflected light in six wavelength bands. We have annual histories from 1984 to 2025 in these files; a model observation is one location's full history, not each year's row treated as a separate building. Up to three observations exist in a year; the supplied code chooses the first QA-valid value per band, removes very high Blue values, and fills temporal gaps within that pixel.

**The four age classes.** We classify a bin of the pixel's area-weighted mean construction year. This is the supplied code's target, which is not always the same as the majority class of mixed buildings. Use the actual boundary rule:

| Class | Madrid | Amsterdam |
|---|---|---|
| 1 | Year ≤1960 | Year ≤1945 |
| 2 | 1960 < year ≤1984 | 1945 < year ≤1984 |
| 3 | 1984 < year ≤2004 | 1984 < year ≤2004 |
| 4 | Year >2004 | Year >2004 |

Those endpoint details come from the organiser's code. For a simple spoken description, say old, pre-satellite, early-satellite and recent eras, while noting the city-specific first boundary. The construction year creates the answer label; it is never an input feature used to predict that answer.

**Source versus target city.** Madrid supplies many labelled examples from which a reusable model is learned. Amsterdam supplies only a small labelled support set for adapting that model. A new city is not automatically interchangeable with the source.

**Domain shift.** The relationship between measurements and classes can change between cities. Here mean spectral features differ, class frequencies differ, and the first age boundary differs. Climate, materials and land cover are plausible influences described by organisers; our experiments do not isolate their individual effects.

**Macro F1.** Precision asks: among pixels predicted as this class, how many are right? Recall asks: among pixels truly in this class, how many did we find? F1 combines precision and recall. Macro F1 gives equal weight to the four class F1 scores, even when one class has many more pixels. It is not overall accuracy. A score of .62 means .62 on this metric, not necessarily 62% of pixels correct.

**Zero-shot.** Predict Amsterdam with a Madrid-trained model and no Amsterdam labels used to fit it. We can still use Amsterdam labels afterwards to measure how well it performed; this does not make its training few-shot.

**Few-shot.** Use a small labelled Amsterdam support set. At 25/class there are four times 25, or 100 labelled pixels total. The query pixels are those being evaluated and must be excluded from that trial's support.

**Prototype classifier.** Average the support feature vectors separately for each class. A query takes the class whose average is closest. This is a simple reference rule; our raw prototype uses fixed Euclidean distance after Madrid scaling, not a learned Madrid neural metric.

**Logistic regression.** A classifier that learns weighted combinations of features and linear class boundaries from the support labels. Despite its name, we use it for four-class classification, not predicting a construction year. Regularisation discourages excessively large weights. It can learn which feature combinations distinguish Amsterdam classes better than treating all distance directions equally.

**Embedding.** A new vector representation of a pixel. Our network compresses 60 inputs into a unit-length vector with 16 numbers. A useful embedding would preserve the information needed by the new city's classifier.

**Metric learning.** Train a representation or distance so same-class examples are close and different-class examples are farther apart. It is one possible transfer strategy, not an organiser requirement.

**Triplet loss.** A training example contains an anchor, a positive example of the same class, and a negative of another class. The loss asks the positive to be closer than the negative by a margin. Our network learned from Madrid triplets only. It was not explicitly trained to make city identity disappear.

**Feature ablation.** Test a restricted group of inputs to understand what information is useful. We tested groups alone, such as annual changes or all variability statistics. We did not test removing every group from the full representation, so we cannot claim each group is essential.

**Why leakage matters.** A model can appear successful by using information it will not have on a new prediction. Examples include construction year as an input, the same pixel's years split between train and test, fitting preprocessing on evaluation data, or repeatedly tuning to a test set. We excluded support from query per trial, but repeatedly inspected query scores to choose methods. Our numbers are therefore internal development results; official held-out evaluation is still pending.

**Standardisation.** Put features on comparable numerical scales using estimated means and SDs. A Madrid scaler contains feature-distribution information, not learned age-class relationships. That is why a scaler alone is a weak transfer argument.

**Random Forest and the current transfer candidate.** A forest contains many decision trees trained using Madrid features and age labels. At up to 50 labels/class, the existing method multiplies adjusted forest prediction scores with the target prototype's distance-based scores. At 100 and 200/class, it compares query leaf assignments with the most common support leaf for each class in each tree. Both reuse learned Madrid information. Describe the first branch as a score-product heuristic and the second as modal-leaf agreement. Do not claim proven Bayesian calibration.

## Our decision timeline

| Step | What we observed and thought | What we tested and found | Decision and its limit |
|---|---|---|---|
| Reproduce baseline | We needed a trustworthy starting point | Recreated the organiser features and prototype scores; the forest's numbers differed from stored notebook results | Keep a local reference; the forest discrepancy's cause is unresolved |
| Audit baseline | Repeated years and nearby pixels could inflate results | One row per pixel; no duplicate pixel-year rows in prior audit; nearby pixels frequently cross random folds; source scaler was fitted before CV | Preserve pixel separation; explain spatial and preprocessing limits |
| Raw logistic | A class average might be too restrictive | Same support/query splits, but learn linear boundaries; .6187 at 25/class versus .5960 for prototypes | Strong target benchmark; not automatically a compliant transfer method |
| Triplet embedding | Madrid labels might teach reusable similarity | Fixed 16-dimensional embedding; prototype .5804 and logistic .5584 at 25/class | Reject this configuration; no proof that metric learning in general fails |
| Hard negatives | More challenging negative examples might help | Nearest of eight source negatives; loss stayed near .2 and target scores worsened | Reject; “collapsed” is an unproven diagnosis |
| Feature subsets | Changes or variability might transfer better than absolute levels | Variability was best reduced group; annual changes alone were weak; all-60 logistic won at 25–200 | Reject change-only superiority; retain full features for the benchmark |
| Shift statistics | Large city differences might identify bad features | Absolute means shifted most, but lower-shift groups did not reliably perform better | Use the diagnostic insight; do not turn it into a universal feature-selection rule |
| Support rescaling | Target support scales might help | .6149 at 25/class, slightly below .6187; no budget mean improved | Reject this rescaling; do not say all domain alignment fails |
| Teammate's LDA and RF | Genuine transfer still needed | LDA lost; RF product/modal-leaf pipeline reports .5976 at 25 and .6215 at 200 | RF is provisional primary because it uses the source model; reproduce its artifacts/results |
| Final audit | Best F1 and best submission compliance were different | Raw numbers reproduced, missing model and inference path found, misleading claims identified | Freeze new architectures; validate and package existing RF transfer before making slides |

This is the logical sequence of the investigation. It is not a claim that all teammate branches were run in that chronological order: the pulled log is incomplete.

## Five facts everyone must remember

1. This is Madrid → Amsterdam transfer, with four classes and five label budgets. Twenty-five per class means 100 labels total.
2. Raw all-60 logistic is verified at .6187 ± .0227 for 25/class and .6867 ± .0056 for 200/class, but it learns its age-class coefficients from Amsterdam only.
3. Our tested Madrid triplet methods lost. We know the outcome; their causal failure explanation is not established.
4. Lower marginal feature shift did not reliably identify the best tested subset. Annual changes alone lost; the complete representation was best at 25–200/class for the raw logistic benchmark.
5. The existing genuine RF transfer candidate is weaker at 25/class (.5976 reported), and still needs a saved-model/inference/reproduction gate. All results are internal; SD bars do not certify generalisation.

## Judge questions and factual answers

**1. If your best model is Amsterdam logistic regression, where is the transfer learning?**
The raw logistic benchmark only reuses Madrid scaling. We do not claim that it substantively transfers Madrid age supervision. Our provisional primary uses the Madrid-trained forest's predictions and partitions; its final numbers must be attached to the reproduced artifact.

**2. Why should we reward a metric-learning attempt that performed worse?**
The attempt alone deserves no special reward. Its value is the controlled comparison and the limited lesson that source class separation did not ensure better target classification in this setup. We do not present failure as a successful model.

**3. Why did you use logistic regression?**
It is a cheap, regularised way to learn class boundaries from a small support set. On identical trials it outperformed raw prototypes at 25–200/class. We did not prove it is optimal among all classifiers.

**4. Why did you not tune the neural network more?**
The fixed setup and one mining adjustment lost, and time was needed for a compliant runnable submission. More tuning might help, but it would also reuse the same development scores. Stopping was a resource decision, not a proof that every network would fail.

**5. Why keep all 60 features when some have large city shift?**
All-60 logistic beat each tested isolated subset at 25–200/class. Marginal mean shift does not measure all class information. We did not prove every feature is necessary, and variability-only was marginally higher at 5/class.

**6. Did you leak Amsterdam information?**
Within each fit, support and query were disjoint and target-trained parameters used support only. We did inspect full supplied target features and query results for analysis and method selection. Consequently these are development results, not an untouched test estimate. We have not accessed official held-out labels.

**7. Why is Class 2 difficult?**
For raw logistic at 25/class its pooled F1 is .556 and recall .517, with errors especially toward Classes 1 and 3. Spectral overlap and different historical boundaries are plausible explanations, but we did not isolate a cause. Class 2 is not simply the smallest class; it is the largest in both datasets.

**8. Why does 25/class matter?**
The organisers score a specific low-data prize at that budget. It uses only 100 target labels total, so source knowledge and support selection matter more than they might with many local labels.

**9. Why should SD bars be interpreted cautiously?**
The ten trials reuse many of the same pixels. The bars describe sensitivity to our support draws, not independent uncertainty about new cities. For neural methods the source network was also held fixed across those trials.

**10. Would this generalise to a third city?**
We have not tested a third city. Our two-city study motivates the problem but does not establish worldwide portability.

**11. Is .6867 your official held-out result?**
No. It is the average internal macro F1 of the raw logistic benchmark at 200 labels/class. Organisers have separate held-out data, and the primary transfer pipeline has different results.

**12. Is your probability multiplication genuinely Bayesian?**
We should call it a score-product heuristic. The forest and prototype scores use the same query features; we have not established a generative likelihood or independent evidence sources. Normalised scores are not automatically calibrated probabilities.

**13. Why divide by Madrid class frequencies when the forest is balanced?**
That is an inherited heuristic whose probabilistic assumptions are not established. Balanced training changes how class frequencies affect scores. We will report its empirical behaviour accurately rather than claim an exact prior correction.

**14. Did the transferred model improve every budget?**
The pulled two-tier method does not beat the stored organiser 25/class number. Its apparent gain over the fixed-split prototype at 25 is very small, and it trails raw logistic there. The architecture is genuinely transfer-based; the performance claim is narrower.

**15. Did hard negatives cause the embedding to collapse?**
The loss stalled close to the margin and target performance worsened. We did not save the diagnostics needed to prove collapse, such as embedding variance and distance distributions.

**16. Why not claim the spatial-validation drop measures leakage exactly?**
The local random run and teammate spatial run lack a fully matched provenance comparison. Nearby-pixel overlap is a warning, but we cannot attribute the whole numerical difference to it. Spatial blocks also touch without buffers.

**17. What does your feature-shift result actually show?**
Average spectral means differ strongly between the cities, but the magnitude of those mean differences did not reliably rank target classification usefulness across our six overlapping subsets. It does not prove causation or universal feature invariance.

**18. Is the nearest-leaf method a standard RF proximity kernel?**
Our implementation matches queries against each class's modal leaf in each tree. Standard average pairwise proximity would average agreement with all support samples. Those are different procedures.

**19. Did normalisation remove domain shift?**
We tested support-based rescaling, which did not improve the raw logistic mean. Shared centring does not change prototype distances; scaling changes feature weights and regularisation. We did not test every form of alignment.

**20. How do you know comparisons use the same splits?**
Our core experiments read one saved file; the audit checked all 50 support/query sets and exactly reproduced raw scores. The teammate's RNG logic yields the same index sets given the same row order, but its original cache is missing, so historical pixel equivalence still needs confirmation.

**21. Where are the trained models? Can we run them now?**
The current checkout lacks the Stage 1 artifact. The runner can train it, but loading it and predicting externally supplied unlabelled query data still needs packaging. Before submission this answer must be replaced with a demonstrated artifact path and command.

**22. What is original here?**
The strongest contribution is the diagnostic connection between controlled feature-subset results and city shift, plus an honest source-versus-target comparison. We credit the organiser features and suggested metric/leaf ideas; we do not claim to have invented them.

**23. Why does your 25-shot prototype score differ from the notebook?**
The organiser run drew an extra 10/class condition first. That consumed random draws and changed the later support samples. Our fair comparison uses .5960 on the saved five-budget splits, not .5986 from the other stream.

**24. Are Madrid and Amsterdam classes exactly comparable?**
Classes 3 and 4 share boundaries. Classes 1 and 2 have a different historical split: 1960 in Madrid and 1945 in Amsterdam. Transfer must accommodate that difference; a class ID is not an identical calendar interval in both cities.

## Rehearsal and handoff checks

Each speaker should explain their segment without reading code, state which method owns each displayed number, and answer both “what did we establish?” and “what remains untested?” Have another member interrupt with one hostile question. Correct the answer using the audit rather than inventing a favourable explanation.

Before slide creation, Alex and Samson must demonstrate the saved-model adaptation command on unlabelled query inputs. Ubayd should freeze the final result table and claims. Israel should confirm the presentation logistics and prepare the concise limitations/Q&A transitions. After those gates, write a first-slide abstract of at most 150 words and a justification of at most 300 words covering design, transfer and F1 interpretation.
