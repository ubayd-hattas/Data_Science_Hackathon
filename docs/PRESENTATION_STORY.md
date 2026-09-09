# Presentation — data-science story (10 slides, ~10 min)

Told the way a DS team walks through a project: an observation, a hypothesis, an
experiment, a number, what it means, then the next question. Each slide has 2–3
**short paragraph points** (a sentence or two each) and **one visual**. Full
spoken script in `docs/TALK_SCRIPT.md`. Rubric coverage tagged **[R]**.

Key numbers: zero-shot 0.36 raw → 0.65 aligned · few-shot 0.66 / 0.68 / 0.70 /
0.72 / 0.74 at 5 / 25 / 50 / 100 / 200 known buildings per group · Madrid
in-city 0.66 · provided-notebook baseline 0.43 / 0.42–0.67.

---

## 1 · Can a model trained in one city work in another?

- **The question.** Building-age records exist for a handful of cities. If a
  model trained where labels are plentiful could transfer to a city with almost
  none, you could map building age almost anywhere. We test that with Madrid → Amsterdam.
- **The setup.** Predict one of four construction-era classes per 30 m Landsat
  pixel; adapt to Amsterdam with 5–200 labelled pixels per class; report macro-F1.

**Visual:** title, authors, team; the 149-word abstract in a small box.
**[R]** abstract on slide 1.

---

## 2 · The transfer gap is real and it's two effects, not one

- **Observation.** A Random Forest trained on Madrid features scores **0.66**
  in-city but only **0.43** applied straight to Amsterdam.
- **Why.** Covariate shift — different building materials, wetter climate,
  different Landsat sensors across 40 years — moves where the features sit. On
  top of that, label shift: Amsterdam has ~1.6× more of the oldest class.
- **Implication.** Retraining alone won't fix this; the feature distributions and
  the class priors both have to be handled.

**Visual:** two aerial/Landsat crops — Madrid (warm) vs Amsterdam (cool), same
building age, captioned *same class, different distribution*.
**[R]** Challenge Understanding — domain shift.

---

## 3 · Hypothesis: the model is mis-calibrated, not incapable

- **The idea.** The decision rules the forest learned are probably fine — old
  stock weathers, new stock has a construction signal. What's wrong is the
  *coordinates*: a threshold that means "old" in Madrid lands somewhere else in
  Amsterdam.
- **The test that follows.** If that's true, aligning the feature distributions
  before prediction — with no Amsterdam labels — should recover most of the gap.

**Visual:** one large figure — the zero-shot number **0.43**, a red gap bar down
to the Madrid line at 0.66.
**[R]** Challenge Understanding — the framing.

---

## 4 · The move that worked: align the distributions, class by class

- **Method.** Plain CORAL reshapes the whole Madrid feature cloud to Amsterdam's
  covariance. We go further — **class-conditional CORAL**: predict Amsterdam
  once, then for each class re-align Madrid's examples of that class to the
  covariance of the pixels the model assigned to it, and refit. Two rounds. Only
  the model's own pseudo-labels are used.
- **Result.** Zero-shot macro-F1 goes **0.36 → 0.58** (pooled CORAL) **→ 0.65**
  (class-conditional). No target labels touched.
- **Discipline.** Every alignment statistic comes from the *unlabelled* Amsterdam
  pool; the only labelled target data entering the model is the support set.

**Visual:** 3-box pipeline (Fig. 1) and/or a before/after PCA scatter — two
clouds separated, then overlapping.
**[R]** Model Design + Transfer Strategy (pillars 1 & 2); "addressed domain shift".

---

## 5 · Then a sequence of small, measured gains — and the low-data regime is its own problem

- **Ablation, biggest first.** Class-conditional alignment **+0.07** (zero-shot).
  Budget-scaled feature whitening **+0.15 at 5 labels** — the largest single
  lever in the low-data regime. Blending a local model with the Madrid prior
  **+0.04 at 5 labels**. Inverse-distance neighbour smoothing **+0.01** at every
  budget. A 500-configuration search **+0.02–0.03**.
- **Key finding.** The few-shot pipeline is *not* the full pipeline with less
  data. With 5–25 labels the local model is unstable, so the blend leans on
  Madrid and the whitening is dialled down; with 200 it leans local. The recipe
  changes with the label budget.

**Visual:** an ascending step-chart — baseline → each gain stacked → final.
**[R]** design decisions / what worked; low-data mechanics (why 25/class is distinct).

---

## 6 · Five things we tried that didn't help — reported on purpose

- **Learned representations lost.** A triplet-loss embedding added variance with
  no gain — the 108 features are already linearly separable, so there was
  nothing for it to learn. Gradient boosting matched the forest in-city but
  collapsed at 5 labels.
- **"Sound" corrections backfired.** Ordinal training made errors *smaller* but
  not *fewer*, and macro-F1 only counts right vs wrong. Label-shift EM needs
  calibrated probabilities the domain gap doesn't provide. Self-training on the
  unlabelled pool amplified its own ~35% early error.
- **The pattern.** Every attempt to out-model the data lost to using it more
  carefully.

**Visual:** five ✗ rows, plain, with the one-line reason each.
**[R]** Originality — insightful failure; scientific soundness.

---

## 7 · We audited our own evaluation

- **What we checked.** Our support pixels are drawn at random per class — as the
  protocol specifies. We measured how many sit adjacent to a query pixel.
- **Finding.** ~80% of support pixels have an immediate map-neighbour in the
  query set, stable across every budget. Two teammates measured this
  independently.
- **What we do about it.** No rule is broken and query labels are never used —
  but part of every few-shot score reflects same-block proximity, not pure
  cross-location generalisation. We state the number, and we kept the neighbour
  smoothing prediction-only so it can't compound the effect.

**Visual:** a small pixel grid — filled support pixels and their outlined
touching query neighbours.
**[R]** scientific soundness; Originality — insightful analysis.

---

## 8 · Error analysis: the ceiling is in the data, not the model

- **Where the errors concentrate.** Classes 3 and 4 (built during the satellite
  record) carry a visible construction event and are predicted well. Classes 1
  and 2 are both pre-1984 — no event to separate them — and that's where most of
  the residual error is (class-2 recall ≈ 0.52).
- **Consistent with prior work.** The building-age literature reports the same:
  pre-war stock is the hardest to date from remote sensing anywhere.

**Visual:** 4-bar per-class F1 — classes 3 & 4 tall, 1 & 2 short.
**[R]** F1 interpretation — limitations; what we learnt.

---

## 9 · Result: with ~50 labels per class, transfer reaches the in-city ceiling

- **The table.** Madrid 5×5 CV **0.664 ± 0.004**. Amsterdam few-shot **0.66 /
  0.68 / 0.70 / 0.72 / 0.74** at 5 / 25 / 50 / 100 / 200 per class, ± 0.003–0.009.
  Provided-notebook baseline over the same budgets: 0.42 → 0.67.
- **The curve.** Steep to ~50 labels, then it flattens — right at the Madrid
  in-city score. The alignment does the heavy lifting; the labels buy the last
  few points.
- **Reading it.** Past ~50 labelled buildings per class — about 200 total —
  moving the model to a new city costs essentially nothing.

**Visual:** the F1-vs-log₂(labels) curve, large, with ±1 SD error bars and a
dashed Madrid line; the table small beside it. **Both are required deliverables.**
**[R]** the required table + plot with error bars; F1-interpretation pillar.

---

## 10 · Conclusion: cross-city transfer is a calibration problem

- **Three takeaways.** A model in a new city is mis-calibrated, not incapable —
  per-class alignment recovers most of the gap with zero labels. About 50
  labelled pixels per class buy in-city accuracy. And on 30 m Landsat, careful
  use of the distribution beat every learned-representation alternative we tried.
- **What's new.** The iterative, pseudo-label-driven per-class alignment, and the
  framing that transfer is a spectrum — different machinery at different label
  budgets.

**On slide, small:** ⟨A⟩ features & pipeline · ⟨B⟩ domain alignment · ⟨C⟩
evaluation harness · ⟨D⟩ analysis & write-up.
**[R]** Originality — framing; Presentation — team, all four named.

---

## Story arc

| # | Beat | DS move |
|---|---|---|
| 1 | Can it move cities? | the question |
| 2 | The gap is two effects | the observation |
| 3 | Mis-calibrated, not incapable | the hypothesis |
| 4 | Align class by class | the experiment that worked |
| 5 | A sequence of measured gains | the ablation + a key finding |
| 6 | Five that didn't help | negative results |
| 7 | We audited our evaluation | the honesty check |
| 8 | The ceiling is in the data | error analysis |
| 9 | 50 labels reach the ceiling | the result |
| 10 | It's a calibration problem | the conclusion |

## Timing & handoffs — total ≈ 10:00

A: 1–3 (2:15) · B: 4–5 (3:30) · C: 5→6 hand, 9 (2:15) · D: 6–8, 10 (2:00).
Slow on 4, 5, 9.

## Figures to generate (I can produce these as PNGs from the data)

1. Madrid vs Amsterdam crops (slide 2)
2. before/after PCA scatter (slide 4)
3. ascending step-chart of the gains (slide 5)
4. F1 curve restyled to the deck (slide 9) — base is `results/transfer_curve.png`
5. per-class F1 bars (slide 8)
