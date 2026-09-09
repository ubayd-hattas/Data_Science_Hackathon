# Presentation — final version (10 slides, ~10 min)

Data-science story: question → observation → hypothesis → experiment → result →
honest analysis → conclusion. Rubric-aligned; every requirement tagged **[R]**.

**How to use this file.** Each slide gives:
- the **heading**,
- **paragraph points** you can paste straight onto the slide (trim to taste —
  each is one self-contained thought),
- a **► VISUAL** line saying exactly what picture / table / chart goes there and
  whether it is a required deliverable,
- speaker and time.

Full spoken script: `docs/TALK_SCRIPT.md`.

**Numbers used throughout:** Madrid in-city macro-F1 **0.664 ± 0.004** ·
Amsterdam zero-shot **0.358 raw → 0.578 pooled CORAL → 0.651 class-conditional** ·
Amsterdam few-shot **0.662 / 0.677 / 0.704 / 0.721 / 0.737** (± 0.003–0.009) at
**5 / 25 / 50 / 100 / 200** labelled pixels per class · provided-notebook
baseline **0.43** zero-shot and **0.42 / 0.55 / 0.61 / 0.64 / 0.67** few-shot.

---

## Slide 1 · Can a model trained in one city work in another?
*A · 0:30*

- Building-age records exist for only a handful of cities. If a model trained
  where labels are plentiful could transfer to a city with almost none, building
  age could be mapped almost anywhere — that is the problem worth solving.
- Concretely: we train on **Madrid**, where labels exist, and adapt to
  **Amsterdam** using only a small number of local labels. The task is to place
  each 30 m Landsat pixel in one of four construction-era classes, scored on
  macro-F1.

**► VISUAL:** title, authors, team name, and the **149-word abstract in a small
box** in the lower third. *(Abstract on slide 1 is a required deliverable.)*
Optional faint full-bleed aerial image of a European city behind the text.
**[R]** abstract on slide 1.

---

## Slide 2 · The transfer gap is real — and it is two effects, not one
*A · 1:00*

- A Random Forest trained on our Madrid features scores **0.66** tested on Madrid,
  but only **0.43** when applied straight to Amsterdam. That 0.23 drop is the
  whole problem.
- The first cause is **covariate shift**: different building materials, a wetter
  climate, and a different mix of Landsat sensors across 40 years all move where
  the features sit. The second is **label shift**: Amsterdam has about 1.6× more
  of the oldest class than Madrid.
- Because both the feature distributions *and* the class proportions differ,
  retraining alone cannot close the gap — the distributions have to be aligned.

**► VISUAL:** two side-by-side image crops — a Madrid block (warm tint, labelled
MADRID) and an Amsterdam block (cool tint, labelled AMSTERDAM), both the same age
class, captioned *same class, different distribution*. If no real tiles are
available, use two large labelled colour fields.
**[R]** Challenge Understanding — domain shift.

---

## Slide 3 · Our hypothesis: the model is mis-calibrated, not incapable
*A → B · 0:45*

- The decision rules the forest learned are almost certainly sound — old stock
  weathers, newer stock carries a construction signal. What is wrong is the
  *coordinates*: a feature value that means "old" in Madrid lands somewhere else
  in Amsterdam.
- This gives a testable prediction: if the problem is calibration, then aligning
  the two cities' feature distributions **before** prediction — using no
  Amsterdam labels — should recover most of the 0.23 gap. Slides 4–5 test it.

**► VISUAL:** one large figure — the zero-shot score **0.43** with a red gap bar
rising to the Madrid in-city line at **0.66**. Small inset: a 3-box thumbnail of
the pipeline (Madrid → align → predict) so the audience has the map before the
detail.
**[R]** Challenge Understanding — problem framing.

---

## Slide 4 · The experiment that worked: align the distributions, class by class
*B · 1:30*

- Standard CORAL reshapes the whole Madrid feature cloud to match Amsterdam's
  covariance. We extend it: **class-conditional CORAL** predicts Amsterdam once,
  then re-aligns each Madrid class to the covariance of the pixels the model
  assigned to that class, and refits. Two rounds. Only the model's own
  pseudo-labels are used — never a real Amsterdam label.
- Zero-shot macro-F1 moves **0.36 → 0.58** with pooled CORAL, then **→ 0.65**
  with the class-conditional version. The hypothesis holds: most of the gap was
  calibration.
- **Leakage discipline:** every alignment statistic is computed from the
  *unlabelled* Amsterdam pool; the only labelled target data that ever enters the
  model is the per-trial support set.

**► VISUAL:** the pipeline diagram as **Fig. 1** (three boxes: `Madrid features →
Random Forest (500 trees)` → `class-conditional CORAL ×2` → `predict Amsterdam
0.36→0.65`). Strongly recommended second figure: a **before/after PCA scatter** —
Madrid and Amsterdam point clouds separated, then overlapping after alignment.
**[R]** Model Design + Transfer Strategy (pillars 1 & 2); "did the model address domain shift".

---

## Slide 5 · Then a sequence of small, measured gains — and the low-data regime is its own problem
*B → C · 2:00*

- **Ablation, largest first.** Class-conditional alignment **+0.07** (zero-shot).
  Budget-scaled feature whitening **+0.15 at 5 labels/class** — the single biggest
  lever in the low-data regime. Blending a local model with the Madrid prior
  **+0.04 at 5 labels**. Inverse-distance neighbour smoothing **+0.01** at every
  budget. A 500-configuration feature-and-hyperparameter search **+0.02–0.03**.
- **Key finding.** The few-shot pipeline is not the full pipeline with less data.
  With 5–25 labels the local model is unstable, so the blend leans on Madrid and
  the whitening is dialled down; with 200 it leans local. The recipe *changes*
  with the label budget — this is why the 25-labels-per-class regime needs its
  own treatment.

**► VISUAL:** an **ascending step-chart** — the baseline bar on the left, each
gain stacked on top, the final score on the right. (I can generate this PNG.)
**[R]** design decisions / what worked; Challenge Understanding — low-data mechanics.

---

## Slide 6 · Five things we tried that did not help — reported on purpose
*C → D · 1:15*

- **Learned representations lost.** A triplet-loss embedding added variance with
  no gain — the 108 features are already linearly separable, so it had nothing to
  learn. Gradient boosting matched the forest in-city but collapsed at 5 labels.
- **"Sound" corrections backfired.** Ordinal training made errors *smaller* but
  not *fewer*, and macro-F1 only counts right versus wrong. Label-shift EM needs
  calibrated probabilities the domain gap does not provide. Self-training on the
  unlabelled pool amplified its own ~35% early error.
- **The pattern:** every attempt to out-model the data lost to using it more
  carefully. We show the failures because the reasoning behind them is part of
  the result.

**► VISUAL:** five ✗ rows, plain text, each with its one-line reason. No chart
needed.
**[R]** Originality — insightful failure; scientific soundness.

---

## Slide 7 · We audited our own evaluation
*D · 1:00*

- **What we checked.** Support pixels are drawn at random per class, exactly as
  the protocol specifies. We measured how many of them sit immediately adjacent
  to a query pixel on the map.
- **Finding.** About **80%** of support pixels have a touching query neighbour,
  and this holds at every budget. Two team members measured it independently. At
  30 m resolution, adjacent pixels are often the same building.
- **What we do about it.** No rule is broken and query labels are never used —
  but part of every few-shot score reflects same-block proximity, not pure
  cross-location generalisation. We state the number on the slide, and we kept
  the neighbour smoothing prediction-only so it cannot compound the effect.

**► VISUAL:** a small pixel grid — filled squares for support pixels, outlined
squares for their touching query neighbours — showing how interleaved they are.
**[R]** scientific soundness; Originality — insightful analysis.

---

## Slide 8 · Error analysis: the ceiling is in the data, not the model
*D · 0:45*

- **Where the errors concentrate.** Classes 3 and 4 were built during the
  satellite record, so they carry a visible construction event and are predicted
  well. Classes 1 and 2 are both pre-1984 — there is no event to separate them —
  and that is where most of the residual error sits (class-2 recall ≈ 0.52).
- **This matches prior work.** The building-age literature reports the same
  pattern: pre-war stock is the hardest to date from remote sensing anywhere,
  not just here.

**► VISUAL:** a **4-bar chart** — macro-F1 per age class, classes 3 & 4 tall,
classes 1 & 2 short. (I can generate this PNG.) Alternative: a 4×4 confusion
matrix with row-normalised proportions.
**[R]** F1 interpretation — limitations; what we learnt.

---

## Slide 9 · Result: with ~50 labels per class, transfer reaches the in-city ceiling
*C · 1:45*

- **The table.** Madrid 5×5 cross-validation: **0.664 ± 0.004**. Amsterdam
  few-shot: **0.66 / 0.68 / 0.70 / 0.72 / 0.74** at 5 / 25 / 50 / 100 / 200
  labels per class, with ± 0.003–0.009. The provided-notebook pipeline over the
  same budgets: 0.42 → 0.67 — so this is +0.07 to +0.23 per budget.
- **The curve.** Steep improvement to about 50 labels, then it flattens — right
  at the Madrid in-city score. The alignment does the heavy lifting; the labels
  buy the last few points. A comparable public competition reached ~58%
  satellite-only on a 7-class version; we are on harder 30 m data.
- **Reading it.** Past ~50 labelled buildings per class — roughly 200 total —
  moving the model to a new city costs essentially nothing.

**► VISUAL (both required deliverables):**
- **Fig. 2 — the curve:** macro-F1 vs log₂(labels per class), ±1 SD error bars, a
  dashed horizontal line at the Madrid score 0.66. Large, centred. Source:
  `results/transfer_curve.png` (or the restyled SVG in the HTML deck).
- **Tab. 1 — the table:** Madrid CV + the five Amsterdam scores, mean ± SD,
  booktabs style (rule above header, rule below header, rule at bottom, no
  vertical lines). Small, beside or below the curve. Source:
  `results/deliverable_table.csv`.
**[R]** the required table + plot with error bars; F1-interpretation pillar.

---

## Slide 10 · Conclusion: cross-city transfer is a calibration problem
*D → all · 0:40*

- **Three takeaways.** A model in a new city is mis-calibrated, not incapable —
  per-class alignment recovers most of the gap with zero labels. About 50
  labelled pixels per class buy in-city accuracy. On 30 m Landsat, careful use of
  the distribution beat every learned-representation alternative we tried.
- **What is new here.** The iterative, pseudo-label-driven per-class alignment,
  and the framing that transfer is a spectrum requiring different machinery at
  different label budgets.
- **Next.** A spatial-block-disjoint evaluation for a stricter generalisation
  estimate, and testing the same recipe on a third city.

**► VISUAL:** none — three lines of text plus, small at the bottom:
⟨A⟩ features & pipeline · ⟨B⟩ domain alignment · ⟨C⟩ evaluation harness ·
⟨D⟩ analysis & write-up.
**[R]** Originality — framing; Presentation — team, all four named.

---

## Slide-by-slide asset checklist

| slide | asset | required? | have it / make it |
|---|---|---|---|
| 1 | abstract box | **yes** | `docs/WRITTEN_JUSTIFICATION.md` has the text |
| 2 | Madrid vs Amsterdam crops | no, strong | make — needs raw tiles or false-colour render |
| 3 | 0.43 → 0.66 gap figure + pipeline thumbnail | no | make — simple |
| 4 | Fig. 1 pipeline diagram | no, strong | make — boxes/arrows |
| 4 | before/after PCA scatter | no, strong | make — from the 108 features |
| 5 | ascending step-chart of gains | no, strong | make — from the ablation numbers |
| 6 | five ✗ rows | no | plain text |
| 7 | support/query adjacency grid | no | make — small illustrative grid |
| 8 | per-class F1 bars or confusion matrix | no, strong | make — one eval run |
| 9 | **Fig. 2 F1 curve with error bars** | **yes** | `results/transfer_curve.png` |
| 9 | **Tab. 1 F1 table with ± SD** | **yes** | `results/deliverable_table.csv` |
| 10 | — | — | text only |

**Bold = required by the rubric.** The rest are "strongly recommended" — they
carry the story and reduce words. I can generate every "make" item as a PNG from
the project data; say which.

## Timing & handoffs — total ≈ 10:00

| speaker | slides | minutes |
|---|---|---|
| A | 1, 2, 3 | 2:15 |
| B | 4, 5 | 3:30 |
| C | 5→6 handoff, 9 | 2:15 |
| D | 6, 7, 8, 10 | 2:00 |

Slow down on slides 4, 5 and 9. Slides 6–8 are brisk and confident.
