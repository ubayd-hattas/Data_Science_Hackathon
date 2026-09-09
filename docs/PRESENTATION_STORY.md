# Slide content — 10 slides, ~10 min

Only what goes **on** the slides: heading, bullets, and the picture/table to add.
Spoken lines are in `docs/TALK_SCRIPT.md`.

---

## 1 · Can a model trained in one city work in another?

- Building-age records exist for only a few cities
- Train where labels are plentiful (Madrid), transfer to where they aren't (Amsterdam)
- Task: 4 age classes per 30 m pixel · score: macro-F1

**► Add:** authors, team name, and the **149-word abstract** in a small box *(required on slide 1)*.

---

## 2 · The transfer gap is real — two effects, not one

- Madrid model on Madrid: **0.66** · on Amsterdam: **0.43**
- Look different — materials, climate, changing satellites over 40 years
- Age mix differs — oldest class 1.6× more common in Amsterdam
- Retraining alone can't fix this

**► Picture:** two crops side by side — Madrid block vs Amsterdam block, same age, *"same class, different distribution"*.

---

## 3 · Our hypothesis: mis-calibrated, not incapable

- The rules are right; the numbers sit in the wrong place
- If so, line up the two cities' data **before** predicting — with no Amsterdam labels
- That's the test →

**► Picture:** the **0.43 → 0.66** gap as a bar; small pipeline thumbnail beside it.

---

## 4 · The move that worked: align class by class

- Reshape Madrid's data to match Amsterdam — **one age class at a time**
- Uses the model's own first guesses · **no Amsterdam answers** · repeat ×2
- Zero labels: **0.36 → 0.58 → 0.65**
- Only the given examples ever touch the model

**► Picture:** pipeline diagram (**Fig. 1**) + before/after scatter — clouds apart, then overlapping.

---

## 5 · Then a run of small, measured gains

- Line up the cities → **+0.07**
- Untangle features, gently when data is thin → **+0.15 at 5 labels**
- Local model + Madrid model vote together → **+0.04**
- Neighbouring patches vote → **+0.01**
- Overnight search, 500 setups → **+0.02**
- Few labels ≠ full pipeline with less data — a different recipe

**► Chart:** ascending step-chart — baseline → each gain stacked → final.

---

## 6 · Five things that didn't work — shown on purpose

- Neural network — nothing to learn, data already separable
- Ordinal training — smaller mistakes, not fewer
- Self-training — teaches itself its own errors
- Gradient boosting — collapses at 5 labels
- Fixing the age mix — needs trust it doesn't have across cities
- Every attempt to out-think the data lost to using it carefully

**► No picture** — five plain rows.

---

## 7 · We audited our own evaluation

- Examples drawn at random per class — as instructed
- **~80%** of them sit right next to a test patch (measured twice)
- 30 m pixels that touch are often the same building
- Part of the score is proximity, not skill — **we say so**
- Smoothing kept prediction-only so it can't compound it

**► Picture:** small pixel grid — support pixels filled, touching test pixels outlined.

---

## 8 · Error analysis: the ceiling is in the data

- Classes 3 & 4 — built on camera → predicted well
- Classes 1 & 2 — both pre-1984, no construction event → can't be split
- Most remaining error is classes 1↔2 (class-2 recall ≈ 0.52)
- Prior work reports the same for pre-war buildings

**► Chart:** 4 bars — score per age class; 3 & 4 tall, 1 & 2 short.

---

## 9 · Result: ~50 labels reach the in-city ceiling

- Madrid on itself: **0.66 ± .004**
- Amsterdam: **0.66 / 0.68 / 0.70 / 0.72 / 0.74** at 5 / 25 / 50 / 100 / 200 per class
- Baseline (provided notebook): 0.42 → 0.67
- Steep to ~50, then flat — right at the Madrid line
- ~200 checked buildings total → transfer costs nothing

**► Table + plot (both required):**
- **Fig. 2:** score vs log₂(labels/class), error bars, dashed Madrid line — large
- **Tab. 1:** Madrid + 5 Amsterdam scores, mean ± SD — small, beside it

---

## 10 · Conclusion: it's a calibration problem

- Mis-calibrated, not incapable — alignment recovers most of the gap, zero labels
- ~50 labels per class buy home-city accuracy
- Careful data use beat every fancier model
- New here: group-by-group alignment from pseudo-labels + "few vs many need different machinery"
- Next: block-disjoint evaluation · a third city

**► No picture** — plus, small: ⟨A⟩ features · ⟨B⟩ alignment · ⟨C⟩ evaluation · ⟨D⟩ analysis.

---

### Timing — total ≈ 10:00

| speaker | slides | min |
|---|---|---|
| A | 1–3 | 2:15 |
| B | 4–5 | 3:30 |
| C | 5→6, 9 | 2:15 |
| D | 6–8, 10 | 2:00 |

Slow on 4, 5, 9. Brisk on 6–8.

### Figures to make (from the project data — ask and I'll produce PNGs)

slide 2 city crops · slide 4 before/after scatter · slide 5 gains step-chart ·
slide 8 per-class bars · slide 9 restyled curve (base: `results/transfer_curve.png`).
