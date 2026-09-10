# Slide 2 — Line the cities up first, the model does the rest

**Speaker:** B · **Time:** 2:00 · **Figure:** `fig_slide2_pipeline.png` (right side, the only figure on this slide)
**Subheading:** Fix the calibration, not the model — no Amsterdam labels for the big jump

---

## 0. Layout

```
┌────────────────────────────────────────────────┐
│  TITLE (the claim)                             │
│  subheading                                    │
├──────────────────────────┬─────────────────────┤
│  • bullet (not expanded)  │                     │
│  • bullet (not expanded)  │  fig_slide2_pipeline│
│  • bullet (not expanded)  │  .png               │
│  • bullet (not expanded)  │                     │
└──────────────────────────┴─────────────────────┘
```

Bullets left, **pipeline diagram only** on the right — it already shows the
method (Madrid features → line up ×2 → predict, 0.36→0.65, "no Amsterdam
labels used"), so the bullets don't need to repeat it; they cover what the
diagram *doesn't* say. The before/after scatter (`fig_slide2_scatter.png`,
Madrid's cloud landing on Amsterdam's) is **backup**, not on this slide — it
naturally comes up again near slide 3's results, or in Q&A.

## 1. What this earns on the rubric

- **This is your biggest slide for points.** It carries the *Model Design &
  Adaptation* criterion (does the model address domain shift?), the *Spectral
  Representation* criterion (features that generalise, not memorise Madrid),
  and — the single highest-weighted line in the whole rubric — **Methodological
  Innovation** (worth up to 40 points overall): "did they use an unconventional
  or creative domain adaptation technique?" The group-by-group, pseudo-label-
  driven line-up **is** your answer to that question. Say so explicitly.
- Also the **Transfer Strategy** pillar of the written justification — this
  slide is the spoken version of that paragraph. Keep them consistent.

## 2. What this slide says

Before using the Madrid model on Amsterdam, we **reshape Madrid's numbers so
they line up with Amsterdam's** — and we do it **one age group at a time**,
using the model's own first guesses. No Amsterdam answers are used. That single
step takes the no-label score from **0.36 to 0.65**. When we do have a few
labels, we add a small local model that votes alongside the Madrid one.

## 3. In plain words

- **The line-up.** Imagine two thermometers reading the same room but calibrated
  differently — one says 18°, the other 21°. The model learned its rules on the
  first thermometer. Before trusting it on the second, you shift the readings so
  they match. We do that with the two cities' features.
- **One age group at a time** (the bit that's ours). Instead of shifting the
  whole Madrid cloud once, we shift the *old-buildings* part to match Amsterdam's
  old buildings, the *new-buildings* part to match Amsterdam's new buildings, and
  so on. We don't know which Amsterdam patch is which age — so we let the model
  guess, do the shift, then re-guess. Twice. This is our main original idea.
- **When we have a few labels.** Three things:
  1. **Untangle the features.** Many of our 108 numbers secretly say the same
     thing (brightness in blue ≈ green ≈ red), which triple-counts one idea.
     Untangling makes each idea count once. We do this *gently* when labels are
     scarce, because the maths needs enough examples to be stable.
  2. **Train a small local model** on the handful of Amsterdam labels we have.
  3. **Let the two models vote together** — lean on Madrid when labels are few,
     lean on the local model as they grow. The balance shifts automatically.
- **The rule we never break:** only the labels we're handed ever touch the model.
  The reshaping, the untangling, everything else uses only the *unlabelled*
  Amsterdam data, which the challenge allows.

## 4. The numbers on this slide

| number | meaning |
|---|---|
| 0.36 → 0.65 | Amsterdam macro-F1 with **no labels**, before vs after the line-up |
| ×2 | rounds of the group-by-group line-up (2 vs 4 tested; 4 gave no gain, see §10) |

## 5. Bullets for the slide (left column, not expanded — the diagram carries the detail)

```
• One age group at a time — our main idea
• A few labels: untangle the features, blend a small local model with Madrid
• Few labels ≠ the full pipeline with less data — a different recipe
• Only the labels we're given ever touch the model
```

## 6. Timing breakdown (2:00 total)

| segment | time | content |
|---|---|---|
| the line-up | 0:45 | thermometer analogy, group-by-group, ×2, 0.36→0.65 |
| flag the originality | 0:15 | "this is our main idea" — say it, don't just imply it |
| the few-label machinery | 0:40 | untangle, small local model, blend |
| the discipline | 0:15 | only given labels touch the model |
| bridge | 0:05 | to the results |

This is the slide worth slowing down on — it carries the most rubric weight.
Don't rush it to save time elsewhere; cut slide 4 or 6 instead if you're behind.

## 7. What to actually say (~50 s, or up to 1:45 if you have the room)

> "We don't touch the model architecture. We fix the calibration.
>
> Step one: line the two cities up. We reshape Madrid's features so their spread
> matches Amsterdam's. And we do it group by group — reshape Madrid's old
> buildings onto Amsterdam's old buildings, and so on. We don't know Amsterdam's
> ages, so we use the model's own guesses, reshape, guess again. Two rounds.
> This is the part that's ours.
>
> That alone takes the no-label score from 0.36 to 0.65. No Amsterdam labels.
>
> Step two, once we have a few labels: we untangle the features — a lot of them
> double-count the same signal — train a small local model, and blend it with the
> aligned Madrid model. Few labels, trust Madrid. More labels, trust local.
>
> Throughout, the only Amsterdam labels that reach the model are the ones the
> challenge hands us."

## 8. Handoff to Speaker C

> "So that's the method. Next — did it actually work? [Name of C] has the numbers."

## 9. If you're running short — cut to this

Drop the "untangle the features" explanation (it's the hardest idea to compress)
and say: *"With a few labels we also train a small local model and blend it with
the aligned Madrid one — trust Madrid early, trust local as labels grow."*
~10 seconds saved. Never cut the "this is our main idea" line — that's the
originality credit.

## 10. If someone asks

- **"Isn't using the model's own guesses circular?"** A little, but the guesses
  only steer the reshaping, not the final answer, and two rounds is enough —
  more doesn't help.
- **"What's this technique called?"** CORAL — matching the spread and
  correlations of one dataset to another. Standard trick; we made it work per
  age group, driven by pseudo-labels.
- **"Why two rounds?"** We compared 2 vs 4 rounds directly (`results/polish_scores.json`):
  4 rounds gave the same score from 50 labels up and was a touch worse at 5 and
  25. No benefit to going further, so we kept 2. (We have not tested 1 or 3 —
  don't claim a full sweep if asked precisely.)
- **"How can untangling features hurt at low data?"** It estimates relationships
  between features from your examples. With 5 examples per class those estimates
  are garbage, so it adds noise. With 50+ they're solid and it pays off.
- **"Does the alignment actually work — can we see it?"** Yes — pull up
  `fig_slide2_scatter.png`: after the line-up, Madrid's cloud sits on top of
  Amsterdam's.
- **"What if a class has very few Amsterdam patches for the model to guess
  onto?"** It's less stable for that class specifically — this is the same
  reason we scale the feature-untangling by label count later. We haven't
  built a separate fix for a severely rare class beyond what the pipeline
  already does.
- **"Why not align using real Amsterdam labels once you have some, instead of
  guesses?"** We do, for the few-shot stage — the group-by-group step uses
  guesses because it runs at zero labels; the local model in step two uses
  whatever real labels we're given.
- **"How is this different from standard domain adaptation research?"**
  Standard CORAL aligns the whole dataset once. Doing it per age class, driven
  by the model's own iterated predictions, is the part that isn't off-the-shelf.
- **"Could you align Amsterdam onto Madrid instead of Madrid onto Amsterdam?"**
  We tried the direction we use because it lets us keep training on Madrid's
  real labels throughout; aligning the other way would need retraining on
  reshaped labels, which is messier and untested here.

## 11. Common mistakes presenting this slide

- **Not naming the originality.** Don't just describe the group-by-group
  line-up and move on — say the words "this is our main idea" or "this is what's
  original here." Judges are explicitly scoring for this; make it easy to find.
- **Getting lost in "untangle the features."** It's the most abstract idea in
  the deck. Use the double-counting framing ("blue, green and red are all
  saying the same thing") rather than the maths.
- **Skipping the leakage line.** "Only the labels we're given ever touch the
  model" is a one-sentence insurance policy against the disqualification-risk
  deduction. Always say it, even if rushed.

## 12. Words to avoid

| don't say | say instead |
|---|---|
| CORAL / covariance alignment | line the cities up |
| class-conditional | one age group at a time |
| pseudo-labels | the model's own first guesses |
| whitening / ZCA / decorrelation | untangle the features |
| ensemble / probability blend | let the two models vote together |
