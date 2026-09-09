# Slide 2 — Line the cities up first, the model does the rest

**Speaker:** B · **Time:** 2:00 · **Figures:** `fig_slide2_pipeline.png` (top, full width)
+ `fig_slide2_scatter.png` (right half) — before: the two cities' clouds sit apart;
after the line-up: Madrid's cloud has moved onto Amsterdam's, centres on top of each other

---

## 1. What this slide says

Before using the Madrid model on Amsterdam, we **reshape Madrid's numbers so
they line up with Amsterdam's** — and we do it **one age group at a time**,
using the model's own first guesses. No Amsterdam answers are used. That single
step takes the no-label score from **0.36 to 0.65**. When we do have a few
labels, we add a small local model that votes alongside the Madrid one.

## 2. In plain words

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

## 3. The numbers on this slide

| number | meaning |
|---|---|
| 0.36 → 0.65 | Amsterdam macro-F1 with **no labels**, before vs after the line-up |
| ×2 | rounds of the group-by-group line-up (we tested 1–4; 2 was best) |

## 4. What to actually say (~50 s)

> "We don't touch the model architecture. We fix the calibration.
>
> Step one: line the two cities up. We reshape Madrid's features so their spread
> matches Amsterdam's. And we do it group by group — reshape Madrid's old
> buildings onto Amsterdam's old buildings, and so on. We don't know Amsterdam's
> ages, so we use the model's own guesses, reshape, guess again. Two rounds.
>
> That alone takes the no-label score from 0.36 to 0.65. No Amsterdam labels.
>
> Step two, once we have a few labels: we untangle the features — a lot of them
> double-count the same signal — train a small local model, and blend it with the
> aligned Madrid model. Few labels, trust Madrid. More labels, trust local.
>
> Throughout, the only Amsterdam labels that reach the model are the ones the
> challenge hands us."

## 5. If someone asks

- **"Isn't using the model's own guesses circular?"** A little, but the guesses
  only steer the reshaping, not the final answer, and two rounds is enough —
  more doesn't help.
- **"What's this technique called?"** CORAL — matching the spread and
  correlations of one dataset to another. Standard trick; we made it work per
  age group, driven by pseudo-labels.
- **"Why two rounds?"** We swept 1 to 4. Two was the sweet spot; more started to
  drift.
- **"How can untangling features hurt at low data?"** It estimates relationships
  between features from your examples. With 5 examples per class those estimates
  are garbage, so it adds noise. With 50+ they're solid and it pays off.

## 6. Words to avoid

| don't say | say instead |
|---|---|
| CORAL / covariance alignment | line the cities up |
| class-conditional | one age group at a time |
| pseudo-labels | the model's own first guesses |
| whitening / ZCA / decorrelation | untangle the features |
| ensemble / probability blend | let the two models vote together |
