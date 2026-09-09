# Slide 5 — We stress-tested our own few-shot result

**Speaker:** D · **Time:** 1:30 · **Figure:** `fig_slide5_spatial.png`
(per-class bars and the confusion matrix are **backup only** — pull them up if asked)

---

## 1. What this slide says

We checked how the few labelled patches are chosen and found that **~80% of them
sit right next to a patch we're scored on**. So we re-ran the whole pipeline with
the labels forced to be **far away** — a full map-tile from anything we score.
**About half the few-shot gain disappeared.** The no-label result is untouched.

## 2. In plain words

- The challenge picks the few labelled patches **at random**. Patches next to
  each other on the map look almost identical — often the same building or block.
- So when a labelled patch sits next to a scored patch, the model is basically
  being shown the answer to its neighbour. Across every label budget, ~80% of our
  labelled patches have a scored patch right next to them. Two teammates measured
  this separately and got the same number.
- **This is not cheating.** We never use the scored patches' answers, and random
  sampling is the organiser's own rule. But it means our few-shot scores are a
  little flattering.
- **To measure how much:** we re-ran the exact same pipeline, but only allowed
  labels from map-tiles a full tile away from whatever we were scoring — so no
  labelled patch is anywhere near a scored one.
- **Result:** the gain from 50–200 labels dropped from about **+0.05 to +0.09**
  down to about **+0.01**. Roughly half the few-shot improvement was "my
  neighbour was labelled," not the model genuinely learning to handle new areas.
- **The no-label result (0.36 → 0.65) uses no labels at all**, so there's no
  neighbour effect — it's completely unaffected. That's our solid, trustworthy
  number.
- **We also corrected an earlier assumption.** We used to say classes 1 and 2
  (the two oldest) can't be told apart. That was true of the rough starter model.
  Our final pipeline separates them fine — **class 1 is actually our best class
  (~0.78)**, and classes 2, 3, 4 sit together near 0.70.

## 3. The picture

- **Blue line** — the gain the labels give on the organiser's normal (random)
  test. Rises from +0.03 at 25 labels to +0.09 at 200.
- **Orange line** — the gain when the labels are held a map-tile away. Flat, near
  zero, with a wide shaded band.
- **The gap between the two lines is the neighbour effect.**
- The shaded band is because scoring one small tile at a time is noisy
  (±0.07–0.09). The direction is solid; the exact points are soft.

## 4. What to actually say (~40 s)

> "We stress-tested our own result. The few labelled patches are picked at
> random, and it turns out about 80% of them sit right next to a patch we're
> scored on — and neighbouring patches look almost the same.
>
> That's not against the rules — the organiser's split is random and we never
> touch the scored answers. But we wanted to know how much of our few-shot number
> was real. So we re-ran everything with the labels held a full map-tile away.
>
> About half the few-shot gain vanished. [point at the gap] That half was the
> labels being next to what we scored.
>
> The no-label result — 0.36 to 0.65 — uses no labels, so it has no neighbour
> effect. It's untouched. That's the number we stand behind."

## 5. If someone asks

- **"So are your numbers wrong?"** No — they're the right numbers for the
  organiser's test. We're being upfront that part of the few-shot lift is
  proximity, and pointing at the no-label result as the robust one.
- **"Why is the orange line bumpy, even negative once?"** Scoring one small tile
  at a time is noisy — plus or minus 0.07 to 0.09. The trend is reliable; a
  single point isn't.
- **"Did you break the leakage rule?"** No. Leakage means using the test answers
  in training — we never do. This is a different thing: nearby patches looking
  alike. We disclose it and we measured it.
- **"What changed about classes 1 and 2?"** The rough baseline model couldn't
  separate the two oldest bands. Our final pipeline, with the construction-timing
  features and the line-up, does. Class 1 is now our strongest.

## 6. Words to avoid

| don't say | say instead |
|---|---|
| spatial autocorrelation | nearby patches look almost the same |
| spatial leakage / data leakage | labelled patches sitting next to scored patches |
| support set / query set | the labelled patches / the patches we're scored on |
| spatial-block cross-validation | holding the labels a map-tile away |
| standard error | how noisy a single tile's score is |
