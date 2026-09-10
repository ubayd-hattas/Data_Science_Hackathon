# Slide 5 — We checked how much of the few-shot gain is real

**Speaker:** D · **Time:** 1:30 · **Figure:** `fig_slide5_bars.png` — grouped bars,
"what the labels add" on a normal test (tall blue) vs with labels held a map-tile
away (short orange), at 50 / 100 / 200 labels per class.

---

## 0. Layout

```
┌─────────────────────────────────────────────────┐
│  TITLE: We checked how much of the few-shot     │
│         gain is real                            │
│  ┌──────────────────┐   01 · random labels…      │
│  │                  │   02 · held away…          │
│  │ fig_slide5_bars  │   03 · half vanished…      │
│  │ .png             │                            │
│  └──────────────────┘                            │
│  footer (small): we also corrected an earlier   │
│  claim — class 1 is our strongest, not weakest  │
└─────────────────────────────────────────────────┘
```

The three points **walk through the chart in order** — they are not three
separate topics. 01 = why the chart exists, 02 = what the second bar is,
03 = what it means. The class 1/2 correction is a **small footer line**, not
a fourth numbered point — mixing it in with the audit made an earlier draft of
this slide read as two unrelated topics and confused even us. Keep it separate
and small.

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
- **Separately, we also corrected an earlier assumption** (footer line, not part
  of the audit): we used to say classes 1 and 2 (the two oldest) can't be told
  apart. That was true of the rough starter model. Our final pipeline separates
  them fine — **class 1 is actually our best class (~0.78)**, and classes 2, 3, 4
  sit together near 0.70.

## 3. The picture (`fig_slide5_bars.png`)

- Three pairs of bars — one pair for each label budget (50, 100, 200 per class).
- **Tall blue bar** — how much the labels add on the organiser's normal test
  (+0.05, +0.07, +0.09).
- **Short orange bar** — how much they add when the labels are held a map-tile
  away (about 0 every time; a hair negative at 50, within noise).
- **The blue bar towering over the orange one is the neighbour effect.**
- Per-tile scoring is noisy, so read the *pattern* (blue big, orange near zero at
  every budget), not the exact orange values.

## 4. The three points, exactly as they go on the slide

```
01 · The few labelled patches are picked at random —
     80% land next to a patch we're graded on
02 · We re-ran it with the labels held a full map-tile away
     (the short bars)
03 · Half the gain vanished — it was proximity.
     Zero-shot (no labels) is untouched — that's the solid result
```
```
footer (small): we also corrected an earlier claim — class 1 is our
strongest class, not our weakest; no single unsolvable pair
```

## 5. What to actually say (~40 s)

> "We stress-tested our own result. The few labelled patches are picked at
> random, and it turns out about 80% of them sit right next to a patch we're
> scored on — and neighbouring patches look almost the same.
>
> That's not against the rules — the organiser's split is random and we never
> touch the scored answers. But we wanted to know how much of our few-shot number
> was real. So we re-ran everything with the labels held a full map-tile away —
> [point at the short bars].
>
> About half the few-shot gain vanished. That half was the labels being next to
> what we scored.
>
> The no-label result — 0.36 to 0.65 — uses no labels, so it has no neighbour
> effect. It's untouched. That's the number we stand behind.
>
> One more thing while we're being honest about our own numbers: we used to say
> the two oldest classes can't be told apart. That turned out to be true of the
> rough starter model, not ours — class 1 is actually our strongest."

## 6. If someone asks

- **"So are your numbers wrong?"** No — they're the right numbers for the
  organiser's test. We're being upfront that part of the few-shot lift is
  proximity, and pointing at the no-label result as the robust one.
- **"Why is the orange bar sometimes near zero or slightly negative?"** Scoring
  one small tile at a time is noisy — plus or minus 0.07 to 0.09. The trend is
  reliable (small everywhere); a single bar's exact value isn't.
- **"Did you break the leakage rule?"** No. Leakage means using the test answers
  in training — we never do. This is a different thing: nearby patches looking
  alike. We disclose it and we measured it.
- **"What changed about classes 1 and 2?"** The rough baseline model couldn't
  separate the two oldest bands. Our final pipeline, with the construction-timing
  features and the line-up, does. Class 1 is now our strongest.

## 7. Words to avoid

| don't say | say instead |
|---|---|
| spatial autocorrelation | nearby patches look almost the same |
| spatial leakage / data leakage | labelled patches sitting next to scored patches |
| support set / query set | the labelled patches / the patches we're scored on |
| spatial-block cross-validation | holding the labels a map-tile away |
| standard error | how noisy a single tile's score is |
