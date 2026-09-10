# Slide 4 — Simple fixes beat clever ones

**Speaker:** D · **Time:** 1:15 · **No figure — build this slide as native Canva text.**

---

## 0. Layout

```
┌────────────────────────────────────────────────┐
│  TITLE: Simple fixes beat clever ones          │
│                                                │
│   ✗  Neural network                             │
│      features already separable — nothing      │
│      to learn                                   │
│                                                │
│   ✗  Ordinal training                           │
│      smaller mistakes, not fewer; the score    │
│      only counts right vs wrong                │
│                                                │
│   ✗  Self-training                              │
│      learns its own early mistakes, then       │
│      compounds them                            │
│                                                │
│   ✗  Gradient boosting                          │
│      ties us with many labels, collapses at 5  │
│                                                │
│   ✗  Correcting the age mix                     │
│      needs confident probabilities the city    │
│      gap breaks                                │
└────────────────────────────────────────────────┘
```

**Build this as five rows of real text, not the `fig_slide4_didntwork.png`
image.** The PNG is a picture of text — you can't restyle it in Canva and its
font won't match your deck. Use it only as a backup reference for the wording,
or in the HTML/research deck where it was originally made.

No subheading, no emoji, no separate closing line — the title **is** the
thesis. Don't restate it again at the bottom; one clear statement, once.

## 1. What this slide says

Things we tried that did **not** beat the simple pipeline: a neural network,
ordinal training, self-training, gradient boosting, and correcting for the age
mix between cities. Each failed for a specific, understandable reason. The
pattern: every attempt to out-think the data lost to a simpler fix that just
used the data more carefully.

## 2. In plain words — one line per dead end

- **Neural network.** The challenge notes suggested one. But our features are
  already easy to tell apart, so there was nothing for a network to learn — it
  just added noise.
- **Ordinal training** (telling the model the classes are in order — 1 is near 2,
  far from 4). It made our wrong answers *less wrong* (off by one instead of off
  by three), but macro-F1 only counts right versus wrong, so the score didn't
  move.
- **Self-training** (let the model label the unlabelled data, then learn from its
  own labels). At low label counts the model is about 35% wrong, so it teaches
  itself its own mistakes and they pile up round after round.
- **Gradient boosting** (a different model type, usually a bit stronger on
  table data). It tied us when labels were plentiful and fell apart with only 5.
- **Correcting the age mix** (Amsterdam has more old buildings than Madrid, so
  in principle you can re-weight for that). It needs the model's confidence
  scores to be trustworthy across cities — they aren't, so it amplified error
  instead of fixing it.
- **Also tried, also negative (Q&A only, not on the slide):** richer
  neighbourhood features (+0.001), and synthetic "mixup" support examples
  (slightly worse — random forests don't gain from blended points).

## 3. Why we show failures

The challenge rubric **explicitly rewards** explaining why a reasonable idea
didn't work. A team that shows its dead ends and the reason for each reads as
more competent than one that only shows wins.

## 4. What to actually say (~35 s)

> "We show what didn't work on purpose — it's the same story as what did.
>
> The neural net the challenge suggested: it lost, because the data was already
> easy to separate, so there was nothing to learn. Ordinal training made our
> mistakes smaller but not fewer, and the score only counts right versus wrong.
> Self-training taught the model its own early mistakes. Gradient boosting tied
> us with lots of labels and collapsed at five. And correcting for the age mix
> between cities needed confidence we didn't actually have.
>
> Every one of these was a reasonable idea. Each lost to a simpler accounting
> fix. That's the pattern of the whole project."

## 5. If someone asks

- **"Did you tune the neural net properly?"** Yes — several sizes and settings.
  The ceiling was the data, not the model.
- **"Would boosting win with more data?"** In-city, maybe by a hair. But the
  challenge is the low-label case, and that's exactly where it breaks.
- **"Isn't ordinal training standard for age bands?"** It is, and it won a
  similar public competition — but that was judged on error *size*. This one
  isn't.
- **"What about the age-mix correction — can't you fix the confidence issue?"**
  Possibly with calibration, but the city gap itself distorts calibration — it's
  the same underlying problem we're already solving with the line-up.

## 6. Words to avoid

| don't say | say instead |
|---|---|
| embedding / representation learning | a neural network that re-describes the data |
| ordinal regression / CORN loss | telling the model the classes are in order |
| semi-supervised / self-training | let the model label the rest and learn from that |
| gradient boosting / XGBoost / HistGBM | a different model type, usually strong on tables |
| label-shift correction / calibration | correcting the age mix |
