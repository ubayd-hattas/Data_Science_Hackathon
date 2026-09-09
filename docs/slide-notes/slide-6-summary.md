# Slide 6 — Cross-city transfer is a calibration problem, not a hard one

**Speaker:** all (each says their line) · **Time:** 0:40 → Q&A · **Figure:** none

---

## 1. What this slide says

The takeaways. A model in a new city isn't incapable — it's mis-tuned. The
reusable win is the **label-free line-up** (0.36 → 0.65). Local labels help most
when they're **near** what you're predicting. Careful use of the data beat every
fancier model. Next: make the labels-held-away test standard, and try a third
city.

## 2. In plain words

- **The headline.** A working model that fails in a new city usually just needs
  its number-ranges adjusted — and you can do that with **no local labels**.
- **What's genuinely new from us:**
  1. Doing the adjustment **one age group at a time**, using the model's own
     guesses.
  2. The finding that **"a few labels" and "many labels" need different
     machinery** — not the same pipeline scaled down.
- **What we'd do next:**
  - Always test with labels **held far away**, not just the random split.
  - Add a **third city** to check the method isn't specific to Madrid → Amsterdam.
- **Team roles** are on the slide so it's clear who did what — the rubric checks
  this and checks that every member speaks.

## 3. The numbers on this slide

None new. It's a recap — do **not** introduce a number here that wasn't on an
earlier slide.

## 4. What to actually say (~35 s, split across the team)

> "Three things to take away.
>
> One — a model in a new city is mis-tuned, not incapable. Lining the cities up
> recovers most of the gap with zero labels.
>
> Two — local labels help most when they sit near what you're predicting; on the
> standard split about fifty per class buy home-city accuracy.
>
> Three — careful use of the data beat every fancier model we tried.
>
> Our original contribution is the group-by-group line-up driven by the model's
> own guesses, and the framing that few and many labels need different machinery.
>
> Roles are on the slide. Happy to take questions."

## 5. If someone asks

- **"What's the single most reusable result?"** The no-label line-up: 0.36 → 0.65
  with zero Amsterdam labels. It has no caveats.
- **"Would this work for city X?"** That's exactly the third-city test we'd run
  next. The mechanism — align, then add labels — isn't Madrid-specific.
- **"If you had one more week?"** Spatial-block-disjoint evaluation as the
  default, and a third city.

## 6. Words to avoid

| don't say | say instead |
|---|---|
| calibration (unexplained) | the model's number-ranges being off |
| generalisation gap | how much worse it does in a new city |
| few-shot regime | the case where you only have a handful of labels |
