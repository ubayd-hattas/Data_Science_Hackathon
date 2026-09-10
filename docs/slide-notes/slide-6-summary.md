# Slide 6 — It's a calibration problem, not a hard one

**Speaker:** all (each says their line) · **Time:** 0:40 → Q&A · **Figure:** none

---

## 0. Layout

```
┌──────────────────────────────────────────────────────────────┐
│   It's a calibration problem, not a hard one          ← TITLE │
│                                                              │
│   • A new-city model is mis-tuned, not incapable      ← 3     │
│                                                        BULLETS│
│   • The reusable win: label-free group-by-group               │
│     alignment — 0.36 → 0.65                                   │
│                                                              │
│   • Careful use of the data beat every fancier model         │
│                                                              │
│   New here: per-group alignment · few vs many          ← small│
│   labels need different machinery                       line 1│
│                                                              │
│   A — features · B — the line-up · C — evaluation      ← small│
│   D — analysis                                          line 2│
└──────────────────────────────────────────────────────────────┘
```

No figure. Four zones top to bottom: **title** (the thesis, stated plainly —
not the word "summary"), **three bullets** (big text, the takeaways), **small
line 1** (what's original — the rubric scores this explicitly), **small line
2** (who did what — the rubric checks every member is credited and spoke).

**Check your team name is consistent** with slide 1 — don't let one slide say
"F1DEVS UNLIMITED" and another say a different team name.

## 1. What this slide says

The takeaways. A model in a new city isn't incapable — it's mis-tuned. The
reusable win is the **label-free line-up** (0.36 → 0.65). Careful use of the
data beat every fancier model. It's a recap: **introduce no new numbers here.**

## 2. In plain words

- **The headline.** A working model that fails in a new city usually just needs
  its number-ranges adjusted — and you can do that with **no local labels**.
- **Lead with the line-up, not the label count.** An earlier draft of this slide
  closed on "50 labels = home-city accuracy" — but slide 5 just showed that half
  of that gain is proximity. Ending the whole talk on the most-caveated number is
  the wrong note. Close on the thing with **no caveat**: the zero-label line-up.
- **What's genuinely new from us:**
  1. Doing the adjustment **one age group at a time**, using the model's own
     guesses.
  2. The finding that **"a few labels" and "many labels" need different
     machinery** — not the same pipeline scaled down.
- **Team roles** are on the slide so it's clear who did what — the rubric checks
  this and checks that every member speaks.

## 3. The numbers on this slide

None new. It's a recap — do **not** introduce a number here that wasn't on an
earlier slide.

## 4. Bullets + footer lines, exactly as they go on the slide

```
• A new-city model is mis-tuned, not incapable
• The reusable win: label-free group-by-group alignment — 0.36 → 0.65
• Careful use of the data beat every fancier model
```
```
New here: per-group alignment · few vs many labels need different machinery

A — features · B — the line-up · C — evaluation · D — analysis
```

## 5. What to actually say (~35 s, split across the team)

> "Three things to take away.
>
> One — a model in a new city is mis-tuned, not incapable. Lining the cities up
> recovers most of the gap with zero labels.
>
> Two — that label-free line-up, 0.36 to 0.65, is the result we'd stand behind
> anywhere. Labels help too, but slide five showed some of that is proximity.
>
> Three — careful use of the data beat every fancier model we tried.
>
> Our original contribution is the group-by-group line-up driven by the model's
> own guesses, and the framing that few and many labels need different machinery.
>
> Roles are on the slide. Happy to take questions."

## 6. If someone asks

- **"What's the single most reusable result?"** The no-label line-up: 0.36 → 0.65
  with zero Amsterdam labels. It has no caveats.
- **"Would this work for city X?"** That's exactly the third-city test we'd run
  next. The mechanism — align, then add labels — isn't Madrid-specific.
- **"If you had one more week?"** Spatial-block-disjoint evaluation as the
  default, and a third city.

## 7. Words to avoid

| don't say | say instead |
|---|---|
| calibration (unexplained) | the model's number-ranges being off |
| generalisation gap | how much worse it does in a new city |
| few-shot regime | the case where you only have a handful of labels |
