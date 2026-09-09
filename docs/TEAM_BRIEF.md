# Team Brief — what we did, in plain words


> **A note on the word "band"**: it has two meanings here. A **spectral band** is
> one of the 6 colours the satellite measures (Blue, Green, Red, and three
> infrared). An **age class** is one of the 4 groups we predict (1 oldest ... 4
> newest). This brief uses "age class" for the second; older docs sometimes say
> "band" — same thing.
Share this with the team. It explains the project, the findings, and who can
present what. No jargon that isn't spelled out.

---

## 1. The task, restated simply

> Look at a satellite's view of a small patch of city (30 m × 30 m). Guess how
> old the buildings on it are — not the exact year, one of **four age classes**.

The catch that makes it a research problem, not a homework exercise:

- We only have good building-age records for **two cities: Madrid and Amsterdam**.
- The real goal is a method that works for **any city** with only a tiny amount
  of local checking.
- So the challenge is: **train on Madrid, then make it work on Amsterdam** with
  as few Amsterdam examples as possible (5, 25, 50, 100 or 200 per age class).

The score is **macro-F1**: 0 = useless, 1 = perfect, and all four age classes
count equally so you can't win by only getting the common ones right.

---

## 2. What the satellite actually gives us

Not a photo. It measures **how much light bounces off the ground in six
"colours"** (three we can see, three infrared), **once a year for about 40
years**. Different building materials and ageing (weathering, grime, new roofs)
change that bounce pattern over time.

We turn each patch's 40-year × 6-colour history into **a list of summary numbers**
("features") — averages, how much things wobble year to year, the size and timing
of the biggest sudden change, and what the neighbouring patches look like. Those
numbers are what the model sees.

---

## 3. What we built

A three-part pipeline:

1. **Train a model on Madrid** (a Random Forest — basically a big committee of
   decision trees that vote).
2. **Zero labels in Amsterdam?** First reshape the Madrid data so its numbers
   line up with Amsterdam's, *then* train. Then just run it.
3. **A few labels in Amsterdam?** Clean up the Amsterdam features, train a small
   local model on the handful of labels, and let it vote *together* with the
   Madrid model.

Everything that looks at Amsterdam uses only the **unlabelled** data (which we're
allowed) plus the small set of labels we're given — never anything we shouldn't
have. (This matters: the rubric disqualifies teams for "data leakage".)

---

## 4. The findings — what worked and why

Every real improvement was **a fix to how existing information was used**, not a
fancier model. Simple beat clever every time on this dataset.

| what we did | plain explanation | effect on the score |
|---|---|---|
| **CORAL alignment** | Madrid's numbers and Amsterdam's sit in different ranges. Reshape Madrid's to match Amsterdam's overall shape before training, so the model's rules land in the right place. | zero-shot **0.45 → 0.55** |
| **Feature "untangling" (whitening)** | Lots of our features secretly say the same thing (brightness in blue ≈ green ≈ red). That triple-counts one idea. Untangling makes each idea count once. | few-shot **+0.06** at mid budgets |
| **Dialling the untangling by budget** | Untangling needs enough data. With only 5 labels it backfires, so we turn it down when labels are scarce and up when they're plenty. | **+0.15 / +0.06 / +0.03** at 5 / 25 / 50 labels |
| **Two models voting together** | Blend the local Amsterdam model's guess with the Madrid model's. When labels are very few, lean on Madrid; as labels grow, lean local. | **+0.04** at 5 labels, smaller elsewhere |
| **Change-point + neighbourhood features** | Add "when did the big change happen?" and "what's around this patch?". An overnight search found these help once the other settings are tuned around them. | few-shot **+0.02–0.03** across the board |

### Things we tried that did NOT work (worth presenting — it shows rigour)

| what we tried | why it seemed sensible | why it failed |
|---|---|---|
| A small neural network | the challenge notes suggested one | our features are already easy to separate — nothing for it to learn; it just added noise |
| Telling the model the classes are ordered (1 near 2, far from 4) | it's true, and won a similar competition | it makes wrong answers *smaller*, but the score only counts right vs wrong |
| Correcting for Amsterdam having more old buildings | the imbalance is real | the trick needs the model's confidence to be trustworthy across cities — it isn't, so it made things worse |
| "Self-training" — let the model label the unlabelled data and learn from that | standard semi-supervised idea | at low label counts the model is ~35% wrong, so it just teaches itself its own mistakes |
| Gradient boosting instead of Random Forest | usually a bit better on tables | tied in-city, and fell apart with only 5 labels |

**One-liner for the slide:** *every attempt to be cleverer than the data lost;
every attempt to use the data more carefully won.*

---

## 5. The numbers

*(final full-data numbers land when the last run finishes; these are the tuned
held-out results — expect the final table within ±0.01)*

| how many Amsterdam labels per class | our macro-F1 | starting baseline |
|---:|---:|---:|
| 0 (zero-shot) | ~0.55 | 0.43 |
| 5 per class | ~0.61 | 0.42 |
| 25 per class | ~0.66 | 0.55 |
| 50 per class | ~0.69 | 0.61 |
| 100 per class | ~0.70 | 0.64 |
| 200 per class | ~0.71 | 0.67 |
| **Madrid, tested on itself** (the ceiling) | **0.63** | — |

**The headline:** with about **100 labelled buildings per class**, our
Madrid-trained model does **as well on Amsterdam as a model does on its own home
city**. The curve climbs fast up to ~50 labels, then flattens — more labels
barely help after that.

---

## 6. The honest limitation to state up front

**Age classes 1 and 2 are the hardest and always will be with this data.** Both are
buildings from *before 1984*, which is when the satellite record starts. For
newer buildings we can literally see the construction happen (bare ground →
building site → finished roof). For pre-1984 buildings there's no such event —
just a settled surface — so telling "old" from "slightly less old" is genuinely
close to impossible here. Published research on pre-war buildings says the same
thing. Most of our remaining errors are classes 1↔2.

---

## 7. Who presents what (suggestion)

| slide(s) | who | what they say |
|---|---|---|
| Problem + data | member A | the task, why a satellite can sense building age, the cross-city challenge |
| Features | member A or B | how we turn 40 years of light into a list of numbers; one row per patch |
| The approach + CORAL | member B | the three-part pipeline; what "reshaping Madrid to match Amsterdam" means; the 0.45 → 0.55 jump |
| Few-shot mechanism | member C | untangling features, dialling it by budget, two models voting |
| Results table + curve | member C or D | read the numbers *with their error bars*; the "flattens at the home-city score" point |
| What didn't work | member D | the five dead ends, one line each; the "simple beat clever" theme |
| Limitations + next steps | member D | classes 1↔2 are a data limit; next would be more feature engineering |

Rubric checks that **everyone speaks** and that it's **clear who did what** —
so split it and say so.

---

## 8. Two-minute version (if someone asks "so what did you actually find?")

1. A model trained in one city is *miscalibrated* in another, not stupid —
   its rules are right, its number-ranges are off. A one-step reshape
   ("CORAL") fixes most of that for free.
2. With ~100 checked buildings per age class, that reshaped model works as well
   in the new city as any model works at home.
3. The wins all came from **using the data more carefully** (aligning number
   ranges, removing double-counted features, blending two models). Every
   attempt at a fancier model lost.
4. The oldest two age classes can't be cleanly separated from 30 m satellite data
   — there's no construction event to see — and that's where our errors are.
