# Team Brief — what we did, in plain words

Everything below is written for someone who wasn't in the weeds. No jargon that
isn't spelled out. **Read this first**, then the slides.

> **A note on the word "band"** — it has two meanings in this project. A
> **spectral band** is one of the 6 colours the satellite measures (Blue, Green,
> Red, and three infrared ones). An **age class** is one of the 4 groups we
> predict (1 = oldest … 4 = newest). This brief says "age class" for the second;
> some older docs say "band" — same thing.

---

## 0. Where the code is

Branch: **`israel-transfer-pipeline`** on
<https://github.com/ubayd-hattas/Data_Science_Hackathon>

```bash
git clone https://github.com/ubayd-hattas/Data_Science_Hackathon.git
cd Data_Science_Hackathon
git checkout israel-transfer-pipeline
pip install -r requirements.txt
```

The data files aren't in git (too big). Put `madrid_train.parquet` and
`amsterdam_data.parquet` into `data/`, then run `notebooks/5-Transfer.ipynb`
top to bottom — it reproduces every number in this brief with a fixed seed.

| where | what's in it |
|---|---|
| `notebooks/1–4` | the notebooks the organisers gave us, untouched |
| `notebooks/5-Transfer.ipynb` | **our pipeline** — one seeded run, produces the results table + plot |
| `src/` | the reusable code (loading, features, alignment, evaluation) |
| `scripts/` | every experiment we ran, re-runnable |
| `results/` | the F1 table and the curve |
| `deliverables/` | the slide deck |
| `docs/` | this brief plus the deeper write-ups |

---

## 1. The task, restated simply

> Look at a satellite's view of a small patch of city (30 m × 30 m). Guess how
> old the buildings on it are — not the exact year, one of **four age classes**.

The catch that makes it a research problem rather than a homework exercise:

- We only have good building-age records for **two cities: Madrid and Amsterdam**.
- The real goal is a method that works for **any city** with only a tiny amount
  of local checking.
- So the challenge is: **train on Madrid, then make it work on Amsterdam** using
  as few Amsterdam examples as possible (5, 25, 50, 100 or 200 per age class).

The score is **macro-F1**: 0 = useless, 1 = perfect, and all four age classes
count equally, so you can't win by only getting the common ones right.

---

## 2. What the satellite actually gives us

Not a photo. It measures **how much light bounces off the ground in six
"colours"** (three we can see, three infrared), **once a year for about 40
years**. Building materials and ageing — weathering, grime, new roofs — change
that bounce pattern over time.

We squash each patch's 40-year × 6-colour history into **a list of summary
numbers** ("features"): averages, how much things wobble year to year, the size
and timing of the biggest sudden change, and what the neighbouring patches look
like. Those numbers are what the model actually sees. 108 of them per patch.

---

## 3. What we built

A three-part pipeline:

1. **Train a model on Madrid** — a Random Forest, basically a committee of 500
   decision trees that vote.
2. **Zero labels in Amsterdam?** First reshape the Madrid data so its numbers
   line up with Amsterdam's, *then* train, then run it.
3. **A few labels in Amsterdam?** Clean up the Amsterdam features, train a small
   local model on the handful of labels we have, and let it vote *together* with
   the Madrid model.

Everything that looks at Amsterdam uses only the **unlabelled** data (which we're
allowed) plus the small set of labels we're given. Nothing else. **This matters:
the rubric disqualifies teams for "data leakage".**

---

## 4. The findings — what worked, biggest first

Every real improvement was **a fix to how existing information was used**, not a
fancier model.

| rank | what we did | plain explanation | effect |
|:---:|---|---|---|
| 1 | **Dialling the feature-untangling by label count** | Untangling redundant features needs data. With only 5 labels it backfires, so we turn it down when labels are scarce and up when they're plentiful. | **+0.15 / +0.06 / +0.03** at 5 / 25 / 50 labels |
| 2 | **Class-conditional CORAL** | Instead of reshaping the whole Madrid cloud to Amsterdam once, reshape it *age-class by age-class*, using the model's own first guesses on the unlabelled Amsterdam data. Repeat twice. | zero-shot **0.58 → 0.65**; also **+0.02** at the 5-label point, which nothing else moved |
| 3 | **CORAL alignment** (the base version) | Madrid's numbers and Amsterdam's sit in different ranges. Reshape Madrid's to match Amsterdam's overall shape *before* training, so the model's rules land in the right place. | zero-shot **0.36 → 0.58** |
| 4 | **Feature "untangling" (whitening)** | Lots of our features secretly say the same thing (brightness in blue ≈ green ≈ red). That triple-counts one idea. Untangling makes each idea count once. | few-shot **+0.06** at mid budgets |
| 5 | **Two models voting together** | Blend the local Amsterdam model's guess with the Madrid model's — lean on Madrid when labels are few, lean local as they grow. | **+0.04** at 5 labels, smaller elsewhere |
| 6 | **New features + an overnight 500-setting search** | Adding "when did construction happen" + "what's nearby", plus a bigger forest to use them. A search over one half of Amsterdam, checked once on the other half. | **+0.02–0.03** across the curve |
| 7 | **Neighbourhood smoothing of the predictions** | After the model predicts, blend each patch's probabilities with its 8 map-neighbours (closer neighbours weighted more). City blocks share a build era, so a lone odd prediction is usually a mistake. | **+0.02** across the curve |

### Things we tried that did NOT work

Worth presenting — the rubric explicitly rewards explaining *why* a reasonable
idea failed.

| what we tried | why it seemed sensible | why it failed |
|---|---|---|
| A small neural network | the challenge notes suggested one | our features are already easy to separate — nothing for it to learn; it just added noise |
| Telling the model the classes are ordered (1 near 2, far from 4) | it's true, and won a similar public competition | it makes wrong answers *smaller*, but the score only counts right vs wrong |
| Correcting for Amsterdam having more old buildings | the imbalance is real and measured | the trick needs the model's confidence to be trustworthy across cities — it isn't, so it amplified the error |
| "Self-training" — let the model label the unlabelled data and learn from that | standard semi-supervised idea | at low label counts the model is ~35 % wrong, so it just teaches itself its own mistakes |
| Gradient boosting instead of Random Forest | usually a bit better on tables | tied in-city, and fell apart with only 5 labels |
| Richer neighbourhood features (several map scales + spread) | more context should help | +0.001 — the single 8-neighbour average already captures it |
| Synthetic support examples ("mixup") | helps neural nets in low-data settings | slightly negative — random forests don't gain from blended points |
| Anchoring the smoothing to known labels | known labels *are* ours to use | it worked (+0.006 more) but the gain was "same building next door", not skill — a teammate's audit caught it, so we dropped it (see §7a) |

**One-liner for the slide:** *every attempt to be cleverer than the data lost;
every attempt to use the data more carefully won.*

---

## 5. The two new feature groups (what they are, with examples)

**Change-point features** — for each colour: how big the single biggest
year-to-year jump was, **when** it happened, which direction, and the slow drift
over 40 years.

> *Example.* A house built in 1955 has a flat, gently drifting infrared line —
> biggest jump ≈ 0.02, and *when* it happened is meaningless noise. An office
> built in 2016 goes car park → building site → glass roof: infrared swings by
> 0.20 in one year, and "when" ≈ 0.80 (near the end of the record). Flats built
> in 1994 swing the same way but "when" ≈ 0.25. So jump *size* separates old from
> new, and jump *timing* separates class 3 from class 4.

**Why it travels between cities:** construction looks the same everywhere —
bare ground → site → roof is physics, not a Madrid quirk.

**Neighbourhood features** — for each patch, average the features of its 8
nearest patches on the map and add those as extra columns.

> *Example.* One patch's own readings are borderline between class 2 and class 3
> and the model can't decide. But its 8 neighbours are unambiguously class 2 — a
> uniform 1970s estate. The neighbourhood average tips it to class 2. It's like
> reading a smudged word by looking at the rest of the sentence.

**Why it works:** cities are built in waves — whole blocks share a construction
era. **The catch:** Madrid's street layout is nothing like Amsterdam's, so these
features *hurt* the zero-label case (a model leaning on "what's around me" has
memorised Madrid's geography). They only pay off once the alignment step or some
local labels are in place.

---

## 6. The numbers (final, full data, tuned pipeline)

| Amsterdam labels per age class | our macro-F1 | starting baseline |
|---:|---:|---:|
| 0 — zero-shot, class-conditional CORAL | **0.65** | 0.43 |
| 5 per class | **0.66** ± 0.003 | 0.42 |
| 25 per class | **0.68** ± 0.006 | 0.55 |
| 50 per class | **0.70** ± 0.009 | 0.61 |
| 100 per class | **0.72** ± 0.007 | 0.64 |
| 200 per class | **0.74** ± 0.005 | 0.67 |
| *Madrid, tested on itself* (the ceiling) | *0.66 ± 0.004* | — |

**The headline (on the organiser's random split):** by about **100 labelled
buildings per age class**, the reshaped Madrid model matches a home-city model —
0.72 vs 0.66. The curve climbs fast to ~50 labels, then flattens. And the
**zero-labels** number jumped from 0.58 to 0.65 once the alignment was done one
age class at a time instead of all at once.

**The caveat (§7a):** most of that few-shot climb is the random support labels
sitting next to what we score. Hold the labels a map-tile away and the gain from
50–200 labels drops to ~+0.01. The **zero-shot 0.65** carries no such caveat —
it uses no labels — so treat *that* as the transferable number and the few-shot
curve as an upper bound.

*How we know the tuning is real:* it used a **50/50 split of Amsterdam** —
settings chosen on one half, scored once on the other half the search never saw.
The improvement survived that test before we ran the final numbers.

---

## 7. The honest limitations to lead with

### 7a. Most of the few-shot gain is spatial proximity — we measured it

The few-shot support set is drawn at random, and a teammate's audit found that
**~80 % of the support pixels we draw sit right next to a query pixel** on the
map (a 30 m pixel and its neighbour are often the same building or block). This
is a property of the organiser's random-sampling rule, not a bug — we never
touch query labels. But because satellite pixels close together look alike, it
flatters the few-shot curve.

**We then re-ran the whole pipeline to see how much.** `scripts/run_spatial_block.py`
draws the support labels only from map tiles a full tile away from whatever is
being scored — no support pixel anywhere near a query pixel — and scores the
identical model. Result:

| labels per class | gain from the labels, random split | gain, labels held a tile away |
|---:|---:|---:|
| 25  | +0.03 | +0.01 |
| 50  | +0.05 | ≈ 0   |
| 100 | +0.07 | +0.01 |
| 200 | +0.09 | +0.01 |

So **roughly half the few-shot improvement on the organiser's split is the
labels sitting next to what we score**, not the model generalising to new
ground. The **zero-shot** number (0.36 → 0.65) uses *no* labels, so it has no
support/query adjacency and is completely unaffected — **that is our solid,
transferable result.** (Per-tile F1 is noisy, ±0.07–0.09, so the direction is
firm but the exact deltas are soft. Detail:
`docs/AMSTERDAM_SPATIAL_ADJACENCY_AUDIT.md`,
`results/spatial_block_eval_gap{1,2}.json`.)

We also keep the neighbourhood smoothing on **predictions only** — no known
label ever leaks into a neighbour's answer.

### 7b. The two oldest age classes

Both classes 1 and 2 are buildings from *before 1984*, when the satellite record
starts, so there is no construction event to see — only a settled surface. The
*raw baseline* can't tell them apart, which is where the "pre-war stock is
hardest to date" literature applies. **Our final pipeline does separate them**,
though: per-class F1 is **class 1 ≈ 0.78 (our strongest class)**, with classes 2,
3 and 4 all near **0.70** — no single unsolvable pair. We corrected an earlier
assumption here; say the corrected version.

---

## 8. Who presents what (suggestion — adjust freely)

| slide(s) | who | what they cover |
|---|---|---|
| 1–2 cover + contents | whoever opens | the one-line pitch, then the agenda |
| 3 the problem | member A | the task, why a satellite can sense building age, the cross-city challenge |
| 4 data → features | member A or B | 40 years of light → a list of numbers; one row per patch |
| 5 approach diagram | member B | the three-part pipeline; the leakage discipline |
| 6 zero-shot / CORAL | member B | what "reshaping Madrid to match Amsterdam" means; the 0.36 → 0.58 jump |
| 7 few-shot mechanism | member C | untangling features, dialling it by budget, two models voting |
| 8–9 results table + curve | member C or D | read the numbers **with their error bars**; the "meets the home-city score" point |
| 10 what didn't work | member D | the five dead ends, one line each; the "simple beat clever" theme |
| 11 what we audited | member D | ~half the few-shot gain is spatial proximity (we measured it); zero-shot is unaffected; class 1 is actually our strongest |
| 12 team | all | who did what |

The rubric checks that **every member speaks** and that it's **clear who did
what** — so split it and say so out loud.

---

## 9. What we need from each of you

1. **Make at least one real commit** to the branch. `0geder` has done this
   (the spatial-adjacency audit). Everyone else still needs one — fix a typo
   here, add your name to the slides, adjust a plot colour. The judges look at
   `git log`. This is the cheapest points on the whole rubric.
2. **Fill in your name** on slides 1 and 12, and what you owned.
3. **Rehearse your slides out loud** at least once, together.
4. Someone should **ask the organisers**: do they want four or five Amsterdam F1
   scores? (Notebook 1 says "four" but lists five sample sizes.) The word cap is
   settled at 500 (rubric). A wrong guess on the F1 count is a point deduction.

---

## 10. Two-minute version — if someone asks "so what did you find?"

1. A model trained in one city isn't *stupid* in another, it's **miscalibrated**
   — its rules are right, its number-ranges are off. A one-step reshape (CORAL)
   fixes most of that for free, with zero local labels.
2. With **~100 checked buildings per age class**, that reshaped model works as
   well in the new city as any model works at home.
3. Every win came from **using the data more carefully** — aligning number
   ranges, removing double-counted features, blending two models. Every attempt
   at a fancier model lost.
4. We **stress-tested our own few-shot result**: with the local labels held a
   map-tile away from what we score, about half the few-shot gain disappears —
   it was spatial proximity. The zero-shot reshape (0.36 → 0.65) uses no labels
   and stands unaffected. That's the honest headline.
