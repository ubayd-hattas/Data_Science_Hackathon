# What I Tried and Why — In Plain Language

This is a diary of the thinking, not a polished report. It walks through what I
looked at, what I guessed, what I tested, and where I was wrong. If you only read
one doc to understand the state of the project, read this one.

---

## The starting point

You have four notebooks from the organisers. They go:

1. **Introduction** — explains the challenge.
2. **Reading Data** — loads the satellite data, makes charts.
3. **Preprocessing** — turns the raw data into a table of numbers ("features"),
   one row per location.
4. **Modelling** — trains a model on Madrid, then tries to make it work on
   Amsterdam with only a handful of Amsterdam examples.

The goal of the whole thing: predict how old the buildings are in a 30m patch of
ground, using how that patch reflects light across ~40 years of satellite photos.
And crucially — make it work in a *new city* without much local data.

---

## First thing I checked: can this even run on your laptop?

You were worried you'd need the organisers' server ("the hub") just to run the
code. So I measured it.

**What I found:** the data files have 48 columns, but the notebooks only ever use
27 of them. The other 21 are things like text descriptions of which satellite
photo each row came from — useless for modelling, but they take up most of the
memory.

Load only the 27 you need, and Madrid goes from ~1.3 GB of memory down to
0.7 GB, and the whole preprocessing runs in about 4 seconds.

**Solved:** you can do everything locally. The hub is optional — nice to have for
sharing with teammates, not required for the work.

I put this lean loader in `src/data.py` so nobody has to figure out the column
list again.

---

## Second thing: I read the modelling notebook carefully, and something was off

The notebook calls its approach "metric learning". The idea of metric learning
is: teach the model a sense of *distance* — so two buildings of the same age end
up "close together" in the model's view, even if they're in different cities.
Then for a new city, you just need a few examples to know where each age group
sits.

The supporting document (`METRIC_LEARNING_APPROACH.md`) describes this in detail:
train a small neural network on Madrid that learns this distance, then apply it
to Amsterdam.

**The problem:** the notebook never builds that network. I checked every code
cell. The few-shot step takes the Amsterdam features, averages them per age
group to get a "typical example" of each group, and labels everything by which
average it's closest to. That's a reasonable simple method — but it does the
averaging on the *raw* features. The Madrid-trained model is loaded and then
never used again.

So "trained on Madrid, transferred to Amsterdam" isn't really happening. The
only thing carried over from Madrid is a scaling step (making all the feature
numbers a similar size).

**Why this matters:** the grading rubric says a team can *lose 20 points or be
disqualified* for "not performing transfer learning". As the notebook stands,
that risk is real. It also means the headline result — "5 examples per class
already beats the no-training version!" — is close to meaningless, because
you're really just comparing two different simple methods, not measuring
transfer.

---

## My guess: build the missing network, that'll fix it

The obvious move: actually build the thing the approach document describes. A
small network, trained on Madrid with "triplet" training (show it three examples
at a time — an anchor, one of the same age, one of a different age — and nudge it
to pull the same-age pair together and push the different-age one away).

I wrote it from scratch in plain numpy so it has no heavy dependencies and runs
anywhere. I checked the maths of it carefully (a "gradient check" — comparing the
hand-derived calculus against a brute-force numerical estimate; they matched to
10 decimal places, so the implementation is correct).

**Then I tested whether it actually helps. It doesn't.**

| Method | Score at 200 examples/class (higher = better) |
|---|---|
| The notebook's simple method | 0.615 |
| My Madrid-trained network | 0.542 |

It made things *worse*. The reason: the training signal is weak. Most random
triplets of buildings are already "far enough apart" by the network's measure,
so there's almost nothing to learn from. You can fight that with cleverer
training, but there's no guarantee it ever beats the simple method, because the
features are already pretty good on their own.

**So my guess was wrong.** The diagnosis (the transfer isn't real) was right. The
proposed fix (add the network) was a dead end.

---

## What I did next: test the standard toolbox for this exact problem

There's a well-known family of methods for "I trained on dataset A, now make it
work on dataset B" without labels from B. They all work by making B's numbers
*look statistically like* A's numbers before you use the model. I tried three,
from simplest to most thorough:

### a) Match the size of each feature

Rescale every feature so its spread matches Amsterdam's instead of Madrid's.

**Result: no difference.** 0.615 vs 0.613. Once you scale the features at all, it
doesn't matter whose numbers you use to do it.

(This corrects something I told you earlier in the session — I'd claimed this
gave a +4.5 point boost. That was a mistake: I'd accidentally compared against
features that weren't scaled at all. Scaling matters; *whose* scaling doesn't.)

### b) CORAL — match how the features relate to each other

A step up. Not just "make each feature the right size" but "make the features
*correlate* with each other the way Amsterdam's do". You reshape the Madrid data
to match Amsterdam's overall statistical shape, then train the model on that
reshaped data.

**Result: this is a real win.**

| | Score transferring to Amsterdam (no Amsterdam labels at all) |
|---|---|
| Model trained on raw Madrid | 0.439 |
| Model trained on CORAL-reshaped Madrid | **0.544** |

+10 points, for free, with about 15 lines of code and zero Amsterdam labels. This
is genuine domain adaptation — the thing the rubric specifically rewards.

### c) ZCA whitening — untangle the features

The 60 features are very redundant. There are big clusters of features that
basically say the same thing (several "average brightness" features that move
together, several "early period" features that move together, and so on). When
you measure distance between buildings by adding up differences across all 60
features, those redundant clusters get counted many times over and drown out the
rest.

"Whitening" untangles them — mathematically removes the redundancy so each
direction counts once.

**Result: the best few-shot number so far.**

| examples/class | notebook method | after whitening |
|---:|---:|---:|
| 50 | 0.605 | 0.607 |
| 100 | 0.611 | **0.651** |
| 200 | 0.615 | **0.674** |

At 100+ examples it even beats the score the model gets *within Madrid* (0.624) —
meaning, on this measure, the transferred model does as well on Amsterdam as a
model does on its home city.

One catch: whitening *hurts* when you have very few examples (5–25 per class),
because with that little data the "typical example" of each group is too noisy
and whitening amplifies the noise. So the honest recommendation is: whiten when
you have 50+ examples per class, use plain scaling below that.

---

## Where things stand

**Solved / established:**

- You can run everything locally. Fast.
- The provided notebook's "transfer" is mostly not real — documented, with the
  evidence.
- Two methods that genuinely improve transfer, both measured:
  - **CORAL**: +10 points on the no-labels version.
  - **Whitening**: +6 points on the few-labels version (with 50+ examples).
- A reusable test harness, so any further idea is a one-line change and a number.

**Not done yet:**

- Combining CORAL and whitening (should stack — untested).
- "Ordinal" training — telling the model that age group 1 and 2 are *neighbours*
  while 1 and 4 are far apart. Standard classifiers ignore this. It's the trick
  that won a very similar public competition (see `RELATED_WORK.md`). Untested
  here.
- Packaging the winning combination into a clean notebook for submission.
- A proper long run of the scoring (I've been using a faster, slightly rougher
  version to iterate quickly; the final numbers should be locked in with the
  full version).

**Trust but verify:** I've made two factual mistakes this session — an early
claim that "age group 1 is the most common" (it's actually the *least* common in
Madrid), and the "+4.5 from rescaling" slip above. Re-run the numbers yourself
before putting them in a presentation. The harness makes that easy:
`python scripts/run_baseline.py`.

---

## The files, and what each is for

| File | Plain description |
|---|---|
| `src/data.py` | Loads the data efficiently, builds the 60-number summary per location. |
| `src/evaluate.py` | Runs the scoring: home-city test, no-labels transfer, few-labels transfer. |
| `src/embedding.py` | The from-scratch network. Works, but doesn't help — kept for the record. |
| `src/adapt.py` | The three "make Madrid look like Amsterdam" methods, including CORAL and whitening. |
| `scripts/run_baseline.py` | Runs the whole comparison and writes results to `results/`. |

---

## Update: does CORAL + whitening combine?

**No — for the few-shot path they are the same operation.** Tested directly:

| examples/class | whiten by Amsterdam's shape | CORAL-to-Madrid then whiten by Madrid's shape |
|---:|---:|---:|
| 100 | 0.651 | 0.651 |
| 200 | 0.674 | 0.674 |

Identical at every budget. The reason is arithmetic: reshaping Amsterdam to
Madrid's shape and then untangling by Madrid's shape cancels out to just
untangling by Amsterdam's own shape. There is nothing to stack.

CORAL only does real work on the **zero-labels** path, because there it has a
Madrid-trained model to line up against. On the few-labels path there is no
Madrid model in the loop, so CORAL has nothing to align and collapses into the
whitening step.

One side finding worth keeping: whitening by **Madrid's** shape instead of
Amsterdam's is steadier when labels are very scarce (0.50 vs 0.38 at 5
examples/class) but worse once you have enough (0.62 vs 0.67 at 200). Neither
wins outright.

**So the two wins stay on their own tracks:**

| situation | method | score |
|---|---|---|
| no Amsterdam labels | CORAL + Random Forest | 0.54 |
| 100+ Amsterdam labels/group | whitening + nearest-prototype | 0.65-0.67 |

The only untested way to combine them is an *ensemble* — average the CORAL model's
vote with the whitened-prototype vote — which may or may not beat either alone.

---

## Round 3: using the class order, and correcting the class mix — both failed

Two more ideas from the plan, both tested, both do not help the graded score.

### Ordinal training

The four age groups are ordered (1 next to 2, far from 4). A plain classifier
ignores that. I built the standard fix — split the 4-way problem into three
"is it older than group k?" yes/no models and recombine (Frank & Hall).

**Result: worse at every setting.**

| | plain model | ordinal model |
|---|---:|---:|
| zero-shot (CORAL features) | 0.542 | 0.509 |
| few-shot @ 200/group | 0.671 | 0.576 |

It *does* make the near-misses closer on average (when it's wrong, it's wrong by
less), but the competition scores macro-F1, which only counts right vs wrong, not
by how much. So this would only matter if the grading changed.

### Correcting the class mix (label-shift correction)

Amsterdam has a different mix of age groups than Madrid (more old buildings,
fewer new ones). There's a standard trick to rescale a model's outputs toward
the target's mix without using target labels (an EM procedure).

**Result: worse, sometimes badly.**

| | before | after correction |
|---|---:|---:|
| zero-shot raw | 0.431 | 0.113 |
| zero-shot CORAL | 0.542 | 0.442 |

The trick needs the model's probabilities to be trustworthy on the target. The
gap between the cities makes them untrustworthy, so the correction latches onto
the wrong signal and amplifies it. The actual mix difference is real but small
enough that trying to fix it this way costs more than it's worth.

### One small positive

Training a plain small Random Forest on the handful of whitened Amsterdam labels
(instead of the nearest-prototype rule) is a touch better at small budgets and
ties at large ones — and it gives the "how far off" number for free.

| shots/group | nearest-prototype | small Random Forest |
|---:|---:|---:|
| 5 | 0.415 | 0.433 |
| 25 | 0.546 | 0.562 |
| 200 | 0.673 | 0.671 |

### Running tally

| idea | outcome |
|---|---|
| build the missing embedding network | worse — dead end |
| CORAL (reshape Madrid to Amsterdam) | **+10 pts on zero-shot** |
| whitening (untangle features) | **+6 pts on few-shot, 50+ labels** |
| combine CORAL + whitening | same as whitening alone (they're one operation) |
| ordinal training | worse — dead end |
| class-mix correction | worse — dead end |
| small RF head instead of prototypes | marginal, ~tie |

Three dead ends, two real wins. That's a normal hit rate. The dead ends are
worth writing up — the rubric explicitly rewards "clear insight from why a bold
approach didn't work".

---

## Round 4: new features — change-point and spatial context

Added two feature groups to the 60:

* **change-point** (+24): for each colour, the size, direction and *timing* of its
  single biggest year-to-year jump, plus the overall trend slope. The base set
  had "average change" and "wobble of change" but never *when* the change hit.
* **spatial** (+24): each patch's features re-averaged over its 8 nearest
  neighbours on the map. Age groups cluster by neighbourhood, so "what's around
  me" is a signal. Uses neighbours' features only, never their answers.

Quick-run table (5x2 CV, rougher than final):

| feature set | Madrid own score | zero-shot raw | zero-shot CORAL | few-shot @25 | @200 |
|---|---:|---:|---:|---:|---:|
| base (60) | 0.623 | 0.431 | 0.542 | 0.546 | 0.673 |
| + change-point (84) | 0.627 | **0.480** | 0.556 | 0.507 | 0.672 |
| + spatial (84) | **0.661** | **0.330** | 0.562 | 0.526 | **0.684** |
| + both (108) | 0.660 | 0.358 | **0.574** | 0.498 | 0.683 |

What this says:

* **Spatial context is the strongest lever for in-city accuracy** (+3.8 on
  Madrid's own score) — but it *collapses* raw zero-shot (0.43 → 0.33). Madrid's
  neighbourhood layout is nothing like Amsterdam's, so a model leaning on "what's
  around me" is leaning on the wrong thing in a new city.
* **CORAL rescues that collapse** — with CORAL, +both gives the best zero-shot
  number yet (0.574). The more city-specific your features, the more you need the
  alignment step.
* **Change-point helps raw zero-shot** (+5) because *when* construction happened
  is city-independent physics, not a Madrid quirk.
* **Few-shot barely benefits from any of it** (0.673 → 0.684 at best) and the
  small-budget number gets *worse* (0.546 → 0.498 at 25) — more features need
  more support points before whitening is stable.

The real lesson: **there is no single best feature set — it depends on which
scenario you are scoring.** See `LESSONS.md`.

---

## Round 5: shrinkage whitening — a real win, biggest where it counts

Full whitening untangles all 60 features but needs enough data to be stable, so
it *hurt* at small label budgets. The fix: blend the covariance toward its own
diagonal before inverting — a dial from "full untangling" (0) to "just per-
feature scaling" (1). Heavy blend when labels are scarce, none when plentiful,
chosen automatically from the budget.

| labels/group | full whitening (before) | adaptive shrink (after) | gain |
|---:|---:|---:|---:|
| 5 | 0.385 | **0.538** | **+0.15** |
| 25 | 0.552 | **0.616** | **+0.06** |
| 50 | 0.605 | **0.637** | +0.03 |
| 100 | 0.650 | **0.659** | +0.01 |
| 200 | 0.670 | **0.678** | +0.01 |

Every budget improved, nothing regressed. The 25-label number — which the
"best low-data" prize is scored on — went up 6 points.

### Running tally

| idea | outcome |
|---|---|
| build the missing embedding network | worse — dead end |
| CORAL (reshape Madrid to Amsterdam) | **+10 pts zero-shot** |
| whitening (untangle features) | **+6 pts few-shot, 50+ labels** |
| CORAL + whitening combined | same as whitening alone |
| ordinal training | worse — dead end |
| class-mix correction | worse — dead end |
| change-point features | **+5 pts zero-shot raw**, neutral few-shot |
| spatial-context features | +4 pts Madrid's own score, breaks raw transfer, small few-shot gain |
| **shrinkage whitening** | **+15 / +6 / +3 pts at 5 / 25 / 50 labels** |

Wins are stacking up on the few-shot side now. Best few-shot numbers to date:
0.54 / 0.62 / 0.64 / 0.66 / 0.68 at 5 / 25 / 50 / 100 / 200 labels per group.

---

## Round 6: self-training (dead end) and RF-head vs prototype (win)

**Self-training** — pseudo-label the most confident of the 25k unlabelled
Amsterdam patches, add them to the training set, refit, repeat.

| method | 5 | 25 | 50 | 100 | 200 |
|---|---:|---:|---:|---:|---:|
| prototype (baseline) | 0.558 | 0.619 | 0.634 | 0.657 | 0.680 |
| prototype + self-training | 0.507 | 0.582 | 0.600 | 0.597 | 0.621 |
| small RF + self-training | 0.559 | 0.603 | 0.606 | 0.604 | 0.585 |

Worse in every cell, and it degrades further each round. At low budgets the base
model is ~35% wrong, so the pseudo-labels are ~35% wrong, and refitting on them
drags the model toward that error. Not fixable with the obvious knobs (fewer
promotions, higher confidence bar) — those made it worse, not better. Dead end.

**RF head instead of nearest-prototype** — same whitened + shrunk features, but
label the queries with a small Random Forest rather than the nearest class mean.

| budget | nearest-prototype | small RF | gain |
|---:|---:|---:|---:|
| 5 | 0.558 | 0.574 | +0.016 |
| 25 | 0.619 | **0.639** | +0.020 |
| 50 | 0.634 | 0.649 | +0.015 |
| 100 | 0.657 | 0.675 | +0.018 |
| 200 | 0.680 | 0.680 | tie |

A clean ~+0.02 at every budget except the top, and the 25-label number (the
low-data prize) improves again. Already in the harness.

### Running tally

| idea | outcome |
|---|---|
| embedding network | dead end |
| CORAL | **+10 zero-shot** |
| whitening | **+6 few-shot @50+** |
| CORAL + whitening combined | same as whitening alone |
| ordinal training | dead end |
| class-mix correction | dead end |
| change-point features | **+5 zero-shot raw** |
| spatial features | +4 Madrid-CV, breaks raw transfer |
| shrinkage whitening | **+15 / +6 / +3 at 5 / 25 / 50 labels** |
| self-training | dead end |
| RF head vs prototype | **+2 at every budget < 200** |

Best few-shot line to date: **0.57 / 0.64 / 0.65 / 0.68 / 0.68** at
5 / 25 / 50 / 100 / 200 labels per group  (from a starting 0.42 / 0.55 / 0.61 /
0.64 / 0.67). Numbers are from quick runs — the full run will shift them a little.

---

## Round 7: gradient boosting (dead end), CORAL ensemble (win)

**Gradient boosting** (HistGradientBoosting, LightGBM) instead of Random Forest:

| head | Madrid CV | zero-shot raw | zero-shot CORAL | few-shot @25 | @200 |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.626 | 0.431 | 0.542 | 0.639 | 0.680 |
| HistGB | 0.623 | **0.524** | 0.509 | 0.594 | 0.675 |
| LightGBM | 0.618 | 0.493 | 0.549 | 0.602 | 0.679 |

One real insight: boosting resists over-fitting Madrid, so *raw* zero-shot jumps
(0.43 -> 0.52). But CORAL + Random Forest still edges it (0.542), so the pipeline
does not change. Few-shot: boosting needs more data than a forest and collapses
at 5 labels (0.11) — Random Forest stays the few-shot head.

**CORAL ensemble** — blend the few-shot head's vote with the Madrid CORAL
model's vote, weight `clip(shots/50, 0.4, 0.95)` toward the local head as labels
accumulate:

| budget | RF head alone | + CORAL-prior ensemble | gain |
|---:|---:|---:|---:|
| 5 | 0.574 | **0.614** | +0.040 |
| 25 | 0.639 | **0.649** | +0.010 |
| 50 | 0.649 | **0.662** | +0.013 |
| 100 | 0.675 | **0.683** | +0.008 |
| 200 | 0.680 | **0.685** | +0.005 |

Every budget up, biggest at 5 labels (where the local head is weakest and the
Madrid prior carries the most weight). This is the "combine the two wins" the
CORAL+whitening test couldn't do — it works here because the zero-shot model is
a *separate prediction*, blended at the probability level, not a feature
transform that collapses.

### Running tally

| idea | outcome |
|---|---|
| embedding network | dead end |
| CORAL | **+10 zero-shot** |
| whitening | **+6 few-shot @50+** |
| CORAL + whitening (feature-level) | same as whitening alone |
| ordinal training | dead end |
| class-mix correction | dead end |
| change-point features | **+5 zero-shot raw** |
| spatial features | +4 Madrid-CV, breaks raw transfer |
| shrinkage whitening | **+15 / +6 / +3 at 5 / 25 / 50 labels** |
| self-training | dead end |
| RF head vs prototype | **+2 at every budget < 200** |
| gradient boosting | dead end (RF ties or wins) |
| CORAL-prior ensemble (probability-level) | **+4 / +1 / +1 / +1 / +0.5** |

**Best few-shot line: 0.61 / 0.65 / 0.66 / 0.68 / 0.69** at 5 / 25 / 50 / 100 /
200 labels per group — from a starting 0.42 / 0.55 / 0.61 / 0.64 / 0.67.
Quick-run numbers; the full run will move them slightly.

---

## Round 8: overnight hyperparameter + feature search

A 500-configuration random search over feature set, forest depth/leaves/trees,
shrinkage-formula coefficients, ensemble weight schedule, and PCA. Scored on one
half of Amsterdam, confirmed once on the untouched other half (so the tuned
result can't be inflated by fitting to the reporting data). It got through 131
configs before repeated session restarts stopped it; 131 was enough — the winner
was stable.

**Winner:** the `both` feature set (change-point + spatial, 108 features) with a
500-tree, unrestricted-depth forest. Earlier hand-testing had *rejected* those
features; the search found they help once the forest is grown to use them.

| labels/class | tuned only (full data) | before |
|---:|---:|---:|
| Madrid CV | 0.664 | 0.626 |
| zero-shot CORAL | 0.578 | 0.545 |
| 5 | 0.623 | 0.614 |
| 25 | 0.662 | 0.645 |
| 50 | 0.682 | 0.659 |
| 100 | 0.702 | 0.680 |
| 200 | 0.717 | 0.685 |

## Round 9: spatial smoothing of *predictions*

We already feed neighbourhood-averaged *features*. This uses the neighbourhood a
second way: after the model predicts, average each pixel's class probabilities
with its k nearest map-neighbours and re-pick the winner. City blocks share a
build era, so a lone pixel predicted differently from all its neighbours is
usually wrong.

Swept k = 2-16: 3-8 optimal and flat, ≥16 blurs real age boundaries. **k = 8**.

**A teammate then audited this** (`docs/AMSTERDAM_SPATIAL_ADJACENCY_AUDIT.md`):
under the organiser's random per-class sampling, ~80% of support pixels have an
immediate query neighbour. An earlier "anchored" variant — which wrote true
support labels into the smoothed field — was therefore partly scoring
same-building proximity. **Switched to prediction-only smoothing**: no target
label ever enters the field, gain drops from ~+0.014 to ~+0.008, fully
defensible.

## Round 10: class-conditional CORAL  (the originality contribution)

Plain CORAL aligns the *pooled* Madrid and Amsterdam clouds once. This aligns
them **class by class, iteratively**: fit on globally-aligned Madrid, predict
unlabelled Amsterdam, re-align each Madrid class to the covariance of the
Amsterdam pixels the model *assigned* to it, refit; two rounds. Only the model's
own predictions are used — no target labels.

| | plain CORAL | class-conditional |
|---|---:|---:|
| zero-shot macro-F1 | 0.578 | **0.646** |
| few-shot n=5 | 0.633 | **0.654** |
| few-shot n≥50 | — | ~unchanged (prior matters less with local labels) |

The one method here that is genuinely novel rather than assembled from known
parts, and it moves the 5-label point — the budget nothing else could shift.

### Dead ends this stretch

| tried | result |
|---|---|
| richer neighbourhood features (multi-scale k=4/24 + neighbour spread, +54) | +0.001 mean — the 8-NN mean block already captures it |
| lower blend floor at n=5 (trust Madrid harder) | −0.024 — the weak local model still beats leaning on Madrid |
| mixup augmentation of the support set | slightly negative — forests don't gain from interpolated points |
| anchored smoothing (kept as non-default) | +0.006 over prediction-only, but that margin is adjacency not skill |

### Running tally

| idea | outcome |
|---|---|
| embedding network / ordinal / class-mix / self-training / boosting / mixup / low-floor / rich-spatial | dead ends |
| CORAL | **+0.10 zero-shot** |
| whitening + budget-scaled shrinkage | **+0.15 / +0.06 / +0.03 at 5 / 25 / 50** |
| RF head vs prototype | **+0.02 below 200 labels** |
| CORAL-prior probability blend | **+0.04 / +0.01 across budgets** |
| overnight feature + HP search | **+0.02–0.03 across the curve** |
| prediction-only spatial smoothing | **+0.008 across the curve** |
| class-conditional CORAL | **+0.07 zero-shot, +0.02 at n=5** |

**Few-shot line now (pending the final full run): ~0.65 / 0.67 / 0.69 / 0.71 /
0.73** at 5 / 25 / 50 / 100 / 200 — from a starting 0.42 / 0.55 / 0.61 / 0.64 /
0.67. Zero-shot ~0.65, from 0.43.
