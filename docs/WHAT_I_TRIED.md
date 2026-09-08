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
