# What We've Actually Learned

Not results — *understanding*. The results live in `WHAT_I_TRIED.md`; this is
the picture they add up to. Plain language.

---

## 1. The problem is really two problems wearing one label

The four age groups split cleanly down the middle of the satellite record:

| groups | built | during the 40-year satellite record? |
|---|---|---|
| **1 & 2** | before 1984 | no — the satellite only ever saw a finished building |
| **3 & 4** | 1984 onwards | yes — the satellite watched the construction happen |

These need *different kinds of evidence*:

- **Groups 3 & 4** are a *"when did the ground change?"* problem. There's a real
  event on camera — bare soil, then a building site, then a roof. You date it by
  finding the jump.
- **Groups 1 & 2** are a *"what does old look like?"* problem. No event to find —
  only slow weathering, grime, how settled the surface looks. Much subtler.

This is why **groups 1 and 2 are the ones the model confuses**. There's no
construction event to separate them, and "pre-1945 brick" vs "1945–1984 brick"
is a genuinely faint spectral difference. The published literature says the same
thing — pre-war building stock is the hardest to date anywhere, in every study.

**Implication:** most of the winnable signal is in groups 3 & 4. Effort spent
trying to crack 1-vs-2 with this data has a low ceiling.

---

## 2. "Transfer" isn't one thing — it's a spectrum, and each point wants a different pipeline

The challenge asks for scores at 0, 5, 25, 50, 100, 200 known Amsterdam examples
per group. We kept treating this as one problem. **It's a spectrum**, and what
wins at one end loses at the other:

| you have… | best approach found | why |
|---|---|---|
| **0 labels** | reshape Madrid to Amsterdam's shape (CORAL), then a Random Forest | no Amsterdam data to learn from, so all you can do is make Madrid *look* like Amsterdam first |
| **5–25 labels** | plain features, plain scaling, nearest-average | too little data to estimate anything fancy — extra features and untangling just add noise |
| **50+ labels** | untangle the features (whitening), nearest-average or a small Random Forest | now there's enough data for the untangling maths to be stable, and it pays off |

There is **no single feature set or method that's best everywhere**. A serious
submission reports a *curve*, and the curve is allowed to be produced by
*different machinery at different points*. That's not cheating — it's the honest
answer to "how much local data do you have?"

---

## 3. The features that help you at home can hurt you abroad

Clearest lesson of the whole session. We added **neighbourhood context** — each
patch also described by what its 8 map-neighbours look like.

| | Madrid predicting Madrid | Madrid predicting Amsterdam (no labels) |
|---|---:|---:|
| without neighbourhood context | 0.623 | 0.431 |
| with neighbourhood context | **0.661** | **0.330** |

It's the **best boost to home accuracy we found** and it **breaks transfer worst
of all.** Why: Madrid's neighbourhoods are laid out nothing like Amsterdam's
(density, block size, street pattern, how old buildings cluster). A model that
leans on "what's around me" has learned *Madrid's* geography, and Madrid's
geography is wrong in Amsterdam.

The general principle: **the more a feature encodes something specific to the
training city, the better it scores at home and the worse it travels.** Raw
brightness travels okay. Timing-of-construction travels well (it's physics).
Neighbourhood layout barely travels at all.

**And** — the alignment step (CORAL) matters *more* the more city-specific your
features are. With neighbourhood context added, CORAL went from a +10 rescue to
being essential — without it the model is worse than useless (0.33, below random
in places).

---

## 4. The graded metric quietly rejects some "improvements"

The score is **macro-F1** — every prediction is simply right or wrong, and all
four groups count equally. Two ideas died on this rock:

- **Ordinal training** (teaching the model that group 1 is near 2, far from 4)
  made the model's mistakes *smaller* — when wrong, wrong by less. But macro-F1
  doesn't care about "by how much", so it counted for nothing and the added
  machinery cost accuracy.
- **Fixing the class mix** (Amsterdam has more old buildings than Madrid) is
  theoretically sound, but the method needs the model's confidence scores to be
  trustworthy, and across a city gap they aren't. It amplified the error.

**Lesson:** before implementing a method, check it actually moves *the metric
you're graded on*, not a metric that sounds related.

---

## 5. Simple, unsupervised alignment beats a trained network

We built the neural-network embedding the notebook's own strategy document
describes. It **lost** to nearest-average on plain features.

Meanwhile two methods that are barely 15 lines each — CORAL (match the data
shape) and whitening (untangle the features) — produced every real gain we have.

Why the network failed: the 60 features are *already* good enough to separate the
groups linearly. A network only helps when the raw features are tangled in a way
a straight line can't cut. Here they aren't, so there's nothing for it to learn,
and it just adds noise and variance.

**Lesson for this dataset:** the leverage is in *feature engineering* and
*distribution alignment*, not in *model complexity*. Reach for a bigger model
last, not first.

---

## 6. Redundant features are a hidden tax on distance-based methods

The 60 features carry only ~20 features' worth of independent information — the
rest are echoes (blue/green/red brightness all move together, etc.).

- A **Random Forest doesn't care** — it picks whichever copy is convenient and
  ignores the rest.
- The **nearest-average method cares a lot** — it adds up differences across all
  60, so a triple-counted "brightness" idea drowns out a single-counted "timing"
  idea.

Whitening fixed exactly this and was our biggest few-shot gain. **The method you
choose changes which data problems matter.** Same features, same data — the
redundancy was invisible to one model and a 6-point handicap to another.

---

## 7. Things we verified rather than assumed — and one the notebook got wrong

- The notebook says "Class 1 dominates." It doesn't — **Class 2 is the biggest**
  in both cities, Class 1 is the *smallest* in Madrid. Always count, don't trust
  the caption.
- The notebook's "metric learning" is **never implemented** — the Madrid model is
  loaded and never used in the transfer step. The whole approach was aspirational.
- Preprocessing that "needs the hub" actually runs in **4 seconds on a laptop**
  once you stop loading 21 unused columns.
- The few-shot support/query split has the same spatial-adjacency risk as the
  Madrid CV folds, and it's just as large: across all five required budgets,
  **78–81% of support pixels have at least one immediate spatial neighbour
  sitting in the query set** (measured directly, `src/amsterdam_split_leakage.py` /
  `docs/AMSTERDAM_SPATIAL_ADJACENCY_AUDIT.md`). This doesn't touch query
  *labels* — the leakage discipline elsewhere in this pipeline still holds —
  but it means part of every reported few-shot score may reflect "this pixel's
  neighbour was in my support set," not genuine cross-pixel generalisation.
  Worth stating explicitly in the F1-interpretation section rather than left
  for a judge to find.

**Lesson:** the provided material is a starting point with real errors in it.
Checking it is not busywork — one of these (the unimplemented transfer) is a
disqualification risk under the rubric.

---

## Where that leaves the strategy

1. **Groups 3 & 4** carry the signal — timing-of-construction features, aligned
   across cities, are the priority.
2. **Report a curve**, and let it be produced by scenario-appropriate machinery:
   CORAL-aligned Random Forest at zero labels, whitened nearest-average / small
   Random Forest from 50 labels up.
3. **Always pair city-specific features with CORAL** — the more informative the
   feature at home, the more the alignment step is doing to keep it usable abroad.
4. **Stop reaching for bigger models.** The wins here are all cheap linear-algebra
   steps on well-chosen features.
5. **The 1-vs-2 confusion is close to a hard limit** with 30 m Landsat. Say so in
   the write-up rather than fighting it — the rubric rewards that kind of honest
   diagnosis.
