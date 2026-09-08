# Project Guide — Everything That's Happening, Explained Simply

This is a plain-language walkthrough of the whole project. No prior knowledge assumed.
Read it top to bottom and you'll understand what every notebook does and why.

---

## 1. The big picture

### What are we trying to do?

Look at a satellite image of a city. Pick one small square of ground (30 metres × 30 metres)
that has buildings on it. **Guess how old those buildings are** — not the exact year, just
which of 4 age brackets they fall into.

| Class | Meaning |
|:-----:|:--------|
| **1** | Oldest buildings (before ~1945–1960, depends on the city) |
| **2** | Old buildings (that cut-off year → 1984) |
| **3** | Middle-aged buildings (1984 → 2004) |
| **4** | Newest buildings (2004 → 2024) |

### Why can a satellite possibly know this?

Satellites don't take normal photos. They measure **how much light bounces off the ground
in different colours** (wavelengths) — including colours our eyes can't see (infrared).

Different building materials bounce light differently. A 1950s brick terrace reflects light
differently than a 2015 concrete-and-glass block. Roofs age, get dirty, get replaced.
So the "colour fingerprint" of a patch of ground carries a hint about building age.

The hint is weak and noisy, but with enough data a machine-learning model can pick it up.

### The hard part: making it work in a *new* city

We have good building-age records for **Madrid** and **Amsterdam**. The real goal is a model
that works **anywhere** — Tokyo, Nairobi, São Paulo — where we have almost no records.

So the challenge is set up as:

1. **Train** a model using lots of Madrid data (where we know the answers).
2. **Transfer** it to Amsterdam using only a tiny handful of known Amsterdam examples
   (5, 25, 50, 100, or 200 per class).
3. See how well it does. A model that needs only 25 Amsterdam examples to work well has
   learned something *general* about building age, not just "how Madrid looks".

This idea — train big on one place, adapt cheaply to another — is called **transfer learning**.

---

## 2. Key vocabulary (read this once)

| Term | Plain meaning |
|------|---------------|
| **Landsat** | A NASA/USGS satellite programme photographing Earth since 1972. Free data. |
| **Spectral band** | One "colour channel" the satellite measures. We use 6: Blue, Green, Red, NIR, SWIR1, SWIR2. |
| **NIR / SWIR** | Near-infrared and short-wave infrared — invisible to us, very useful for surfaces. |
| **Reflectance** | How much light bounced back, as a number. Bright surface = high, dark = low. |
| **Pixel** | One 30 m × 30 m square of ground. The basic unit we classify. |
| **Time series** | The same pixel measured once a year for ~40 years → a sequence of numbers. |
| **Feature** | One number describing a pixel that we feed to the model (e.g. "average NIR brightness"). |
| **Feature vector** | The full list of features for one pixel (here: 60 numbers). |
| **Label** | The correct answer for a pixel (its age class, 1–4). |
| **Wide format** | Data layout: one row per (pixel, year), columns for each measurement. |
| **Parquet** | A compressed file format for big tables. Loads much faster than CSV. |
| **Model** | The thing that learns the pattern and makes predictions. Here: a Random Forest. |
| **Cross-validation (CV)** | A fair way to test a model on data it didn't train on. |
| **Macro F1** | The score we care about. 0 = terrible, 1 = perfect. "Macro" = every class counts equally, even small ones. |
| **Zero-shot** | Using the Madrid model on Amsterdam with *no* Amsterdam training at all. |
| **Few-shot** | Adapting to Amsterdam with only a *few* labelled Amsterdam examples. |
| **Metric learning** | Teaching the model a notion of "distance" so same-age buildings sit close together. |
| **Prototype** | The average feature vector of a class — its "typical example". |

---

## 3. The data

Two files (you place them in `data/`):

- `madrid_train.parquet` — the training city.
- `amsterdam_data.parquet` — the transfer-target city.

Each file is a big table. **One row = one pixel, one year.** Because a satellite might catch
a location up to 3 times in a summer, each row can hold up to 3 sets of measurements
(`Blue_1, Blue_2, Blue_3`, and so on), plus quality flags (`qa_valid_1…3`) saying whether
each measurement was clean (no cloud/shadow).

Important columns:

| Column | What it is |
|--------|-----------|
| `pixel_id` | Unique name for a pixel-year. |
| `city`, `year` | Self-explanatory. |
| `px_key`, `py_key` | Grid coordinates — identify a physical location across years. |
| `coverage` | Fraction of the pixel covered by buildings (always ≥ 0.15 here). |
| `weighted_mean_year` | Average construction year of the buildings in the pixel. **This is what age class is derived from.** |
| `Blue_1 … SWIR2_3` | The actual spectral measurements. |
| `qa_valid_N` | Was observation N a clean pixel? |

Note: the files do **not** come with age-class labels. You compute them yourself in
Notebook 3 by bucketing `weighted_mean_year`.

The data is already cleaned: only pixels with ≥15% building coverage, no corrupt values,
no construction years before 1850.

---

## 4. Notebook by notebook

### Notebook 1 — Introduction

No code. It explains the challenge, the dataset, the 4 age classes, and exactly what you
must submit (trained model, adaptation code, a slide deck, a short written justification).

**Prizes** reward different things: best average accuracy across all label budgets, best
peak accuracy, best accuracy with only 25 labels/class (the "did you really generalise?"
prize), and best originality.

---

### Notebook 2 — Reading & Understanding the Data

**Goal: look at the data before modelling anything.** You can't build a good model for data
you haven't eyeballed.

What it does, step by step:

1. **Load** both parquet files.
2. **Derive `age_class`** from `weighted_mean_year` using the city-specific cut-offs
   (Madrid splits class 1/2 at 1960, Amsterdam at 1945; both split 3/4 at 2004).
3. **Make a "flat view" (`df_flat`)** — collapse the up-to-3 observations per row down to
   the single best clean one. Easier to work with.
4. **Remove a known glitch:** some 2003 Amsterdam scenes from Landsat 7 are hazy and give
   absurdly high Blue values. Any row with Blue > 15,000 is dropped. *(You must do the same
   in Notebook 3.)*
5. **Explore with plots:**
   - **Building coverage distribution** — how much of each pixel is actually building.
   - **Age-class distribution** — reveals **class imbalance**. Note the notebook text
     says "Class 1 dominates"; the actual counts say otherwise. **Class 2 is the largest
     in both cities**, and Class 1 is the *smallest* in Madrid:

     | Class | Madrid | Amsterdam |
     |:-----:|-------:|----------:|
     | 1 | 18.1% | **29.1%** |
     | 2 | **35.6%** | **33.9%** |
     | 3 | 24.5% | 25.9% |
     | 4 | 21.8% | **11.1%** |

     Two things follow. A lazy model scores decently by over-guessing Class 2 — which is
     exactly what Notebook 4 observes. And the two cities have *different* class
     proportions (Class 1 is 1.6× more common in Amsterdam, Class 4 half as common),
     which is a second, separate kind of domain shift — see the guide's §5 and
     MASTERCLASS.md.
   - **Single-pixel time series** — pick a few pixels per class, plot 40 years of one band.
     Lesson: individual pixels are noisy; you can't eyeball the age. But there's a slow
     downward drift (surfaces weather/darken) and sometimes a wobble near the construction
     year for classes 3 & 4.
   - **Class-level yearly trends** — average all pixels in a class, per year, with a
     spread band. Lesson: Class 4 is very "spread out" around 2000–2015 — those pixels
     were building sites / bare ground / old buildings at different stages. That
     variability is itself a clue.
   - **Per-pixel temporal statistics** — for each pixel compute the mean and standard
     deviation of its 40-year series, and of its year-to-year changes. Histograms per
     class show:
     - **Mean brightness** rises from Class 1 (darkest) to Class 4 (brightest), in every
       band. Promising signal.
     - **Std of year-to-year change** is small for Classes 1 & 2, larger for 3 & 4. That
       split lands exactly at 1984 (start of the satellite record) — buildings that
       predate the record look "settled"; buildings built during it show construction
       wobble.
   - **Band correlations** — Blue/Green/Red are nearly redundant with each other;
     NIR/SWIR carry extra information. So multispectral > plain RGB for this task.

**Takeaway:** no single feature separates the 4 classes. A model must combine many bands
and many temporal statistics. That's exactly what Notebook 3 builds.

---

### Notebook 3 — Feature Engineering & Preprocessing

**Goal: turn the messy time-series table into one clean row of numbers per pixel, ready
for a model.**

The single most important design decision:

> **One row per pixel — not one row per (pixel, year).**

If you fed the model raw yearly rows, it would treat 2005 and 2006 for the same pixel as
two unrelated examples, and it would never "see" the trend over time. Instead we squash
each pixel's ~40 years into a fixed list of **summary statistics** that describe *how it
changed*.

Steps:

1. **Load** both cities.
2. **Assign age classes** (same bucketing as Notebook 2).
3. **Flatten** to one best clean observation per pixel-year. Drop the Blue > 15,000 haze
   rows.
4. **Record coverage flags** — `has_early_data` / `has_late_data`: did this pixel actually
   have any real observations before 2004 / after 2004? (Almost always yes, but flag the
   rare gaps.)
5. **Gap-fill the time series** — a pixel might be missing a few cloudy years. Fill them:
   - gaps *between* real years → straight-line interpolation,
   - missing years *before* the first real one → copy the first value backward,
   - missing years *after* the last → copy the last value forward.
   Now every pixel has a value for every year, so later statistics are always defined.
6. **Compute 5 spectral indices** per observation — standard band ratios that are more
   robust than raw bands:

   | Index | Roughly measures |
   |-------|------------------|
   | NDVI | vegetation (low = built-up) |
   | NDBI | built-up intensity |
   | UI | urban-ness |
   | MNDWI | water (negative on land) |
   | BSI | bare soil vs vegetation |

7. **Aggregate to 60 features per pixel.** These are:

   | Feature group | How many |
   |---------------|:--------:|
   | Mean of each band over all years | 6 |
   | Std of each band over all years | 6 |
   | Mean of each spectral index over all years | 5 |
   | Std of each spectral index over all years | 5 |
   | Mean of each band, **early period only (1984–2003)** | 6 |
   | Std of each band, early period | 6 |
   | Mean of each band, **late period only (2004+)** | 6 |
   | Std of each band, late period | 6 |
   | Mean year-on-year change per band | 6 |
   | Std of year-on-year change per band | 6 |
   | `has_early_data`, `has_late_data` flags | 2 |
   | **Total** | **60** |

   **Why split early vs late at 2004?** The 1984–2003 window shows what *used to be* at
   that location; 2004+ shows the *current* building. For old buildings (Class 1–2) both
   windows look the same. For new buildings (Class 3–4) the two windows look very
   different — and that contrast is a strong age signal.

8. **Standardise** — rescale every feature to mean 0, standard deviation 1. This is
   essential for distance-based methods in Notebook 4 (otherwise a big-numbered feature
   would dominate). **The scaler is fitted on Madrid only** and then applied to Amsterdam —
   that's the honest protocol, since in real life you wouldn't have Amsterdam statistics
   up front.
9. **Save** everything (`X_madrid`, `y_madrid`, `X_amsterdam`, `y_amsterdam`, feature
   names, the scaler) to `data/preprocessed/preprocessed_data.pkl` so Notebook 4 just
   loads it.

---

### Notebook 4 — Modelling & Evaluation

**Goal: train on Madrid, measure honestly, then transfer to Amsterdam.**

The chosen approach is **metric learning**: learn a feature space where two buildings of
the same age land close together *regardless of city*. If that works, adapting to a new
city is just "find where each Amsterdam class sits in this space".

The baseline model is a plain **Random Forest** — many decision trees that each vote on
the class; the majority vote wins. No tuning. Your job in the hackathon is to beat it.

Steps:

1. **Load** the preprocessed data.

2. **Cross-validation on Madrid** — the honest scoring step.
   - Split Madrid into `k` equal parts ("folds"). Train on `k−1`, test on the held-out 1.
     Rotate so every pixel gets tested once. → `k` scores.
   - A **repeat** re-shuffles and does the whole thing again. Why? The `k` scores from one
     run share most of their training data, so they're correlated and *underestimate* the
     true wobble. Independent repeats give a trustworthy error bar.
   - Output: mean macro-F1 ± std, plus a **confusion matrix** (which classes get mistaken
     for which).
   - **Finding:** F1 is stable to ±0.01, but **Class 2 is over-predicted** — lots of true
     Class 1 pixels get called Class 2, because Class 2 is large and imbalance isn't being
     corrected. (Fixing that is left to you.)

3. **Spatial uncertainty mapping** — instead of just a label, use `predict_proba` to get
   the model's confidence `(p1, p2, p3, p4)` for each pixel, then map it geographically:
   - **Shannon entropy**: how unsure the model is (0 = certain, 2 bits = total guess).
   - **Negative log-probability of the true class**: 0 = confident *and* right; high = wrong
     or unsure.
   - **Finding:** even though raw accuracy is modest, the model gets the *big
     neighbourhoods* right — which suggests a second model working on these probability
     maps could do better.

4. **Final Madrid model** — once CV has told us what to expect, retrain a Random Forest on
   *all* Madrid data. This is the model that goes to Amsterdam.

5. **Feature importance** — which of the 60 features mattered? Answer: lots of them, all
   roughly equal — partly because features are correlated, partly because the forest is
   combining them in complex ways.

6. **Zero-shot transfer to Amsterdam** — run the Madrid model on Amsterdam with **no**
   Amsterdam labels. This measures the raw **domain gap**.
   - **Finding:** accuracy is poor, and Class 2 is wildly over-predicted. Expected —
     Amsterdam brick looks nothing like Madrid concrete. This is the problem few-shot
     is meant to fix.

7. **Few-shot prototype transfer** — the payoff step.
   - Take a small number of *labelled* Amsterdam pixels (5, 25, 50, 100, 200 per class).
   - For each class, average their 60-feature vectors → the class **prototype**.
   - Classify every other Amsterdam pixel by **which prototype it's closest to**
     (Euclidean distance). That's it — it's 1-nearest-neighbour against a smoothed class
     centre.
   - Repeat each budget 10× with different random draws → error bars.
   - **Findings:**
     - Even **5 examples per class** hugely beats zero-shot.
     - **25 per class** adds another ~10 points.
     - **100 per class** basically nails the class centres; more barely helps.
     - Amsterdam per-class recall ends up roughly matching Madrid's.

8. **Summary + ideas to improve** (the notebook's own suggestions):
   - Better features: per-decade stats instead of two periods; linear-trend slope per band;
     year of max/min reflectance; change-point detection.
   - Better metric learning: train a real embedding network with **triplet loss**; use
     Random-Forest leaf co-occurrence as a similarity kernel; domain-adversarial training
     that punishes any feature which reveals *which city* a pixel is from.
   - Better transfer: fine-tune on the Amsterdam labels instead of just prototypes; use the
     unlabelled Amsterdam data for self-training with pseudo-labels.

---

## 5. The metric-learning idea, in one paragraph

Normal classification asks *"what does a 1965 building look like?"* — and the answer is
different in every city, so it doesn't travel. Metric learning instead asks *"what makes
two buildings the **same age**?"* — a **relative** question. If ageing (weathering, material
change, densification) leaves a consistent mark across cities, then a model trained to tell
*same-era* from *different-era* pairs in Madrid will still work in Amsterdam. You train a
small network with **triplet loss** (pull same-age pairs together, push different-age pairs
apart), then classify a new city's buildings by nearest **prototype** (class average) using
just a few local labels. The full write-up is in
[METRIC_LEARNING_APPROACH.md](METRIC_LEARNING_APPROACH.md).

---

## 6. How the pieces connect

```
data/madrid_train.parquet ─┐
data/amsterdam_data.parquet ┘
        │
        ▼
Notebook 2  ── explore, sanity-check, understand the signal
        │
        ▼
Notebook 3  ── flatten ▸ gap-fill ▸ spectral indices ▸ 60 temporal features ▸ standardise
        │
        ▼
data/preprocessed/preprocessed_data.pkl   (X/y for both cities + the scaler)
        │
        ▼
Notebook 4  ── Madrid cross-validation ▸ final Madrid model
        │              │
        │              ▼
        │        zero-shot on Amsterdam   (measures the domain gap)
        │              │
        │              ▼
        └──────▸ few-shot prototype transfer at 5 / 25 / 50 / 100 / 200 labels per class
                       │
                       ▼
        F1 vs label-budget curve  →  this is what you submit
```

---

## 7. What you'd actually change to compete

The provided notebooks are a **baseline**, deliberately un-optimised. Places to improve:

- **Handle class imbalance** — class weights, resampling, or a balanced decision threshold.
  Right now the model leans on Class 2 too hard.
- **Better features** — trend slopes, per-decade stats, change-point year, texture/spatial
  context from neighbouring pixels.
- **Better model** — gradient boosting (XGBoost/LightGBM), or a real learned embedding
  instead of raw standardised features.
- **Better transfer** — fine-tuning, domain adaptation, self-training on unlabelled
  Amsterdam pixels, active learning to choose *which* Amsterdam pixels to label.
- **Always** keep the Madrid-only fitting discipline (scaler, model, feature choices) so
  your Amsterdam score reflects genuine transfer, not leakage.
