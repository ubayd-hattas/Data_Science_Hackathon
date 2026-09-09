# Talk script — 8 slides, ~10 minutes

Spoken word, not bullet points. Say it roughly like this; don't read it verbatim.
Times are targets. `[→]` = advance slide. `[hand to X]` = speaker change.

Total ≈ 9:50, leaving ~10 s slack before the 5-minute Q&A.

---

## Slide 1 · Title  — 0:30  — *Speaker A*

> "We looked at estimating the *age* of buildings from satellite imagery — which
> band of years they were built in — and specifically at making that work in a
> city the model has never seen. We trained on Madrid and transferred to
> Amsterdam. The abstract is on the slide; the short version is that a model
> which fails in a new city isn't broken, it's mis-tuned, and you can fix most of
> that without any local labels."

`[hand to A stays]`

---

## Slide 2 · §1 Problem & contributions  — 1:15  — *Speaker A*

`[→]`

> "The task: every 30-metre Landsat pixel gets sorted into one of four
> construction-era classes, and we're scored on macro-F1 — so all four classes
> count equally, including the rare ones.
>
> We train on Madrid, where we have labels, and adapt to Amsterdam using between
> 5 and 200 labelled pixels per class.
>
> Why is that hard? Two kinds of shift. The obvious one is that Amsterdam
> buildings *look* different to the satellite — different materials, wetter
> climate, different mix of Landsat sensors over the years. The subtler one is
> that the *proportions* of the age classes differ between the cities — class 1
> is about 1.6 times more common in Amsterdam. Both have to be handled.
>
> Our three contributions are on the slide. The one to remember is the first:
> **class-conditional CORAL** — we align the two cities' feature distributions
> one age class at a time, and we do it using only the model's own predictions,
> never Amsterdam labels."

`[hand to B]`

---

## Slide 3 · §2 Data & representation  — 1:00  — *Speaker B*

`[→]`

> "Each pixel has up to 40 years of annual observations in six spectral bands. We
> gap-fill the series and then collapse it into 108 summary numbers per pixel.
>
> Collapsing to one row per pixel is deliberate — if you feed the model one row
> per year it just learns 'this looks like a finished building' and never sees
> the *trend*. The summaries force it to reason about change over time.
>
> The features that matter most for the recent classes are about the *timing* of
> the biggest single-year jump — that's the construction event, and it looks
> similar in any city because it's physics, not a Madrid quirk. That's the whole
> design principle: pick features that transfer because of how construction
> works, not features that happen to separate Madrid."

---

## Slide 4 · §3 Method  — 1:45  — *Speaker B*

`[→]`

> "Here's the pipeline. Stage 1 is a class-balanced Random Forest on the Madrid
> features — 500 trees, nothing exotic.
>
> Before we use it on Amsterdam, we align. Plain CORAL reshapes Madrid's whole
> feature cloud to match Amsterdam's. **Class-conditional** CORAL goes further:
> we predict Amsterdam once, then for each class we reshape Madrid's examples of
> that class to match the Amsterdam pixels the model *thinks* are that class, and
> refit. Two rounds. That single step takes the zero-label score from 0.36 to
> 0.65.
>
> Then the two regimes. With zero labels you just use that aligned model. With
> *n* labels, we whiten the Amsterdam features — strength scaled to how many
> labels we have, because whitening needs data — train a small forest on the
> support set, blend its vote with the Stage-1 model, and finally smooth the
> predictions over each pixel's map-neighbours, weighting closer ones more.
>
> The box at the bottom is the important disclaimer: every statistic we compute —
> the alignment, the whitening, the prior — comes from *unlabelled* Amsterdam.
> The only labelled target data that touches the model is the support set itself."

`[hand to C]`

---

## Slide 5 · §4 Experimental setup  — 1:15  — *Speaker C*

`[→]`

> "Madrid reference is a 5-by-5 repeated stratified cross-validation. Amsterdam is
> evaluated at the five required budgets, 20 random support draws at each, so we
> get an error bar.
>
> Model selection: we ran a 500-configuration search over the feature set and
> hyperparameters, but we scored it on *one half* of Amsterdam and confirmed the
> winner *once* on the untouched other half — so the tuned numbers aren't
> inflated by fitting to the data we report.
>
> Two honesty notes. Repeated resampling isn't k-fold CV — the ± is spread, not a
> confidence interval. And we measured that about 80% of our support pixels have
> an immediate neighbour in the query set, because the sampling is random. We
> don't change the protocol — that's what we're asked to run — but we say the
> number."

---

## Slide 6 · §5 Results  — 2:00  — *Speaker C*

`[→]`

> "Table 1, top to bottom. Zero-shot on raw features is 0.36. With
> class-conditional alignment, 0.65 — that jump is the headline of the method.
>
> Then the few-shot curve — Figure 2. Five labels per class: 0.66. Twenty-five:
> 0.68. Fifty: 0.70. A hundred: 0.72. Two hundred: 0.74.
>
> The dashed line is Madrid scored on *itself* — 0.66 — the in-city ceiling. The
> transfer curve crosses it at about 50 labels per class and keeps climbing. So
> with roughly 200 checked buildings total, the model does as well on Amsterdam
> as any model does on its home city.
>
> The shape matters too: steep to 50, then it flattens. The alignment does the
> heavy lifting; the labels mostly buy you the last few points. And the error
> bars are tight — a few thousandths — so the ordering is real."

`[hand to D]`

---

## Slide 7 · §6 Analysis  — 1:30  — *Speaker D*

`[→]`

> "Left column — what each piece is worth. Class-conditional CORAL, plus 0.07 at
> zero labels. Budget-scaled whitening, plus 0.15 at five labels — that's the
> biggest single lever in the low-data regime. The prior blend and the smoothing
> add a bit more.
>
> Middle column — what didn't work, and this is deliberate. We built the triplet
> embedding the challenge notes suggest; it lost, because the features are
> already linearly separable. Ordinal loss made our mistakes *smaller* but not
> fewer, and the metric only counts right versus wrong. Label-shift correction,
> self-training, gradient boosting — all worse, each for a specific reason on the
> slide. The pattern: every attempt to out-*model* the data lost to better
> *use* of it.
>
> Right column — the ceiling. Classes 1 and 2 are both pre-1984, before the
> satellite record starts, so there's no construction event to separate them.
> That's a data limit, and the building-age literature reports the same thing.
> Most of our remaining error is right there."

`[hand to D stays]`

---

## Slide 8 · §7 Conclusion  — 0:35  — *Speaker D*

`[→]`

> "Three things to take away. A model in a new city is mis-calibrated, not
> incapable — per-class alignment recovers most of the gap with zero labels.
> About 50 labelled pixels per class buys in-city accuracy, and the low-data
> regime needs its own recipe, not a shrunk version of the full one. And on
> 30-metre Landsat, careful use of the distribution beat every
> learned-representation method we tried.
>
> Our originality claim is the iterative per-class alignment, and the framing
> that transfer is a spectrum — different machinery at different label budgets.
>
> Contributions are on the slide. Happy to take questions."

`[all — Q&A]`

---

## If asked in Q&A — quick answers

- **"Why Random Forest not deep learning?"** 30-metre Landsat, ~100k pixels,
  features that are already linearly separable — a forest plus alignment is the
  right capacity. We tried an embedding; it added variance without gain.
- **"Is the spatial adjacency a leak?"** No rule is broken — query labels are
  never used, and the protocol *is* random sampling. But we disclose that part
  of the score is same-block proximity, and we kept the smoothing
  prediction-only so it can't compound it.
- **"Four or five F1 scores?"** Notebook 1 lists five budgets but says "four" in
  the deliverables. We report all five; the plot shows all five.
- **"What's class-conditional CORAL in one line?"** Align each Madrid age class
  to the covariance of the Amsterdam pixels the model assigned to that class,
  then refit — twice, no target labels.
- **"How long to run?"** End-to-end from the notebook, one fixed seed, a few
  minutes plus the cross-validation.
