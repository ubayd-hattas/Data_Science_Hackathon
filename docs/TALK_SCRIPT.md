# Talk script — 7 slides, ~9:40

Spoken word, not bullet points. Say it roughly like this; don't read it out.
`[→]` = advance slide. `[hand to X]` = speaker change. Times are targets.

---

## Slide 1 · Building age from orbit, built to travel — 0:30 — *Speaker A*

> "We estimated the *age* of buildings from satellite imagery — which band of
> years they were built in — and built the method to work in a city it's never
> seen. We trained on Madrid and moved it to Amsterdam. The abstract's on the
> slide; the short version is that a model which fails in a new city isn't
> broken, it's mis-tuned, and you can fix most of that with no local labels."

---

## Slide 2 · A new-city model is mis-tuned, not incapable — 1:15 — *Speaker A*

`[→]`

> "The task: every 30-metre satellite patch goes into one of four
> construction-era groups, and we're scored on macro-F1 — so all four count
> equally, including the rare ones.
>
> We train on Madrid, where we have labels, and adapt to Amsterdam with somewhere
> between 5 and 200 labelled patches per group.
>
> Why does it break? Two reasons. The buildings physically *look* different to
> the satellite — brick versus concrete, a wetter climate, different satellites
> over the forty years. And the *mix* of ages is different — the oldest group is
> about one-and-a-half times more common in Amsterdam. Both have to be handled,
> and neither is fixed by just retraining."

`[hand to B]`

---

## Slide 3 · Line the cities up first — the model does the rest — 2:15 — *Speaker B*

`[→]`

> "First, the data. Each patch has up to forty years of readings in six colours.
> We turn that into about a hundred numbers per patch. The important ones are
> about *when* the biggest year-to-year change happened — that's the construction
> event, and it looks the same in any city because it's physics.
>
> One row per patch, not one per year — otherwise the model just learns 'this is
> a finished building' and never sees the change over time.
>
> Now the method. We train a plain model on Madrid — a Random Forest. Before we
> use it on Amsterdam, we line the two cities up: we predict Amsterdam once, then
> for each age group we reshape Madrid's examples of that group to match the
> Amsterdam patches the model *thinks* are that group, and retrain. We do that
> twice. No Amsterdam answers — just the model's own guesses.
>
> That one step takes the no-label score from 0.36 to 0.65. And the only labelled
> Amsterdam data that ever touches the model is the handful of examples we're
> given."

`[hand to C]`

---

## Slide 4 · Few examples borrow from Madrid; many trust the local data — 1:15 — *Speaker C*

`[→]`

> "Then the local examples. With five to twenty-five, a model trained only on
> those is shaky, so we mostly trust the aligned Madrid model and let the local
> one nudge it. With two hundred, we mostly trust the local model.
>
> The balance shifts automatically as the number grows — nothing is hand-set per
> case. We also untangle the features more gently when there's little data,
> because that step needs enough examples to be stable.
>
> The point: the low-data case isn't the full pipeline with less data. It's a
> deliberately different recipe."

---

## Slide 5 · Fifty local examples = home-city accuracy — 2:00 — *Speaker C*

`[→]`

> "Here are the numbers. No labels: 0.36 raw, 0.65 after the line-up — that jump
> is the headline.
>
> Then the curve. Five examples per group: 0.66. Twenty-five: 0.68. Fifty: 0.70.
> A hundred: 0.72. Two hundred: 0.74.
>
> The dashed line is Madrid scored on *itself* — 0.66 — the best you could hope
> for with unlimited home data. Our transfer curve crosses it at about fifty
> examples per group and keeps climbing. So with roughly two hundred checked
> buildings total, the model does as well on Amsterdam as any model does on its
> home city.
>
> The shape matters: steep to fifty, then flat. The line-up does the heavy
> lifting; the examples buy the last few points. And the error bars are a few
> thousandths, so the ordering is real."

`[hand to D]`

---

## Slide 6 · Simple alignment beat every clever alternative — 1:45 — *Speaker D*

`[→]`

> "What each piece is worth is on the left. The city line-up: plus 0.07 at zero
> labels. Gentler untangling at low data: plus 0.15 at five examples — the
> biggest single lever down there. The blend and the neighbour smoothing add a
> bit more.
>
> The middle column is what didn't work, and we show it on purpose. We built the
> neural network the challenge notes suggest — it lost, because the data was
> already easy to separate, so it had nothing to learn. Ordinal training made our
> mistakes smaller but not fewer, and the score only counts right versus wrong.
> Correcting the age mix, self-training, gradient boosting — all worse, each for
> a specific reason. The pattern: every attempt to out-think the data lost to
> using it more carefully.
>
> On the right, the ceiling. The two oldest groups are both from before the
> satellite record starts, so there's no construction event to tell them apart.
> That's a data limit — the research on old buildings says the same — and it's
> where most of our remaining error is. We also checked that about eighty percent
> of our examples sit right next to a test patch, and we say so."

`[hand to D stays]`

---

## Slide 7 · Cross-city transfer is a calibration problem — 0:40 — *Speaker D*

`[→]`

> "Three things to take away. A model in a new city is mis-tuned, not incapable —
> lining the cities up recovers most of the gap with zero labels. About fifty
> local examples buy home-city accuracy, and the low-data case needs its own
> recipe. And on this kind of satellite data, careful use of the distribution
> beat every fancier model we tried.
>
> Our original bit is doing the line-up group by group, using the model's own
> guesses, and the framing that few and many examples need different machinery.
>
> Roles are on the slide. Happy to take questions."

`[all — Q&A]`

---

## If asked — quick plain answers

- **Why not deep learning?** The data's small and already easy to separate — a
  forest plus the line-up is the right size. We tried a neural net; it added
  noise, no gain.
- **Is the neighbour thing cheating?** No — we never use test answers, and the
  organisers' rule *is* random sampling. But part of the score is nearby
  buildings looking alike, so we disclose the number and kept the smoothing to
  predictions only.
- **Four or five scores?** The instructions say five sizes but "four" in one
  place. We report all five; the plot shows all five.
- **The line-up in one sentence?** Reshape each Madrid age group to match the
  Amsterdam patches the model thinks are that group, then retrain — twice, no
  answers used.
- **How long to run?** From the notebook, one fixed seed, about fifteen minutes.
