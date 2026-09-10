# Slide notes — one plain-language page per slide

Hand these to anyone presenting. Each file now covers, for one slide:

0. **Layout** — a wireframe of where everything goes, ready to build in Canva.
1. **What this earns on the rubric** — which scored criterion this slide carries, and why it matters.
2. **What the slide says** — the one sentence.
3. **In plain words** — the idea, no jargon (or jargon explained in brackets).
4. **The numbers on it** — what each figure means.
5. **Bullets, exact text** — copy-paste ready.
6. **Timing breakdown** — seconds per segment, and what to cut first if you're behind.
7. **What to actually say** — ~30–60 s of spoken script.
8. **Handoff line** — the exact sentence that passes to the next speaker.
9. **If you're running short** — a pre-built shorter version, so nobody improvises a cut live.
10. **If someone asks** — likely questions with short answers.
11. **Common mistakes presenting this slide** — specific to that slide, from watching it get built.
12. **Words to avoid** — jargon that loses a general audience, and the plain phrase to use instead.

The slides themselves (headings, bullets, figures) are in
`deliverables/slides-copypaste.txt`. Figures are in `deliverables/figures/`.

## The deck at a glance

| # | heading (the claim) | figure | speaker | time |
|---|---|---|---|---|
| 1 | A model that fails in a new city is mis-tuned, not broken | `fig_slide1_scatter.png` | A | 1:30 |
| 2 | Line the cities up first — the model does the rest | `fig_slide2_pipeline.png` (only) | B | 2:00 |
| 3 | The alignment does the work; labels buy the last points | `fig_slide3_curve.png` + `fig_slide3_table.png` | C | 2:00 |
| 4 | Simple fixes beat clever ones | none — native text, 5 rows | D | 1:15 |
| 5 | We checked how much of the few-shot gain is real | `fig_slide5_bars.png` | D | 1:30 |
| 6 | It's a calibration problem, not a hard one | none | all | 0:40 |

Total ~8:55 + Q&A. Handoffs: A→B after 1, B→C after 2, C→D after 3, D holds 4–5.

Each slide's note file now opens with a **§0 Layout** wireframe and, where
relevant, the exact bullet/point text to paste into Canva — that's the fastest
way to use these while actually building the slide.

**Backup figures**, not on any slide unless asked: `fig_slide2_scatter.png` /
`alt_slide2_scatter_2panel.png` (the line-up working), `fig_slide5_spatial.png`
(line-chart version of the audit), `fig_slide5_perclass.png`,
`alt_slide5_confusion_from_notebook.png`, `fig_slide4_didntwork.png` (wording
reference only — don't put the image itself on slide 4).

## Which slide earns which rubric points

The rubric has three scored dimensions (Challenge Understanding /30, Originality
/40, Presentation & Collaboration /30) plus deduction rules. Roughly:

| slide | dimension it mainly carries | specific line item |
|---|---|---|
| 1 | Presentation | Abstract & Pitch |
| 2 | Originality (the big one) + Challenge Understanding | Methodological Innovation; Model Design & Adaptation |
| 3 | Presentation + the required deliverable | "masterful interpretation of performance curves"; F1 table + plot |
| 4 | Originality | Insightful Failure |
| 5 | Challenge Understanding + deduction avoidance | "good understanding of limitations"; disarms the leakage-deduction risk |
| 6 | Presentation | Cohesion / team roles; restates Originality once more |

**Slide 5 is the one slide that's also risk management** — the ~80% adjacency
finding is exactly what a sharp judge would probe for, and disclosing it
yourself (with a number attached) turns a possible deduction into a point in
your favour. If time runs short in rehearsal, protect slides 2 and 5 first;
trim 4 or 6 before either of those.

## Rehearsal checklist

- [ ] Everyone has read their own slide's note file at least once
- [ ] Full run-through once, timed — target ~8:55 + Q&A for all six slides
- [ ] Every speaker knows their **handoff line** (§8 in each file) — this is
      what makes the transitions feel rehearsed instead of awkward
- [ ] Every speaker has practiced their **short version** (§9) in case the talk
      runs long — decide in advance who cuts what, don't improvise on stage
- [ ] Team name is the same on slide 1 and slide 6
- [ ] Abstract text (slide 1) matches the numbers actually shown on slides 2–3
- [ ] Nobody says "data leakage" out loud unless it's to explain what you *didn't* do
- [ ] Each of the 4 team members can be seen to speak at least once
- [ ] Someone has the answer ready for: 4-vs-5 F1 scores, and the word cap on
      the written justification (see `docs/WRITTEN_JUSTIFICATION.md` header)

## The five numbers everyone should know cold

| number | what it is |
|---|---|
| **0.66** | Madrid model tested on Madrid — a model doing its own home city ("the ceiling") |
| **0.43** | that same model dropped straight onto Amsterdam, no adjustment |
| **0.36 → 0.65** | Amsterdam, no local labels — before vs after lining the cities up |
| **0.66 / 0.68 / 0.70 / 0.72 / 0.74** | Amsterdam with 5 / 25 / 50 / 100 / 200 labelled patches per class |
| **~half** | how much of the few-shot gain turned out to be labelled patches sitting next to scored ones (slide 5) |

## Words to avoid across the whole deck

| don't say | say instead |
|---|---|
| domain shift / covariate shift | the two cities look different to the satellite |
| CORAL / covariance alignment | line the cities up |
| whitening / decorrelation | untangle the features |
| class-conditional | one age group at a time |
| pseudo-labels | the model's own first guesses |
| spatial autocorrelation / spatial leakage | labelled patches sitting next to scored patches |
| **data leakage** | *(avoid entirely — it has a specific disqualifying meaning; we did NOT do it)* |
| zero-shot / few-shot | with no local labels / with a few local labels |
| "steep to 50, then flat" / "plateau" | "climbs steadily" — the real curve has no plateau through 200 |

## The handoff chain (say these, in order, and the talk flows itself)

1. A, end of slide 1 → *"...that's what the rest of the talk fixes."*
2. B, end of slide 2 → *"So that's the method. Next — did it actually work?"*
3. C, end of slide 3 → *"Before we call that a win, we checked ourselves on it."*
4. D, end of slide 4 → *"And we didn't just check other people's methods — we checked our own result too."*
5. D, end of slide 5 → *"So — what does all of that add up to?"*
6. all, slide 6 → *"Roles are on the slide. Happy to take questions."*

## Glossary — terms you'll hear while rehearsing, decoded once

| term | plain meaning |
|---|---|
| macro-F1 | the score; 0=useless, 1=perfect, all 4 classes weighted equally |
| CORAL / "the line-up" | reshaping one city's features to match another's spread |
| class-conditional | doing the line-up separately per age group, not all at once |
| pseudo-labels | the model's own guesses, used to steer the line-up (never the real answer) |
| whitening / "untangle the features" | removing double-counted signal between the 108 numbers |
| zero-shot | with no local (Amsterdam) labels at all |
| few-shot | with a handful of local labels (5 to 200 per class) |
| macro-F1 "±" value | how much the number wobbles across repeated random draws — not a confidence interval |
| the adjacency finding | ~80% of randomly-sampled support labels sit next to a scored patch |
| "held a map-tile away" | the stress-test where support labels are forced far from scored patches |
| data leakage | using the *test answers* in training — we do not do this; don't say the phrase casually |
