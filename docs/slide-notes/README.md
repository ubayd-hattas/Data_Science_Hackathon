# Slide notes — one plain-language page per slide

Hand these to anyone presenting. Each file explains, for one slide:

1. **What the slide says** — the one sentence.
2. **In plain words** — the idea, no jargon (or jargon explained in brackets).
3. **The numbers on it** — what each figure means.
4. **What to actually say** — ~30–60 s of spoken script.
5. **If someone asks** — likely questions with short answers.
6. **Words to avoid** — jargon that loses a general audience, and the plain phrase to use instead.

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
