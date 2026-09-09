# Slide 3 — The alignment does the work; labels buy the last points

**Speaker:** C · **Time:** 2:00 · **Figures:** `fig_slide3_curve.png` (large) + `fig_slide3_table.png` (beside it)

---

## 1. What this slide says

The results. **No labels: 0.36 → 0.65.** Then with 5 / 25 / 50 / 100 / 200
labelled patches per class: **0.66 / 0.68 / 0.70 / 0.72 / 0.74.** Madrid scored
on itself is **0.66**. The curve climbs fast to about 50 labels, then flattens.

## 2. In plain words

- **The big jump is the free one.** 0.36 to 0.65 with zero Amsterdam labels,
  purely from lining the cities up.
- **After that, labels add a little at a time.** 5 per class gets you 0.66; you
  have to go all the way to 200 to reach 0.74.
- **The dashed line is Madrid scored on itself** (0.66) — roughly the best you
  could hope for with unlimited home data. On this test, our Amsterdam curve
  reaches that line at about 50 labels per class.
- **The shape matters more than any single point:** steep, then flat. The
  line-up does the heavy lifting; the labels are polish.
- **The error bars are tiny** (a few thousandths), so the order of the points is
  real, not luck.
- Both a **table** and a **plot** are on the slide because the challenge asks for
  both, each with error bars.

## 3. The numbers on this slide

| setting | macro-F1 | plain meaning |
|---|---|---|
| Madrid on itself | 0.66 ± 0.004 | the home-city ceiling |
| Amsterdam, no labels, raw | 0.36 | Madrid model, no adjustment |
| Amsterdam, no labels, lined up | **0.65** | after the group-by-group line-up |
| Amsterdam, 5 / class | 0.66 ± 0.003 | |
| Amsterdam, 25 / class | 0.68 ± 0.006 | |
| Amsterdam, 50 / class | 0.70 ± 0.009 | matches the home-city ceiling |
| Amsterdam, 100 / class | 0.72 ± 0.007 | |
| Amsterdam, 200 / class | 0.74 ± 0.005 | |

The provided starter notebook got about **0.42 → 0.67** across the same budgets,
and **0.43** with no labels. We're above it everywhere.

## 4. What to actually say (~50 s)

> "Read the jump first: 0.36 to 0.65 with no local labels. That's the line-up
> doing its job.
>
> Then the curve. Five labelled patches per class: 0.66. Twenty-five: 0.68.
> Fifty: 0.70. A hundred: 0.72. Two hundred: 0.74.
>
> The dashed line is Madrid scored on its own city — 0.66. Our Amsterdam curve
> reaches it around fifty labels per class.
>
> The shape is the point: steep to fifty, then basically flat. Once the cities
> are aligned, extra labels mostly just re-confirm what the model already knows.
> And the error bars are a few thousandths wide, so that ordering is real.
>
> One honest note before the next slide — we checked how much of that label gain
> is genuine, and slide 5 has the answer."

## 5. If someone asks

- **"Is 0.74 a good score?"** In context, yes: the provided starter got
  0.42–0.67, and we match a model working on its own home city.
- **"Why does the curve flatten?"** Once the two cities are aligned, more labels
  mostly re-confirm rules the model already has.
- **"Why is 5-labels barely above zero-shot?"** Five examples per class is almost
  nothing — not enough to learn much beyond what the line-up already gave you.
- **"± is a confidence interval?"** No — it's the spread over repeated random
  draws of the labelled set. It tells you how much the number wobbles run to run.

## 6. Words to avoid

| don't say | say instead |
|---|---|
| zero-shot / few-shot | with no local labels / with a few local labels |
| log-scale x-axis | the gap between points doubles each step |
| variance / standard deviation | how much the number wobbles run to run |
