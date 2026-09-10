# Slide 3 — The alignment does the work; labels buy the last points

**Speaker:** C · **Time:** 2:00 · **Figures:** `fig_slide3_curve.png` (~⅔ width) + `fig_slide3_table.png` (~⅓ width)
**Subheading:** Zero labels already gets you to 0.65 — the labels add a point or two each

---

## 0. Layout

```
┌──────────────────────────────────────────────────────────┐
│  TITLE (the claim)                                        │
│  subheading                                              │
│  • Zero labels: 0.36 → 0.65 (the line-up alone)          │  ← 3 micro-bullets
│  • Climbs ~+0.02 per doubling — 0.66 at 5, 0.74 at 200   │
│  • Crosses the home-city line (0.66) around 50 labels    │
│  ┌────────────────────────┬────────────────────────────┐ │
│  │  fig_slide3_curve.png  │  fig_slide3_table.png      │ │
│  └────────────────────────┴────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

Both figures are **required** — the rubric asks for the plot *and* the table,
both with error bars. Don't drop either. Put the two figure cards on white
panels with padding so background colour doesn't crowd the axis text.

## 1. What this slide says

The results. **No labels: 0.36 → 0.65.** Then with 5 / 25 / 50 / 100 / 200
labelled patches per class: **0.66 / 0.68 / 0.70 / 0.72 / 0.74.** Madrid scored
on itself is **0.66**. The curve climbs steadily — about +0.02 each time the
label budget grows — and crosses the home-city line around 50 labels per class.

> **Correction from an earlier draft of these notes:** don't say "steep to 50,
> then flat." Looking at the actual numbers (0.66/0.68/0.70/0.72/0.74) there is
> **no plateau** — it keeps climbing about +0.02 per step all the way to 200.
> The steepest single jump is 25→50. Say "climbs steadily," not "then flattens."

## 2. In plain words

- **The big jump is the free one.** 0.36 to 0.65 with zero Amsterdam labels,
  purely from lining the cities up.
- **After that, labels add a little at a time, fairly evenly.** 5 per class gets
  you 0.66; each further doubling of the budget adds roughly another 0.02, up to
  0.74 at 200.
- **The dashed line is Madrid scored on itself** (0.66) — roughly the best you
  could hope for with unlimited home data. On this test, our Amsterdam curve
  reaches that line at about 50 labels per class and keeps climbing past it.
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

## 4. Bullets for the slide (thin strip under the subheading)

```
• Zero labels: 0.36 → 0.65 (the line-up alone)
• Climbs ~+0.02 per doubling of labels — 0.66 at 5, up to 0.74 at 200
• Crosses the home-city line (0.66) around 50 labels per class
```

## 5. What to actually say (~50 s)

> "Read the jump first: 0.36 to 0.65 with no local labels. That's the line-up
> doing its job.
>
> Then the curve. Five labelled patches per class: 0.66. Twenty-five: 0.68.
> Fifty: 0.70. A hundred: 0.72. Two hundred: 0.74.
>
> The dashed line is Madrid scored on its own city — 0.66. Our Amsterdam curve
> reaches it around fifty labels per class and keeps climbing.
>
> The shape is the point: it's a steady climb, not a cliff — each time we double
> the labels we gain about two more points. The line-up did the heavy lifting;
> the labels are a steady top-up. And the error bars are a few thousandths wide,
> so that ordering is real.
>
> One honest note before the next slide — we checked how much of that label gain
> is genuine, and slide 5 has the answer."

## 6. If someone asks

- **"Is 0.74 a good score?"** In context, yes: the provided starter got
  0.42–0.67, and we match — then pass — a model working on its own home city.
- **"Why doesn't it plateau?"** It doesn't really — it keeps climbing steadily.
  There's no evidence yet that it's hit a ceiling by 200 labels.
- **"Why is 5-labels barely above zero-shot?"** Five examples per class is almost
  nothing — not enough to learn much beyond what the line-up already gave you.
- **"± is a confidence interval?"** No — it's the spread over repeated random
  draws of the labelled set. It tells you how much the number wobbles run to run.

## 7. Words to avoid

| don't say | say instead |
|---|---|
| zero-shot / few-shot | with no local labels / with a few local labels |
| log-scale x-axis | the gap between points doubles each step |
| variance / standard deviation | how much the number wobbles run to run |
| "steep then flat" / "plateau" | "climbs steadily" — the data doesn't plateau |
