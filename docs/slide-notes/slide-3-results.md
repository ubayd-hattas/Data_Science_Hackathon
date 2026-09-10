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

## 1. What this earns on the rubric

- **This is the rubric's F1-scores requirement, met directly** — the plot and
  table with error bars are named explicitly as required deliverables.
- **Presentation & Collaboration — "masterful interpretation of performance
  curves across sample sizes."** This is graded on whether you talk about the
  *shape* of the curve (steady climb, crosses the home-city line) rather than
  just reading numbers off the table. The "what to actually say" script below
  is built around that.
- **F1 Interpretation pillar** of the written justification — this slide is the
  spoken version of that paragraph. Keep the numbers identical to the doc.

## 2. What this slide says

The results. **No labels: 0.36 → 0.65.** Then with 5 / 25 / 50 / 100 / 200
labelled patches per class: **0.66 / 0.68 / 0.70 / 0.72 / 0.74.** Madrid scored
on itself is **0.66**. The curve climbs steadily — about +0.02 each time the
label budget grows — and crosses the home-city line around 50 labels per class.

> **Correction from an earlier draft of these notes:** don't say "steep to 50,
> then flat." Looking at the actual numbers (0.66/0.68/0.70/0.72/0.74) there is
> **no plateau** — it keeps climbing about +0.02 per step all the way to 200.
> The steepest single jump is 25→50. Say "climbs steadily," not "then flattens."

## 3. In plain words

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

## 4. The numbers on this slide

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

## 5. Bullets for the slide (thin strip under the subheading)

```
• Zero labels: 0.36 → 0.65 (the line-up alone)
• Climbs ~+0.02 per doubling of labels — 0.66 at 5, up to 0.74 at 200
• Crosses the home-city line (0.66) around 50 labels per class
```

## 6. Timing breakdown (2:00 total)

| segment | time | content |
|---|---|---|
| the jump | 0:20 | 0.36 → 0.65, no labels |
| walk the curve | 0:35 | 5→200, five numbers, don't rush them |
| the dashed line | 0:20 | Madrid ceiling, crosses at ~50 |
| the shape | 0:20 | "climbs steadily" interpretation — this is the graded part |
| error bars | 0:10 | tiny, order is real |
| bridge | 0:15 | flag the honest caveat, hand to slide 5 |

Slow down through "walk the curve" and "the shape" — that's specifically what
"masterful interpretation of performance curves" is checking for. Don't speed
past it to save time.

## 7. What to actually say (~50 s)

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

## 8. Handoff to Speaker D

> "Before we call that a win, we checked ourselves on it — [Name of D] will
> show you what we found."

## 9. If you're running short — cut to this

Skip walking all five numbers individually; say *"0.66 up to 0.74 as labels grow
from 5 to 200"* as one phrase and go straight to the dashed-line comparison and
the shape. Never cut the closing bridge line to slide 5 — losing it makes slide
5 look like it's contradicting you instead of extending you.

## 10. If someone asks

- **"Is 0.74 a good score?"** In context, yes: the provided starter got
  0.42–0.67, and we match — then pass — a model working on its own home city.
- **"Why doesn't it plateau?"** It doesn't really — it keeps climbing steadily.
  There's no evidence yet that it's hit a ceiling by 200 labels.
- **"Why is 5-labels barely above zero-shot?"** Five examples per class is almost
  nothing — not enough to learn much beyond what the line-up already gave you.
- **"± is a confidence interval?"** No — it's the spread over repeated random
  draws of the labelled set. It tells you how much the number wobbles run to run.

## 11. Common mistakes presenting this slide

- **Reading the table like a phone book.** Say the shape first ("steady climb,
  crosses the home-city line around 50"), then let the table back you up —
  don't lead with five numbers in a row.
- **Overclaiming the plateau.** The curve does not flatten — don't say it does.
- **Forgetting the bridge to slide 5.** Without it, the audit slide looks like
  you're contradicting your own result instead of stress-testing it.

## 12. Words to avoid

| don't say | say instead |
|---|---|
| zero-shot / few-shot | with no local labels / with a few local labels |
| log-scale x-axis | the gap between points doubles each step |
| variance / standard deviation | how much the number wobbles run to run |
| "steep then flat" / "plateau" | "climbs steadily" — the data doesn't plateau |
