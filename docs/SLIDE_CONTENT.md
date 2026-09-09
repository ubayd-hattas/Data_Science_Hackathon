# Slide structure — minimum words, visual-first

7 slides, ~10 minutes. Rule: the **heading is the sentence**; the slide shows
**one visual** and almost no text. Everything else is spoken —
see `docs/TALK_SCRIPT.md`.

Word budget per slide (not counting the heading and the figure): **≤ 12 words.**

Numbers: zero-shot 0.36 → 0.65 · few-shot 0.66 / 0.68 / 0.70 / 0.72 / 0.74 at
5 / 25 / 50 / 100 / 200 per group · Madrid on itself 0.66.

---

## Slide 1 — Title

**Heading:** Building age from orbit, built to travel

**On the slide:**
- Title (the heading)
- ⟨Author A⟩ · ⟨Author B⟩ · ⟨Author C⟩ · ⟨Author D⟩ — Team ⟨name⟩
- Abstract (149 words) in a **small** box, bottom third — required on slide 1,
  but sized so the audience's eye goes to the title, not the paragraph.

**Visual:** none needed. Optional: one faint full-bleed satellite/aerial image
of a European city behind the text at ~15% opacity.

**Spoken:** the one-line idea — "a model that fails in a new city is mis-tuned,
not broken."

---

## Slide 2 — The problem

**Heading:** A new-city model is mis-tuned, not incapable

**On the slide (≤ 10 words):**
- `4 age groups` · `train Madrid → test Amsterdam` · `score: macro-F1`

**Visual — the centrepiece:** two image panels side by side.
- Left: a Landsat / aerial crop of a Madrid block, warm tint, label **MADRID**
- Right: the same for Amsterdam, cool tint, label **AMSTERDAM**
- Caption under both: *same building age — different fingerprint*
- If no real imagery: two large colour swatches (amber / teal) with the labels.
  A real crop is better — ask if the team has any RGB tiles, or I can render a
  false-colour thumbnail from the feature arrays.

**Spoken:** the two shifts — look different *and* age mix different (oldest group
1.6× more common in Amsterdam).

---

## Slide 3 — The method

**Heading:** Line the cities up first — the model does the rest

**On the slide (labels only):**
- A 3-box pipeline: `Madrid → Random Forest` → `line up, group by group ×2` → `predict Amsterdam`
- One big number under it: **0.36 → 0.65**  (small label: *zero labels*)

**Visual — pick one, or both if they fit:**
1. **Pipeline diagram** (the 3 boxes above) — vertical or horizontal, arrows,
   the accent colour on the "line up" box. This is *Fig. 1*.
2. **Before / after scatter** (strongest single figure): two small 2-D plots.
   Left — Madrid points and Amsterdam points barely overlap. Right — after the
   line-up, they overlap. Caption: *the line-up closes the gap.*
   I can generate this from the feature data (PCA projection) as a PNG.

**Spoken:** one row per patch (not per year) → uses the trend; the "when did it
change" feature; pseudo-labels, twice, no answers used; leakage discipline.

---

## Slide 4 — A few examples

**Heading:** Few examples borrow from Madrid; many trust the local data

**On the slide (≤ 8 words):**
- A single horizontal bar / gradient, labelled at the ends:
  `5–25 → lean on Madrid` ————————→ `200 → trust local`
- optional tiny caption: *balance shifts automatically*

**Visual:** just that one bar. No table, no list. If you want a second element,
a 2-frame sketch: thin support set → "shaky", large support set → "solid".

**Spoken:** why (small local model is unstable); gentler feature untangling when
data is thin; it's a *different* recipe, not a smaller one.

---

## Slide 5 — Results

**Heading:** Fifty local examples = home-city accuracy

**Visual — the star of the deck: the curve.** Large, centred.
- Source: `results/transfer_curve.png` (or the cleaner SVG in
  `deliverables/slides-research.html`).
- Shows: score vs examples per group (log x), error bars, dashed line = Madrid
  in-city score (0.66).
- One short caption: *crosses the home-city line at ~50 · steep, then flat.*

**Supporting — the table**, small, beside or below the curve (booktabs style:
rule above header, rule below header, rule at bottom, no vertical lines):

| Setting | Macro-F1 |
|---|---:|
| No labels, raw | 0.36 |
| No labels, lined up | **0.65** |
| 5 / group | **0.66** ±.00 |
| 25 / group | **0.68** ±.01 |
| 50 / group | **0.70** ±.01 |
| 100 / group | **0.72** ±.01 |
| 200 / group | **0.74** ±.01 |
| *Madrid, on itself* | *0.66 ±.00* |

*One line under the table:* provided-notebook baseline 0.42 → 0.67.

**Spoken:** read the jump 0.36 → 0.65; walk the five budgets; "the line-up does
the work, the labels buy the last points"; error bars are tiny so the order is
real.

---

## Slide 6 — What we learned

**Heading:** Simple alignment beat every clever alternative

**On the slide — two short columns, ~3 words each line:**

| ✓ Worked | ✗ Didn't |
|---|---|
| Line up the cities | Neural network |
| Blend two models | Ordinal training |
| Smooth over neighbours | Self-training · boosting · mix-fix |

**Visual — optional but strong:** a 4-bar chart, score per age group. Groups 3
& 4 tall, groups 1 & 2 short. Caption: *classes 1 & 2 (pre-1984) can't be
separated — a data limit.* I can generate this PNG.

**One-line footer:** ~80% of examples sit next to a test patch — we disclose it.

**Spoken:** one reason per failed idea (neural net: nothing to learn — already
separable); the data limit and why; the adjacency caveat.

---

## Slide 7 — Conclusion

**Heading:** Cross-city transfer is a calibration problem

**On the slide — three lines, nothing else:**
1. Mis-tuned, not incapable
2. ~50 examples buy home-city accuracy
3. Careful data use > fancy models

Then, small: ⟨A⟩ features · ⟨B⟩ the line-up · ⟨C⟩ evaluation · ⟨D⟩ analysis

**Visual:** none. Let the three lines breathe.

**Spoken:** the takeaways; the "what's new" (group-by-group line-up + few-vs-many
needs different machinery); "questions?"

---

## Figures to make (I can generate all three as PNGs)

| for slide | figure | source |
|---|---|---|
| 2 | Madrid vs Amsterdam image panels (false-colour crops) | feature arrays / raw tiles |
| 3 | before / after scatter — clouds apart, then overlapping | PCA of the 108 features |
| 5 | the F1-vs-examples curve | already have `results/transfer_curve.png`; can restyle to match the deck |
| 6 | per-group score bars | one evaluation run |

## Timing & speakers

| slide | min | speaker |
|---|---|---|
| 1 Title | 0:30 | A |
| 2 The problem | 1:15 | A |
| 3 The method | 2:15 | B |
| 4 A few examples | 1:15 | C |
| 5 Results | 2:00 | C |
| 6 What we learned | 1:45 | D |
| 7 Conclusion | 0:40 | D → Q&A |

Handoffs: A→B after 2 · B→C after 3 · C→D after 5. Slow down on 3 and 5.
