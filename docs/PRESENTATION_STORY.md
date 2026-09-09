# Presentation — the story version (10 slides, ~10 min)

Told as a story: a model that can't move house, the fix, the climb, the dead
ends, the audit, the lesson. Anyone should follow it. Every rubric point has a
home (tagged **[R]**).

Per slide: the **heading** (engaging, plain), the **few words on the slide**, the
**visual**, and a one-line **say** for the speaker. Full script in
`docs/TALK_SCRIPT.md`.

Numbers: baseline 0.43 zero-shot / 0.42–0.67 few-shot → ours 0.65 zero-shot /
0.66 · 0.68 · 0.70 · 0.72 · 0.74 at 5 · 25 · 50 · 100 · 200 known buildings per
group. Madrid on itself: 0.66.

---

## 1 · "Can a model move to a new city?"  — 0:30 — *A*

**On slide:** title · ⟨Author A–D⟩ · Team ⟨name⟩ · abstract (149 words, small box)
**Visual:** faint city-from-above image behind the title
**Say:** "We taught a model in Madrid and asked it to work in Amsterdam. This is
what happened."
**[R]** abstract on slide 1

---

## 2 · "Madrid taught it. Amsterdam confused it."  — 1:00 — *A*

**On slide:**
- Guess a building's age group from 40 years of satellite light
- Learns in Madrid — brick, dry, one set of satellites
- Meets Amsterdam — concrete, wet, different satellites
**Visual:** two side-by-side crops, Madrid (warm) vs Amsterdam (cool), same age,
caption *same age — different look*
**Say:** "Same task, same age of building — but it *looks* different to the
satellite, and the mix of ages is different too."
**[R]** Challenge Understanding — domain shift

---

## 3 · "It wasn't stupid — it was mis-tuned"  — 0:45 — *A → B*

**On slide:**
- The rules it learned were right
- The numbers just sat in the wrong place
- Fix the tuning, not the model
**Visual:** one number, big: **zero-shot 0.43** with a small red down-arrow
**Say:** "Dropped into Amsterdam cold, it scored 0.43. Not because it can't tell
old from new — because its dials were set for Madrid."
**[R]** Challenge Understanding — the core insight

---

## 4 · "The turning point: line the cities up"  — 1:30 — *B*

**On slide:**
- Reshape Madrid's data to match Amsterdam — **one age group at a time**
- Uses the model's own first guesses · **no Amsterdam answers**
- Repeat twice
- Zero labels: **0.43 → 0.65**
**Visual:** the 3-box pipeline (Fig. 1); or the before/after scatter — two clouds
apart, then overlapping
**Say:** "We predict Amsterdam once, use those guesses to line up each group,
and retrain. That one step recovers most of the gap — with no labels at all."
**[R]** Model Design + Transfer Strategy (pillars 1 & 2); "did it address domain shift"

---

## 5 · "Then we kept climbing"  — 2:00 — *B → C*

**On slide — a rising list, each with its bump:**
- Line up the cities → **+0.07**
- Untangle the features, gently when data is thin → **+0.15 at 5 labels**
- Let a local model and the Madrid model vote together → **+0.04**
- Let neighbouring patches vote → **+0.01**
- An overnight search over 500 setups → **+0.02**
**Visual:** a small step-chart / ascending bars, baseline → final
**Say:** "Every gain came from *using* the data better — not a bigger model. And
the low-data case needed its own recipe: with 5 labels, lean on Madrid; with
200, trust the local model."
**[R]** design decisions / what worked; low-data mechanics

---

## 6 · "The things that didn't work"  — 1:15 — *C → D*

**On slide — five, one reason each:**
- Neural network — nothing to learn, data already separable
- Ordinal training — smaller mistakes, not fewer
- Self-training — teaches itself its own errors
- Gradient boosting — falls apart at 5 labels
- Fixing the age mix — needs trust it doesn't have across cities
**Visual:** five ✗ rows, plain
**Say:** "We show these on purpose. Every attempt to out-think the data lost to
using it more carefully."
**[R]** Originality — insightful failure; scientific soundness

---

## 7 · "We checked our own homework"  — 1:00 — *D*

**On slide:**
- We drew our training examples at random — as instructed
- Then measured: **~80%** sit right next to a test patch
- 30 m pixels that touch are often the same building
- So part of the score is *proximity*, not skill — **we say so**
**Visual:** a small grid — support pixels (filled) and their touching query
neighbours (outlined)
**Say:** "A teammate audited this. It doesn't break any rule and we never use
test answers — but it's honest to put the number on the slide rather than let a
judge find it."
**[R]** scientific soundness; Originality — insightful analysis

---

## 8 · "What the data will and won't tell you"  — 0:45 — *D*

**On slide:**
- Groups 3 & 4 — built during the satellite era → we can see the construction
- Groups 1 & 2 — both before 1984 → no event to see, can't be split
- That's a data limit, not a bug — the research agrees
**Visual:** 4-bar chart, score per age group — 3 & 4 tall, 1 & 2 short
**Say:** "Most of our remaining error is groups 1 and 2, and it always will be
with this data."
**[R]** F1 interpretation — limitations; what we learnt

---

## 9 · "50 buildings buy a whole city"  — 1:30 — *C*

**On slide:**
- **Table:** Madrid CV 0.66 · Amsterdam 0.66 / 0.68 / 0.70 / 0.72 / 0.74 (± SD)
- **Plot:** score vs known-buildings-per-group, error bars, dashed Madrid line
- Crosses the home-city line at ~50 · steep, then flat
**Visual:** the curve, large, centred; the table small beside it — **both required**
**Say:** "Read the jump — 0.36 to 0.65 with no labels. Then the climb. By about
50 checked buildings per group — 200 total — the transferred model does as well
on Amsterdam as any model does at home."
**[R]** the required table + plot with error bars; F1-interpretation pillar

---

## 10 · "It's a tuning problem, not a hard one"  — 0:45 — *D → all*

**On slide — three lines + names:**
- A new-city model is mis-tuned, not incapable
- ~50 known buildings buy home-city accuracy
- Careful data use beat every fancy model
- ⟨A⟩ features · ⟨B⟩ the line-up · ⟨C⟩ evaluation · ⟨D⟩ analysis
**Visual:** none — let it breathe
**Say:** "Our new bit: the group-by-group line-up, and the idea that few and many
labels need different machinery. Questions?"
**[R]** Originality — framing; Presentation — team, all four named

---

## Arc at a glance

| # | Beat | Story move |
|---|---|---|
| 1 | Can a model move cities? | hook |
| 2 | Madrid taught it, Amsterdam confused it | the setup |
| 3 | It was mis-tuned, not stupid | the diagnosis |
| 4 | Line the cities up | the turning point |
| 5 | Then we kept climbing | the rising action (**how we improved**) |
| 6 | What didn't work | the setbacks (**what didn't work**) |
| 7 | We checked our homework | the honesty beat (**what we audited**) |
| 8 | What the data won't tell you | the limit (**what we learnt**) |
| 9 | 50 buildings buy a whole city | the payoff (**results**) |
| 10 | A tuning problem, not a hard one | the moral (**conclusion**) |

## Timing & handoffs

Total ≈ 10:00. A: 1–3 · B: 4–5 · C: 5–6, 9 · D: 6–8, 10. Slow on 4, 5, 9.

## Figures to generate (I can make these as PNGs)

1. Madrid vs Amsterdam crops (slide 2)
2. before/after scatter — clouds apart → overlapping (slide 4)
3. ascending step-chart of the improvements (slide 5)
4. the F1 curve restyled to match the deck (slide 9) — base is `results/transfer_curve.png`
5. per-group score bars (slide 8)
