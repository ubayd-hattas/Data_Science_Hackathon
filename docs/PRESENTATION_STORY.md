# Slide content — 6 slides, ~10 min

Only what goes on the slides. Spoken lines: `docs/TALK_SCRIPT.md`.

---

## 1 · A model trained in Madrid, moved to Amsterdam

- 4 age classes per satellite patch · score: macro-F1
- Madrid model on Madrid -> on Amsterdam: **0.66 -> 0.43**
- Different look, different age mix

**> Add:** the **149-word abstract** in a small box, team names *(abstract required on slide 1)*.

---

## 2 · The fix: line the cities up first

- Reshape Madrid's data to match Amsterdam, one age class at a time
- Uses the model's own guesses · no Amsterdam labels · repeat x2
- Zero labels: **0.36 -> 0.65**

**> Picture:** pipeline diagram + before/after scatter (clouds apart, then overlapping).

---

## 3 · Then small, measured gains

- Untangle features, gently when data is thin -> **+0.15 at 5 labels**
- Two models vote · neighbours vote · 500-setup search
- Few labels != full pipeline with less data -- a different recipe

**> Chart:** ascending step-chart -- baseline -> each gain stacked -> final.

---

## 4 · What didn't work, what we won't claim

- Neural net, ordinal, self-training, boosting -- all lost to simpler fixes
- Classes 1 & 2 (pre-1984) can't be split -- a data limit
- ~80% of examples sit next to a test patch -- we disclose it

**> Chart:** 4 bars, score per age class -- 3 & 4 tall, 1 & 2 short.

---

## 5 · Result: ~50 labels reach the in-city ceiling

- Amsterdam: **0.66 / 0.68 / 0.70 / 0.72 / 0.74** at 5 / 25 / 50 / 100 / 200 per class
- Madrid on itself: 0.66 · baseline: 0.42 -> 0.67
- Steep to 50, then flat

**> Table + plot (both required):** curve with error bars + dashed Madrid line (large);
Madrid + 5 Amsterdam scores, mean +/- SD (small, beside it).

---

## 6 · Conclusion -- it's a calibration problem, not a hard one

- Mis-tuned, not incapable
- ~50 labels buy home-city accuracy
- Careful data use beat every fancier model
- New: group-by-group alignment · few vs many need different recipes

**> Add, small:** <A> features · <B> alignment · <C> evaluation · <D> analysis.

---

## Timing -- total ~= 10:00

| speaker | slides | min |
|---|---|---|
| A | 1, 2 | 2:30 |
| B | 3 | 2:00 |
| C | 4, 5 | 3:30 |
| D | 6 + Q&A | 1:30 |

Slow on 2 and 5.

## Figures to make (PNGs from the project data, on request)

slide 2 before/after scatter · slide 3 gains step-chart · slide 4 per-class bars ·
slide 5 restyled curve (base: results/transfer_curve.png).
