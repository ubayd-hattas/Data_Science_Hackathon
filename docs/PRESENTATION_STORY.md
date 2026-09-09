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
- We stress-tested our few-shot result: ~half the gain is support/query
  proximity (labels held a map-tile away → +0.05-0.09 drops to ~+0.01)
- Zero-shot (0.36 → 0.65) uses no labels, so it's unaffected -- the solid result
- Corrected an earlier assumption: class 1 is our *strongest* (~0.78); 2-4 near 0.70

**> Chart:** `fig_slide5_spatial.png` -- few-shot gain over zero-shot,
random split vs labels held a map-tile away (two lines).

---

## 5 · Result: ~50 labels reach the in-city ceiling (on the standard split)

- Amsterdam: **0.66 / 0.68 / 0.70 / 0.72 / 0.74** at 5 / 25 / 50 / 100 / 200 per class
- Madrid on itself: 0.66 · baseline: 0.42 -> 0.67
- Steep to 50, then flat -- but see slide 4: much of the climb is spatial proximity

**> Table + plot (both required):** curve with error bars + dashed Madrid line (large);
Madrid + 5 Amsterdam scores, mean +/- SD (small, beside it).

---

## 6 · Conclusion -- it's a calibration problem, not a hard one

- Mis-tuned, not incapable -- the zero-shot line-up recovers most of the gap
- Local labels help most when they're near the target; ~50 buy home-city
  accuracy on the standard split
- Careful data use beat every fancier model
- New: group-by-group alignment · few vs many need different recipes · and we
  measured how much of the few-shot gain is spatial proximity

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

## Figures (in deliverables/figures/, see that README for the slide map)

slide 1 scatter · slide 2 pipeline · slide 3 curve + table ·
slide 4 didn't-work list · slide 5 spatial-block audit (`fig_slide5_spatial.png`).
