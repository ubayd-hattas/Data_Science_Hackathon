# Slide content — 7 slides, fits 10 minutes

Cut to seven so it runs comfortably in the slot with four speakers. Every
heading is a claim you could say out loud as the point of that slide. Plain
language on the slide; the jargon sits in **If asked**.

Numbers: zero-shot 0.36 → 0.65 · few-shot 0.66 / 0.68 / 0.70 / 0.72 / 0.74 at
5 / 25 / 50 / 100 / 200 examples per group · Madrid on itself 0.66.

---

| # | Section | Heading (the claim) | On the slide | If asked | Covers |
|---|---|---|---|---|---|
| 1 | Title & abstract | **Building age from orbit, built to travel** | title · your names · team name · abstract in a box | The whole idea in one line: a model that fails in a new city is *mis-tuned*, not broken. | abstract on slide 1 |
| 2 | The problem | **A new-city model is mis-tuned, not incapable** | • sort each satellite patch into 1 of 4 age groups<br>• train on Madrid, use in Amsterdam<br>• Amsterdam looks different — brick vs concrete, wetter, older satellites | Two shifts: the buildings *look* different, and the *mix* of ages is different. Score is macro-F1, so all four groups count equally. | understanding the challenge |
| 3 | The method | **Line the cities up first — the model does the rest** | • 40 years of readings → ~100 numbers per patch; the key one is *when* the big change happened<br>• reshape Madrid's numbers to match Amsterdam, one age group at a time<br>• no Amsterdam answers used · score jumps **0.36 → 0.65** | One row per patch (not per year) so it uses the trend. We predict Amsterdam once, use those guesses to line up each group, repeat twice. Only the given examples ever touch the model. | design + transfer plan (2 pillars) |
| 4 | A few examples | **Few examples borrow from Madrid; many trust the local data** | • 5–25 examples → mostly trust the Madrid model<br>• 200 examples → mostly trust the local one<br>• the balance shifts on its own as examples grow | Few examples = shaky local model, so lean on Madrid. We also "untangle" the features more gently when data is thin. It's a different recipe, not a smaller one. | why low-data needs its own approach |
| 5 | Results | **Fifty local examples = home-city accuracy** | • table: Madrid score + 5 Amsterdam scores, with error bars<br>• the curve: score vs number of examples<br>• 25 examples already ≈ 0.68 | Dashed line is Madrid tested on itself (0.66). Our curve crosses it near 50 examples, keeps climbing to 0.74. Steep to 50, then flat — the line-up does the work, examples add the last bit. | the required table + curve (pillar 3) |
| 6 | What we learned | **Simple alignment beat every clever alternative** | • worked: the line-up, the two-model blend, neighbour smoothing<br>• didn't: a neural network + four other tricks — all lost<br>• can't split the two oldest groups — the signal isn't in the data | Each failed idea has one clear reason (the neural net had nothing to learn — the data was already easy to separate). And ~80% of our examples sit next to a test patch — we state that openly. | design decisions, honest failures, soundness |
| 7 | Conclusion | **Cross-city transfer is a calibration problem** | 3 takeaways<br>• a new-city model is mis-tuned, not incapable<br>• ~50 examples buy home-city accuracy<br>• careful use of the data beat every fancy model<br>then: who did what — each name | Our new idea: lining the cities up *group by group* using the model's own guesses, plus the point that few vs many examples need different recipes. | originality + team |

---

## Timing — target 9:40, hard stop 10:00

| slide | minutes | speaker |
|---|---|---|
| 1 · Building age from orbit | 0:30 | A |
| 2 · Mis-tuned, not incapable | 1:15 | A |
| 3 · Line the cities up first | 2:15 | B |
| 4 · Few borrow, many trust local | 1:15 | C |
| 5 · Fifty examples = home-city | 2:00 | C |
| 6 · Simple alignment won | 1:45 | D |
| 7 · A calibration problem | 0:40 | D → all for Q&A |

**Handoffs:** A → B after slide 2 · B → C after slide 3 · C → D after slide 5.

**Slow down on slides 3 and 5** — the method and the curve. Slides 6–7 are
brisk and confident.

## Two figures to make

- Slide 3: a "before / after" scatter — the two cities' data barely overlap,
  then overlap after the line-up.
- Slide 6: four bars, score per age group — groups 3 & 4 high, groups 1 & 2 low.

Full spoken script: `docs/TALK_SCRIPT.md`.
Clickable previews: `deliverables/slides-research.html`, `deliverables/slides-preview.html`.
