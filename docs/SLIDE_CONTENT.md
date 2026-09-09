# Slide content — 8 slides, ~10 minutes

Plain language. The **Section** column is the research-style label (what a paper
would call it). **On the slide** stays short — 3 lines, no jargon. **If asked**
is what the speaker can explain in Q&A, still in plain words. **Covers** is the
rubric item.

Numbers are current: zero-shot 0.36 → 0.65; few-shot 0.66 / 0.68 / 0.70 / 0.72 /
0.74 at 5 / 25 / 50 / 100 / 200 examples per group; Madrid on itself 0.66.

---

| # | Section | Slide heading | On the slide | If asked | Covers |
|---|---|---|---|---|---|
| 1 | Title & abstract | **Guessing building age from space — in a city we've never studied** | title · your names · team name · the abstract in a box | The whole idea in one line: a model that fails in a new city is *mis-tuned*, not broken. | abstract on slide 1 |
| 2 | The problem | **The same model doesn't work in a new city** | • sort each satellite patch into 1 of 4 age groups<br>• train on Madrid, use in Amsterdam<br>• Amsterdam looks different — brick vs concrete, wetter, older satellites | Two reasons it shifts: the buildings *look* different, and the *mix* of ages is different. We score with macro-F1, so all 4 groups matter equally. | understanding the challenge |
| 3 | The data | **We describe each patch by how it changed, not how it looks** | • 40 years of satellite readings, 6 colours<br>• turned into ~100 numbers per patch<br>• the key one: *when* the biggest change happened — that's when it was built | One row per patch, not one per year, so the model has to use the trend. We picked features that mean the same thing in any city. | our design choices |
| 4 | The method | **Our fix: reshape Madrid's data to look like Amsterdam — one age group at a time** | • train a model on Madrid<br>• line the two cities' numbers up, group by group<br>• no Amsterdam answers used · score jumps **0.36 → 0.65** | We predict Amsterdam once, use those guesses to line up each group, then repeat. Only the few examples we're given ever touch the model — nothing else. | our design + transfer plan |
| 5 | A few examples | **A few local examples do the rest — "few" needs a different recipe than "many"** | • 5–25 examples → mostly trust the Madrid model<br>• 200 examples → mostly trust the local one<br>• the balance shifts on its own as examples grow | With very few examples the local model is shaky, so we lean on Madrid. We also "untangle" the features more gently when data is thin. | why low-data needs its own approach |
| 6 | Results | **With 50 examples per group, it matches a home-city model** | • table: Madrid score + 5 Amsterdam scores, with error bars<br>• the curve: score vs number of examples<br>• 25 examples already ≈ 0.68 | The dashed line is Madrid tested on itself (0.66). Our curve crosses it near 50 examples. Steep climb to 50, then flat — the line-up does the work, examples add the last bit. | the required table + curve |
| 7 | What we learned | **What worked, what didn't, and what we won't claim** | • worked: the line-up, the blend, the neighbour smoothing<br>• didn't: a neural network + 4 other tricks — all lost to simpler fixes<br>• can't split the two oldest groups — the signal isn't in the data | Each failed idea has one reason (e.g. the neural net had nothing to learn — the data was already easy to separate). Also: ~80% of our examples sit next to a test patch — we say so openly. | design decisions, honest failures, soundness |
| 8 | Conclusion | **It's a calibration problem, not a hard one** | 3 takeaways<br>• a new-city model is mis-tuned, not incapable<br>• ~50 examples buy home-city accuracy<br>• careful use of the data beat every fancy model<br>then: who did what — each name | Our new idea: lining up the cities *group by group* using the model's own guesses, and the point that few vs many examples need different recipes. | originality + team |

---

## Timing (aim for ≈ 9:50)

| slide | minutes | speaker |
|---|---|---|
| 1 Title | 0:30 | A |
| 2 The problem | 1:15 | A |
| 3 The data | 1:00 | B |
| 4 The method | 1:45 | B |
| 5 A few examples | 1:15 | C |
| 6 Results | 2:00 | C |
| 7 What we learned | 1:30 | D |
| 8 Conclusion | 0:35 | D |

Full spoken script: `docs/TALK_SCRIPT.md`.
Clickable previews: `deliverables/slides-research.html`, `deliverables/slides-preview.html`.
Two figures still to make: the before/after "cities line up" scatter (slide 4)
and the per-group score bars (slide 7).
