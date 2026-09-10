# Slide 6 — It's a calibration problem, not a hard one

**Speaker:** all (each says their line) · **Time:** 0:40 → Q&A · **Figure:** none

---

## 0. Layout

```
┌──────────────────────────────────────────────────────────────┐
│   It's a calibration problem, not a hard one          ← TITLE │
│                                                              │
│   • A new-city model is mis-tuned, not incapable      ← 3     │
│                                                        BULLETS│
│   • The reusable win: label-free group-by-group               │
│     alignment — 0.36 → 0.65                                   │
│                                                              │
│   • Careful use of the data beat every fancier model         │
│                                                              │
│   New here: per-group alignment · few vs many          ← small│
│   labels need different machinery                       line 1│
│                                                              │
│   A — features · B — the line-up · C — evaluation      ← small│
│   D — analysis                                          line 2│
└──────────────────────────────────────────────────────────────┘
```

No figure. Four zones top to bottom: **title** (the thesis, stated plainly —
not the word "summary"), **three bullets** (big text, the takeaways), **small
line 1** (what's original — the rubric scores this explicitly), **small line
2** (who did what — the rubric checks every member is credited and spoke).

**Check your team name is consistent** with slide 1 — don't let one slide say
"F1DEVS UNLIMITED" and another say a different team name.

## 1. What this earns on the rubric

- **Team roles line = Cohesion.** "Is it evident all team members contributed
  meaningfully... did they all speak?" is graded directly off this line plus
  who actually said something during the talk. If a name is on the slide but
  that person never spoke, it's worse than not listing roles at all.
- **"New here" line = restates Originality** one more time at the point judges
  are forming their overall impression — the last thing said is disproportionately
  memorable, so this is a second, cheap shot at the Originality criterion.
- **The close overall = Presentation & Collaboration.** A crisp three-point
  recap that doesn't introduce new numbers reads as "communicated complex
  concepts clearly within the time limit" — rambling or adding a new claim here
  undercuts that.

## 2. What this slide says

The takeaways. A model in a new city isn't incapable — it's mis-tuned. The
reusable win is the **label-free line-up** (0.36 → 0.65). Careful use of the
data beat every fancier model. It's a recap: **introduce no new numbers here.**

## 3. In plain words

- **The headline.** A working model that fails in a new city usually just needs
  its number-ranges adjusted — and you can do that with **no local labels**.
- **Lead with the line-up, not the label count.** An earlier draft of this slide
  closed on "50 labels = home-city accuracy" — but slide 5 just showed that half
  of that gain is proximity. Ending the whole talk on the most-caveated number is
  the wrong note. Close on the thing with **no caveat**: the zero-label line-up.
- **What's genuinely new from us:**
  1. Doing the adjustment **one age group at a time**, using the model's own
     guesses.
  2. The finding that **"a few labels" and "many labels" need different
     machinery** — not the same pipeline scaled down.
- **Team roles** are on the slide so it's clear who did what — the rubric checks
  this and checks that every member speaks.

## 4. The numbers on this slide

None new. It's a recap — do **not** introduce a number here that wasn't on an
earlier slide.

## 5. Bullets + footer lines, exactly as they go on the slide

```
• A new-city model is mis-tuned, not incapable
• The reusable win: label-free group-by-group alignment — 0.36 → 0.65
• Careful use of the data beat every fancier model
```
```
New here: per-group alignment · few vs many labels need different machinery

A — features · B — the line-up · C — evaluation · D — analysis
```

## 6. Timing breakdown (0:40 total, split across the team)

| segment | time | who |
|---|---|---|
| bullet 1 | 0:10 | A or whoever opens |
| bullet 2 | 0:10 | B |
| bullet 3 | 0:10 | C or D |
| roles + "questions?" | 0:10 | whoever closes |

Forty seconds is short by design — this is a landing, not a re-pitch. If every
speaker says one bullet, it also visibly demonstrates "all team members spoke"
right at the end, which is graded.

## 7. What to actually say (~35 s, split across the team)

> "Three things to take away.
>
> One — a model in a new city is mis-tuned, not incapable. Lining the cities up
> recovers most of the gap with zero labels.
>
> Two — that label-free line-up, 0.36 to 0.65, is the result we'd stand behind
> anywhere. Labels help too, but slide five showed some of that is proximity.
>
> Three — careful use of the data beat every fancier model we tried.
>
> Our original contribution is the group-by-group line-up driven by the model's
> own guesses, and the framing that few and many labels need different machinery.
>
> Roles are on the slide. Happy to take questions."

## 8. Handoff — into Q&A

> "Roles are on the slide. Happy to take questions."

Whoever says this should already know who's answering what kind of question —
agree beforehand roughly who takes method questions vs numbers questions vs
"what would you do next" questions, so nobody stalls looking at each other.

## 9. If you're running short — cut to this

One sentence, one speaker: *"A model that fails in a new city is mis-tuned, not
broken — the fix needs no local labels, and careful data use beat every
fancier model we tried. Questions?"* Keep the roles line even here — it's the
cheapest point on the whole rubric and costs five seconds.

## 10. If someone asks

- **"What's the single most reusable result?"** The no-label line-up: 0.36 → 0.65
  with zero Amsterdam labels. It has no caveats.
- **"Would this work for city X?"** That's exactly the third-city test we'd run
  next. The mechanism — align, then add labels — isn't Madrid-specific.
- **"If you had one more week?"** Spatial-block-disjoint evaluation as the
  default, and a third city.
- **"What's the biggest risk or weakness in this method?"** That the ~80%
  adjacency effect might be larger in a real deployment than in this dataset —
  we've quantified it here, but a genuinely new city with no nearby labelled
  neighbourhoods at all is the untested edge case.
- **"How would you actually deploy this?"** Run the zero-shot line-up first for
  full city-wide coverage, then spend any labelling budget on a spatially
  spread-out sample rather than a convenient cluster — that's the direct
  practical lesson from slide 5.
- **"What was the hardest part of the project?"** Probably deciding when a
  result was real versus an artefact of how the data was sampled — that's what
  led to the spatial-block audit in the first place.
- **"Which of your results are you most confident will hold up under scrutiny?"**
  The zero-shot 0.36 → 0.65 number — it uses no labels at all, so there's no
  sampling quirk that could be inflating it.

## 11. Common mistakes presenting this slide

- **Introducing a new claim or number here.** It reads as disorganised rather
  than thorough this late in the talk. Everything on this slide should already
  be familiar from an earlier slide.
- **One person delivering the whole slide.** Split it — even one bullet each —
  so "all team members spoke" is unambiguous to the judges.
- **Forgetting to check the team name matches slide 1.**

## 12. Words to avoid

| don't say | say instead |
|---|---|
| calibration (unexplained) | the model's number-ranges being off |
| generalisation gap | how much worse it does in a new city |
| few-shot regime | the case where you only have a handful of labels |
