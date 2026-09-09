# Slide 1 — A model that fails in a new city is mis-tuned, not broken

**Speaker:** A · **Time:** 1:30 · **Figure:** `fig_slide1_scatter.png`
**Subheading on the slide:** *Train in Madrid, deploy in Amsterdam — same model, 0.66 → 0.43*

---

## 1. What this slide says

We predict how old buildings are from satellite data. We train the model in
Madrid and want it to work in Amsterdam. The **exact same model** scores 0.66 in
Madrid but only 0.43 in Amsterdam. That drop isn't because the model is bad —
the two cities *look* different to the satellite, so the model's number-ranges
land in the wrong place. The rest of the talk is how we fix that.

## 2. In plain words

- The satellite doesn't take photos. It measures **how much light of six
  colours bounces off each 30-metre patch of ground**, once a year, for about
  40 years. New roofs, weathering and grime change that bounce pattern over time.
- We boil each patch's 40-year history down to **108 summary numbers**
  ("features"): averages, how bumpy the year-to-year readings are, the biggest
  sudden jump and *when* it happened, and what the neighbouring patches look like.
  Those numbers are all the model ever sees.
- The score is **macro-F1**: 0 = useless, 1 = perfect. All four age groups count
  equally, so you can't win by only getting the common ones right.
- **The picture:** each dot is one patch, with its 108 numbers squashed down to
  two dimensions so we can see them. Madrid dots and Amsterdam dots sit in
  **different regions** — same measurement, different city, different readings.
  The "X" marks the centre of each city. That offset is the whole problem.

## 3. The numbers on this slide

| number | meaning |
|---|---|
| 0.66 | Madrid model tested on Madrid (a model doing its home city) |
| 0.43 | the same Madrid model dropped straight onto Amsterdam, no adjustment |
| 4 | age classes (construction-era bands, oldest to newest) |
| 108 | features per patch |
| 5–200 | labelled Amsterdam patches per class the challenge lets us use |

## 4. What to actually say (~40 s)

> "The task: look at a satellite's view of a 30-metre patch of city and say
> which of four age bands its buildings are in. We're scored on macro-F1, so all
> four bands matter equally.
>
> We have labels for Madrid. We want it to work in Amsterdam with almost no local
> labels. And here's the problem — [point at the plot] — the same features, the
> two cities, sit in different places. The model learned its rules on the Madrid
> cloud. Drop it on Amsterdam unchanged and it goes from 0.66 to 0.43.
>
> That's not a broken model. It's mis-tuned. And you can fix most of it with no
> local labels — that's what the next slide is about."

## 5. If someone asks

- **"Why not just retrain on Amsterdam?"** We barely have Amsterdam labels —
  that's the point of the challenge. The method has to work with 5 to 200 per
  class, sometimes zero.
- **"What are the four classes exactly?"** Four construction-era bands; the cut
  years are in the challenge's own notebook. Class 1 oldest, class 4 newest.
- **"Why satellite and not a building register?"** Registers don't exist for
  every city. A method that reads satellite data can be pointed anywhere.
- **"What's in the 108 numbers?"** Averages and wobble per colour, the size and
  timing of the biggest year-to-year jump (that's the construction event), and
  the same stats averaged over the 8 nearest patches.

## 6. Words to avoid

| don't say | say instead |
|---|---|
| domain shift / covariate shift | the two cities look different to the satellite |
| feature space | the 108 numbers / where the patches land on the plot |
| distribution | the spread of readings for a city |
