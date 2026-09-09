# Amsterdam support/query spatial-adjacency audit

Cross-check contribution, verifying whether the few-shot support/query split
carries the same spatial-leakage risk already known to affect naive random CV
on Madrid (adjacent 30 m pixels are very likely the same building/block).

## What was measured

For each of the five required label budgets (5, 25, 50, 100, 200 per class),
across 10 random trials each: what fraction of the drawn **support** pixels
have at least one orthogonally-adjacent pixel (`px_key`/`py_key` neighbour)
that landed in the **query** set for that same trial.

Script: `src/amsterdam_split_leakage.py`. Output: `outputs/amsterdam_split_leakage.json`.

## Result

| Labels/class | % of support pixels with ≥1 immediate neighbour in query | across 10 trials |
|---:|---:|---:|
| 5   | 81.0% | ± 7.7pp |
| 25  | 79.1% | ± 3.4pp |
| 50  | 80.1% | ± 2.8pp |
| 100 | 79.5% | ± 2.5pp |
| 200 | 78.4% | ± 1.9pp |

Remarkably stable across every budget — roughly 4 in 5 support pixels sit next
to at least one query pixel, regardless of how many labels are drawn.

## What this does and doesn't mean

- **Does not** touch query labels, and does not contradict this pipeline's own
  leakage discipline (CORAL/whitening statistics come from the unlabelled
  Amsterdam pool; the only labelled data entering the model is the support set
  itself). No rule violation here.
- **Does** mean that part of the reported few-shot F1 at every budget may
  reflect a support pixel and its spatially-adjacent query pixel being the
  same building (or the same construction-era block) rather than the model
  genuinely generalising across independent locations — the same mechanism
  that inflates naive random CV, just on the target-city adaptation step
  instead of the source-city training step.
- Worth reporting in the justification's F1-interpretation section as an
  explicit, known limitation rather than an implicit assumption — matches the
  standard already applied to Madrid CV elsewhere in this project.

## Suggested follow-up (not performed here, to avoid re-litigating a frozen result)

A spatial-block-disjoint support/query split (bin pixels into ~500 m blocks,
draw support and query from different blocks entirely) would give a stricter,
geography-independent estimate — analogous to the spatial-block `GroupKFold`
already recommended for Madrid CV. Not run against the frozen pipeline here
since the required evaluation protocol is organiser-specified random per-class
sampling, not block-disjoint sampling; changing it would mean scoring a
different procedure than the one being judged. Flagged as a caveat, not a fix.
