# Deck figures — in slide order

| slide | figure | shows |
|---|---|---|
| 1 · Problem & data | `fig_slide1_scatter.png` | Madrid vs Amsterdam feature clouds: apart, then overlapping after the line-up |
| 2 · What we did | `fig_slide2_pipeline.png` | Madrid RF -> line-up x2 -> predict (0.36 -> 0.65) |
| 3 · Results | `fig_slide3_curve.png` + `fig_slide3_table.png` | the curve (large) and the F1 table (booktabs) side by side |
| 4 · What did not work | `fig_slide4_didntwork.png` | five clever fixes we dropped, each with its one-line reason |
| 5 · What we audited | `fig_slide5_perclass.png` | per-class F1 (15 draws, error bars): class 1 strongest, classes 2-4 near 0.7 |
| 6 · Summary | (no figure) | |

## Alternatives (swap in if you prefer)

| instead of | use | why |
|---|---|---|
| `fig_slide3_curve.png` | `alt_slide3_curve_from_notebook.png` | the exact curve notebook 5 rendered |
| `fig_slide5_perclass.png` | `alt_slide5_confusion_from_notebook.png` | 4x4 confusion matrix instead of bars |

`extra_gains_chart.png` — an ascending "each step contributes" chart; only if you
add a methods slide.

Regenerate all with: `python scripts/make_slide_figures.py`
