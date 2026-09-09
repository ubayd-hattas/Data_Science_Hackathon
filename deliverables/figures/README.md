# Deck figures — in slide order

| slide | figure | shows |
|---|---|---|
| 1 · Problem & data | `fig_slide1_scatter.png` | single panel: 108 standardized features → 2-D, Madrid vs Amsterdam density contours + centroids sit offset (the domain shift) |
| 2 · What we did | `fig_slide2_pipeline.png` + `fig_slide2_scatter.png` | pipeline (Madrid -> line-up x2 -> predict, 0.36 -> 0.65) and the "after the line-up" panel: Madrid's cloud now sits on Amsterdam's (slide 1 showed them apart) |
| 3 · Results | `fig_slide3_curve.png` + `fig_slide3_table.png` | the curve (large) and the F1 table (booktabs) side by side |
| 4 · What did not work | `fig_slide4_didntwork.png` | five clever fixes we dropped, each with its one-line reason |
| 5 · What we audited | `fig_slide5_bars.png` | grouped bars: what the labels actually add, normal test vs labels held a map-tile away - the "held away" bar is near zero at every budget |
| 6 · Summary | (no figure) | |

## Alternatives (swap in if you prefer)

| instead of | use | why |
|---|---|---|
| `fig_slide2_scatter.png` (single "after" panel) | `alt_slide2_scatter_2panel.png` | before + after side by side, if slide 1's scatter isn't fresh in memory |
| `fig_slide3_curve.png` | `alt_slide3_curve_from_notebook.png` | the exact curve notebook 5 rendered |
| `fig_slide5_bars.png` | `fig_slide5_spatial.png` | the 4-budget line-chart version (gain vs labels, two lines) |
| `fig_slide5_bars.png` | `fig_slide5_perclass.png` | per-class F1 (15 draws): class 1 strongest, classes 2-4 near 0.7 |
| `fig_slide5_bars.png` | `alt_slide5_confusion_from_notebook.png` | 4x4 confusion matrix |

`extra_gains_chart.png` — an ascending "each step contributes" chart; only if you
add a methods slide.

Regenerate all with: `python scripts/make_slide_figures.py`
