"""Emit notebooks/5-Transfer.ipynb from a list of cells."""
import json
from pathlib import Path

C = []
def md(s): C.append(("markdown", s))
def code(s): C.append(("code", s))

md(r"""# Notebook 5 — Madrid → Amsterdam Transfer

**Deliverable notebook.** Runs the final pipeline end to end with one fixed seed
and produces the two required artefacts:

1. a table of the Madrid cross-validation macro-F1 and the five Amsterdam
   few-shot macro-F1 scores (5 / 25 / 50 / 100 / 200 labels per class), each
   with an error bar;
2. the macro-F1 vs. log₂(labels per class) plot.

**Pipeline**

| stage | method |
|---|---|
| Features | temporal statistics per pixel (`src/data.py`); optional change-point and spatial-context blocks |
| Madrid reference | 5×5 repeated stratified CV, Random Forest |
| Zero-shot | CORAL-align Madrid features → Random Forest → predict Amsterdam |
| Few-shot (per budget) | adaptive-shrinkage ZCA whitening of Amsterdam features (optional PCA) → small Random Forest head → probabilities blended with the CORAL zero-shot model, weight `clip(n/beta_div, beta_floor, 0.95)` toward the local head, then anchored spatial smoothing of the predicted probabilities over each pixel's 4 map-neighbours |

The pipeline settings (feature set, forest depth, shrinkage and blend
coefficients, PCA) come from `results/overnight_best.json` when the overnight
search has produced a winner, otherwise from hand-chosen defaults. The config in
use is printed in cell 1.

Every step that touches Amsterdam uses only the **unlabelled** feature cloud plus
the small labelled support set drawn per trial — no other Amsterdam label enters
the pipeline (leakage discipline for the rubric).
""")

code(r"""import sys, time, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, "..")
from src.data import build_city
from src.adapt import make_coral, zca_whiten, spatial_smoother
from src.evaluate import (
    madrid_cv, fit_final_rf, zero_shot, few_shot_smoothed_curve,
)

SEED = 42
QUICK = False          # True -> lighter CV / fewer draws for a fast dry run
SHOTS = (5, 25, 50, 100, 200)
DATA = Path("../data")
np.random.seed(SEED)

# ── pipeline config ──────────────────────────────────────────────────────
# If the overnight search has produced a winner, use it; otherwise the
# hand-chosen defaults. This keeps the notebook the single source of truth.
BEST = Path("../results/overnight_best.json")
if BEST.exists():
    _b = json.loads(BEST.read_text())["winning_cfg"]
    FEATS = dict(changepoint=_b["features"] in ("changepoint", "both"),
                 spatial=_b["features"] in ("spatial", "both"))
    RF_KW = dict(n_estimators=_b["n_estimators"], max_depth=_b["max_depth"],
                 min_samples_leaf=_b["min_samples_leaf"],
                 max_features=_b["max_features"])
    SH_INT, SH_SLP = _b["sh_intercept"], _b["sh_slope"]
    BETA_DIV, BETA_FLOOR = _b["beta_div"], _b["beta_floor"]
    EPS, PCA_K = _b["eps"], int(_b["pca"])
    print("using tuned config from overnight_best.json:", _b["features"], "features")
else:
    FEATS = dict(changepoint=False, spatial=False)
    RF_KW = dict(n_estimators=250, max_depth=None, min_samples_leaf=1,
                 max_features="sqrt")
    SH_INT, SH_SLP = 1.44, 0.188
    BETA_DIV, BETA_FLOOR = 50.0, 0.4
    EPS, PCA_K = 1e-5, 0
    print("using hand-chosen default config")


def shrink_for(n, nf):
    eff = n * 60.0 / max(nf, 1)
    return float(np.clip(SH_INT - SH_SLP * np.log2(max(eff, 2.0)), 0.0, 1.0))


print("quick mode:", QUICK, "| features:", FEATS)
""")

md("## 1. Load features")

code(r"""t0 = time.time()
madrid = build_city(str(DATA / "madrid_train.parquet"), **FEATS)
ams    = build_city(str(DATA / "amsterdam_data.parquet"), **FEATS)
Xm, ym = madrid.X, madrid.y
Xa, ya = ams.X, ams.y
classes = np.unique(ym)
print(f"Madrid    {Xm.shape}   class mix {np.round(np.bincount(ym)[1:]/len(ym), 3)}")
print(f"Amsterdam {Xa.shape}   class mix {np.round(np.bincount(ya)[1:]/len(ya), 3)}")
print(f"{len(madrid.feature_names)} features   ({time.time()-t0:.1f}s)")
""")

md("""## 2. Madrid reference — 5×5 cross-validation

The in-city score. Repeated stratified k-fold so the error bar reflects
run-to-run variation, not just one lucky split.""")

code(r"""rf_params = dict(
    **RF_KW, class_weight="balanced", random_state=SEED, n_jobs=-1,
)
if QUICK:
    rf_params["n_estimators"] = min(rf_params["n_estimators"], 150)
cv = madrid_cv(Xm, ym, n_folds=5, n_repeats=2 if QUICK else 5,
               rf_params=rf_params, seed=SEED)
print(cv)
""")

md("""## 3. Zero-shot transfer

Train on all of Madrid, predict Amsterdam with **no** Amsterdam labels. Raw
features versus CORAL-aligned features — the difference is the value of a single
unsupervised alignment step.""")

code(r"""rf_raw = fit_final_rf(Xm, ym, rf_params=rf_params)
f1_zero_raw = zero_shot(rf_raw, Xa, ya)

coral = make_coral(Xm, Xa)                      # fitted on the two unlabelled clouds
rf_coral = fit_final_rf(coral(Xm), ym, rf_params=rf_params)
f1_zero_coral = zero_shot(rf_coral, Xa, ya)

prior_proba = rf_coral.predict_proba(Xa)        # reused by the few-shot ensemble

print(f"zero-shot  raw   {f1_zero_raw:.4f}")
print(f"zero-shot  CORAL {f1_zero_coral:.4f}")
""")

md("""## 4. Few-shot transfer curve

For each label budget *n*: draw *n* labelled Amsterdam pixels per class, whiten
the Amsterdam features with shrinkage tuned to *n*, fit a small Random Forest on
the support set, blend its probabilities with the CORAL zero-shot model, predict
the rest. Repeat with fresh draws to get an error bar.""")

code(r"""from sklearn.decomposition import PCA

def make_head():
    kw = dict(RF_KW)
    if QUICK:
        kw["n_estimators"] = min(kw["n_estimators"], 120)
    return RandomForestClassifier(**kw, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)

def whitened(n):
    "adaptive-shrink whitening for budget n, optionally PCA-compressed"
    w = zca_whiten(Xa, shrink=shrink_for(n, nf), eps=EPS)
    if PCA_K:
        p = PCA(n_components=min(PCA_K, Xa.shape[1]), random_state=SEED).fit(w(Xa))
        return lambda X: p.transform(w(X))
    return w

n_trials = 6 if QUICK else 20
nf = Xa.shape[1]

smoother = spatial_smoother(ams.pixels.to_numpy(dtype=float), k=4)

curves = {}
for n in SHOTS:
    res = few_shot_smoothed_curve(
        Xa, ya, prior_proba, make_head, smoother,
        shots=(n,), n_trials=n_trials, transform=whitened(n), seed=SEED,
        beta_div=BETA_DIV, beta_floor=BETA_FLOOR,
    )
    curves[n] = res.per_shot[n]
    print(f"  n={n:>3}  shrink={shrink_for(n, nf):.2f}  "
          f"macro-F1 {curves[n].mean():.4f} +/- {curves[n].std():.4f}")

fs_mean = np.array([curves[n].mean() for n in SHOTS])
fs_std  = np.array([curves[n].std()  for n in SHOTS])
""")

md("## 5. Deliverable table")

code(r"""rows = [("Madrid 5x5 CV", cv.mean, cv.std),
        ("Amsterdam zero-shot (raw)",   f1_zero_raw,   np.nan),
        ("Amsterdam zero-shot (CORAL)", f1_zero_coral, np.nan)]
rows += [(f"Amsterdam few-shot, {n}/class", m, s)
         for n, m, s in zip(SHOTS, fs_mean, fs_std)]

table = pd.DataFrame(rows, columns=["setting", "macro_F1", "std"])
table["macro_F1"] = table["macro_F1"].round(4)
table["std"] = table["std"].round(4)
Path("../results").mkdir(exist_ok=True)
table.to_csv("../results/deliverable_table.csv", index=False)
table
""")

md("## 6. F1 vs. log₂(labels per class)")

code(r"""fig, ax = plt.subplots(figsize=(8, 5))
x = np.array(SHOTS)
ax.errorbar(x, fs_mean, yerr=fs_std, fmt="o-", lw=2, capsize=4,
            color="#d95f0e", label="Few-shot transfer (mean ± std)")
for n in SHOTS:
    ax.scatter([n]*len(curves[n]), curves[n], color="#d95f0e", alpha=0.25, s=16, zorder=3)
ax.axhline(f1_zero_coral, ls="--", color="#3182bd", label=f"Zero-shot CORAL ({f1_zero_coral:.3f})")
ax.axhline(f1_zero_raw,   ls=":",  color="#9ecae1", label=f"Zero-shot raw ({f1_zero_raw:.3f})")
ax.axhline(cv.mean, ls="--", color="#31a354", label=f"Madrid CV ({cv.mean:.3f})")
ax.fill_between(x, cv.mean-cv.std, cv.mean+cv.std, color="#31a354", alpha=0.12)
ax.set_xscale("log", base=2); ax.set_xticks(x); ax.set_xticklabels(x)
ax.set_xlabel("Labelled Amsterdam pixels per class")
ax.set_ylabel("Macro-F1")
ax.set_title("Madrid → Amsterdam transfer")
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig("../results/transfer_curve.png", dpi=130)
plt.show()
""")

md("""## 7. Confusion matrix at 100 labels / class

Where the remaining errors sit. Rows are the true class, columns the prediction,
normalised per row.""")

code(r"""n_show = 100
tf = whitened(n_show)
Z = tf(Xa)
rng = np.random.default_rng(SEED)
support = np.concatenate([
    rng.choice(np.where(ya == c)[0], n_show, replace=False) for c in classes
])
query = np.setdiff1d(np.arange(len(ya)), support)

head = make_head().fit(Z[support], ya[support])
beta = np.clip(n_show/BETA_DIV, BETA_FLOOR, 0.95)
field = np.empty((len(ya), len(classes)))
field[query] = beta * head.predict_proba(Z[query]) + (1-beta) * prior_proba[query]
oh = np.zeros((len(support), len(classes)))
oh[np.arange(len(support)), np.searchsorted(classes, ya[support])] = 1.0
field[support] = oh
pred = classes[smoother(field)[query].argmax(1)]

cm = confusion_matrix(ya[query], pred, labels=classes).astype(float)
cm /= cm.sum(1, keepdims=True)

fig, ax = plt.subplots(figsize=(5, 4.2))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(4)); ax.set_xticklabels([f"C{c}" for c in classes])
ax.set_yticks(range(4)); ax.set_yticklabels([f"C{c}" for c in classes])
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                color="white" if cm[i,j] > 0.6 else "black")
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title(f"Amsterdam confusion, {n_show} labels/class "
             f"(macro-F1 {f1_score(ya[query], pred, average='macro'):.3f})")
fig.tight_layout(); plt.show()
""")

md("""## 8. Notes for the write-up

- **Zero-shot** raw → CORAL is the single-step domain-adaptation gain, no target
  labels used.
- **Few-shot** rises steeply to ~50 labels/class, then flattens near the Madrid
  in-city score — i.e. with ~100 labels/class the transferred model is about as
  good on Amsterdam as a model is on its home city.
- **Classes 1 and 2** carry most of the residual confusion: both pre-date the
  1984 satellite record, so there is no construction event to separate them.
  This is a data limit, consistent with the building-age literature, not a
  pipeline defect.
- Reproduce: `QUICK = False`, run top to bottom. All randomness is seeded.
""")

nb = {
    "cells": [
        {"cell_type": t, "metadata": {},
         "source": s.splitlines(keepends=True),
         **({"outputs": [], "execution_count": None} if t == "code" else {})}
        for t, s in C
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
out = Path(r"C:\Users\israe\Desktop\Hack4Dev\notebooks\5-Transfer.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, "-", len(C), "cells")
