"""Does Ubayd's logistic-regression head beat our small-RF head on OUR pipeline?

His frozen results show plain logistic regression on the support set clearly
beats nearest-prototype and his two-tier RF at n>=25. We use a small Random
Forest head. This tries, on the *current* pipeline (108 feats, class-conditional
CORAL prior, budget-shrinkage whitening, prediction-only smoothing):

  head:  small RF            (current)
         logistic regression (his best local learner)
         logreg + RF average (probability mean)

and, orthogonally, the blend rule:
  combine:  weighted sum   beta*head + (1-beta)*prior   (current)
            score product  (head * prior) renormalised  (his low-budget branch)

Only fold a change in if it beats the current pipeline at multiple budgets and
the reason is sound.

Usage:  python scripts/run_head_swap.py [--quick]
"""

import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

print = functools.partial(print, flush=True)  # noqa: A001

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapt import (
    class_conditional_coral, spatial_smoother, zca_whiten,
)
from src.data import build_city
from src.evaluate import fit_final_rf

SEED = 42
QUICK = "--quick" in sys.argv
SHOTS = (5, 25, 50, 100, 200)
CFG = json.loads((ROOT / "results" / "overnight_best.json").read_text())["winning_cfg"]


def shrink_for(n, nf):
    eff = n * 60.0 / max(nf, 1)
    return float(np.clip(CFG["sh_intercept"] - CFG["sh_slope"] * np.log2(max(eff, 2.0)), 0, 1))


def rf_head():
    return RandomForestClassifier(
        n_estimators=200 if QUICK else CFG["n_estimators"], max_depth=CFG["max_depth"],
        min_samples_leaf=CFG["min_samples_leaf"], max_features=CFG["max_features"],
        class_weight="balanced", random_state=SEED, n_jobs=-1)


def lr_head():
    return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")  # noqa


HEADS = {"rf": rf_head, "logreg": lr_head}


def main():
    t0 = time.time()
    n_trials = 6 if QUICK else 15
    fb = dict(changepoint=True, spatial=True)
    m = build_city(str(ROOT / "data" / "madrid_train.parquet"), **fb)
    a = build_city(str(ROOT / "data" / "amsterdam_data.parquet"), **fb)
    Xm, ym, Xa, ya = m.X, m.y, a.X, a.y
    nf, classes = Xa.shape[1], np.unique(ym)
    smoother = spatial_smoother(a.pixels.to_numpy(dtype=float), k=8)

    rf_cc = class_conditional_coral(
        Xm, ym, Xa, lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        rounds=2)
    prior = rf_cc.predict_proba(Xa)
    print(f"class-cond CORAL zero-shot: "
          f"{f1_score(ya, classes[prior.argmax(1)], average='macro'):.4f}\n")

    variants = ["rf/sum (current)", "logreg/sum", "rf+logreg/sum",
                "rf/product", "logreg/product"]
    res = {v: {} for v in variants}

    for n in SHOTS:
        Z = zca_whiten(Xa, shrink=shrink_for(n, nf), eps=CFG["eps"])(Xa)
        beta = float(np.clip(n / CFG["beta_div"], CFG["beta_floor"], 0.95))
        rng = np.random.default_rng(SEED)
        acc = {v: [] for v in variants}
        for _ in range(n_trials):
            sup = np.concatenate([
                rng.choice(np.where(ya == c)[0], min(n, (ya == c).sum()),
                           replace=False) for c in classes])
            q = np.setdiff1d(np.arange(len(ya)), sup)

            p_rf = rf_head().fit(Z[sup], ya[sup]).predict_proba(Z)
            p_lr = lr_head().fit(Z[sup], ya[sup]).predict_proba(Z)
            p_avg = 0.5 * (p_rf + p_lr)

            def score(field):
                return f1_score(ya[q], classes[smoother(field)[q].argmax(1)],
                                average="macro", zero_division=0)

            def prod(p):
                w = np.clip(p, 1e-6, None) * np.clip(prior, 1e-6, None)
                return w / w.sum(1, keepdims=True)

            acc["rf/sum (current)"].append(score(beta * p_rf + (1 - beta) * prior))
            acc["logreg/sum"].append(score(beta * p_lr + (1 - beta) * prior))
            acc["rf+logreg/sum"].append(score(beta * p_avg + (1 - beta) * prior))
            acc["rf/product"].append(score(prod(p_rf)))
            acc["logreg/product"].append(score(prod(p_lr)))

        for v in variants:
            res[v][n] = float(np.mean(acc[v]))
        print(f"  n={n:>3}  " + "  ".join(f"{v.split('/')[0][:6]}:{res[v][n]:.3f}"
                                          for v in variants))

    print("\n" + "=" * 82)
    print(f"{'variant':>18} | " + " ".join(f"{s:>7}" for s in SHOTS) + " |  mean   dMean")
    print("-" * 82)
    base = np.array([res["rf/sum (current)"][s] for s in SHOTS])
    bm = base.mean()
    for v in variants:
        r = np.array([res[v][s] for s in SHOTS])
        print(f"{v:>18} | " + " ".join(f"{x:>7.4f}" for x in r)
              + f" | {r.mean():.4f}  {r.mean()-bm:+.4f}")

    (ROOT / "results" / "head_swap_scores.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote results/head_swap_scores.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
