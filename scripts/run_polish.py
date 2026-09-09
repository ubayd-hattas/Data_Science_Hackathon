"""Small polish ideas on the current pipeline. Fold in only what beats baseline
at multiple budgets for a sound reason.

Baseline = 108 feats + class-conditional CORAL prior + budget-shrinkage whitening
           + small-RF head + weighted blend + one pass of k=8 mean smoothing.

Variants:
  smooth-x2 / smooth-x3   iterate the neighbour average 2 / 3 passes
  smooth-wdist            weight neighbours by 1/(1+d) instead of a flat mean
  seed-ensemble-3         average the RF head over 3 seeds (variance down)
  conf-blend              per-pixel blend weight = f(local head confidence):
                          trust the head where its max prob is high, the prior
                          where it is low
  coral-rounds-4          class-conditional CORAL with 4 rounds instead of 2

Usage:  python scripts/run_polish.py [--quick]
"""

import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

print = functools.partial(print, flush=True)  # noqa: A001

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapt import class_conditional_coral, zca_whiten
from src.data import build_city

SEED = 42
QUICK = "--quick" in sys.argv
SHOTS = (5, 25, 50, 100, 200)
CFG = json.loads((ROOT / "results" / "overnight_best.json").read_text())["winning_cfg"]


def shrink_for(n, nf):
    eff = n * 60.0 / max(nf, 1)
    return float(np.clip(CFG["sh_intercept"] - CFG["sh_slope"] * np.log2(max(eff, 2.0)), 0, 1))


def rf_head(seed=SEED, n_est=None):
    return RandomForestClassifier(
        n_estimators=n_est or (200 if QUICK else CFG["n_estimators"]),
        max_depth=CFG["max_depth"], min_samples_leaf=CFG["min_samples_leaf"],
        max_features=CFG["max_features"], class_weight="balanced",
        random_state=seed, n_jobs=-1)


def main():
    t0 = time.time()
    n_trials = 6 if QUICK else 15
    fb = dict(changepoint=True, spatial=True)
    m = build_city(str(ROOT / "data" / "madrid_train.parquet"), **fb)
    a = build_city(str(ROOT / "data" / "amsterdam_data.parquet"), **fb)
    Xm, ym, Xa, ya = m.X, m.y, a.X, a.y
    nf, classes = Xa.shape[1], np.unique(ym)
    coords = a.pixels.to_numpy(dtype=float)
    tree = cKDTree(coords)
    nn = tree.query(coords, k=9)                       # (dist, idx), self + 8
    d8, i8 = nn[0][:, 1:], nn[1]                       # i8 includes self at col 0
    w8 = 1.0 / (1.0 + np.column_stack([np.zeros(len(coords)), d8]))
    w8 = w8 / w8.sum(1, keepdims=True)

    def smooth_mean(P, passes=1):
        for _ in range(passes):
            P = P[i8].mean(axis=1)
        return P

    def smooth_wdist(P):
        return np.einsum("nkc,nk->nc", P[i8], w8)

    def prior_for(rounds):
        rf = class_conditional_coral(
            Xm, ym, Xa, lambda: RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=SEED, n_jobs=-1), rounds=rounds)
        return rf.predict_proba(Xa)

    prior2 = prior_for(2)
    prior4 = prior_for(4)
    print(f"class-cond CORAL zero-shot: 2r "
          f"{f1_score(ya, classes[prior2.argmax(1)], average='macro'):.4f}   4r "
          f"{f1_score(ya, classes[prior4.argmax(1)], average='macro'):.4f}\n")

    variants = ["baseline", "smooth-x2", "smooth-x3", "smooth-wdist",
                "seed-ens-3", "conf-blend", "coral-4r"]
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

            p_head = rf_head().fit(Z[sup], ya[sup]).predict_proba(Z)
            base_field = beta * p_head + (1 - beta) * prior2

            def sc(field):
                return f1_score(ya[q], classes[field[q].argmax(1)],
                                average="macro", zero_division=0)

            acc["baseline"].append(sc(smooth_mean(base_field, 1)))
            acc["smooth-x2"].append(sc(smooth_mean(base_field, 2)))
            acc["smooth-x3"].append(sc(smooth_mean(base_field, 3)))
            acc["smooth-wdist"].append(sc(smooth_wdist(base_field)))

            # seed ensemble on the head
            p_ens = np.mean([rf_head(seed=s).fit(Z[sup], ya[sup]).predict_proba(Z)
                             for s in (SEED, SEED + 1, SEED + 2)], axis=0)
            acc["seed-ens-3"].append(sc(smooth_mean(
                beta * p_ens + (1 - beta) * prior2, 1)))

            # confidence-weighted blend: w = clip(max_prob scaled)
            conf = p_head.max(1, keepdims=True)
            w = np.clip((conf - 0.4) / 0.4, 0.15, 0.95)
            acc["conf-blend"].append(sc(smooth_mean(
                w * p_head + (1 - w) * prior2, 1)))

            acc["coral-4r"].append(sc(smooth_mean(
                beta * p_head + (1 - beta) * prior4, 1)))

        for v in variants:
            res[v][n] = float(np.mean(acc[v]))
        print(f"  n={n:>3}  " + "  ".join(f"{v[:9]}:{res[v][n]:.3f}" for v in variants))

    print("\n" + "=" * 84)
    print(f"{'variant':>14} | " + " ".join(f"{s:>7}" for s in SHOTS) + " |  mean   dMean")
    print("-" * 84)
    bm = np.mean([res["baseline"][s] for s in SHOTS])
    for v in variants:
        r = np.array([res[v][s] for s in SHOTS])
        print(f"{v:>14} | " + " ".join(f"{x:>7.4f}" for x in r)
              + f" | {r.mean():.4f}  {r.mean()-bm:+.4f}")

    (ROOT / "results" / "polish_scores.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote results/polish_scores.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
