"""Two more untried ideas on top of the tuned + smoothed pipeline.

A. Richer neighbourhood *features* (added to the 108, before whitening):
     - multi-scale means: average band-mean & change-wobble over k = 4, 8, 24
       neighbours (immediate block vs. wider district)
     - neighbour SPREAD: the std across the 8 neighbours of each band mean — a
       uniform estate vs. a mixed-age street is itself a signal
   Compared against the current "spatial" block (8-NN mean only).

B. Class-conditional CORAL (the originality angle):
     1. plain CORAL + RF gives a first Amsterdam prediction
     2. use those predictions to estimate a per-class covariance for Amsterdam
     3. re-align Madrid class by class, retrain, re-predict — iterate twice
   Zero Amsterdam labels used; only the model's own predictions.

Everything is scored with the tuned few-shot pipeline incl. spatial smoothing,
so numbers are directly comparable to results/deliverable_table.csv.

Usage:  python scripts/run_more_feats.py [--quick]
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

from src.adapt import make_coral, spatial_smoother, zca_whiten
from src.data import BANDS, build_city
from src.evaluate import fit_final_rf

SEED = 42
QUICK = "--quick" in sys.argv
SHOTS = (5, 25, 50, 100, 200)
CFG = json.loads((ROOT / "results" / "overnight_best.json").read_text())["winning_cfg"]


def shrink_for(n, nf):
    eff = n * 60.0 / max(nf, 1)
    return float(np.clip(CFG["sh_intercept"] - CFG["sh_slope"] * np.log2(max(eff, 2.0)), 0, 1))


def head(n_est=None):
    return RandomForestClassifier(
        n_estimators=n_est or (200 if QUICK else CFG["n_estimators"]),
        max_depth=CFG["max_depth"], min_samples_leaf=CFG["min_samples_leaf"],
        max_features=CFG["max_features"], class_weight="balanced",
        random_state=SEED, n_jobs=-1)


def few_shot(Xa, ya, prior, smoother, nf, label, n_trials):
    classes = np.unique(ya)
    per = {}
    for n in SHOTS:
        Z = zca_whiten(Xa, shrink=shrink_for(n, nf), eps=CFG["eps"])(Xa)
        beta = float(np.clip(n / CFG["beta_div"], CFG["beta_floor"], 0.95))
        rng = np.random.default_rng(SEED)
        sc = []
        for _ in range(n_trials):
            sup = np.concatenate([
                rng.choice(np.where(ya == c)[0], min(n, (ya == c).sum()),
                           replace=False) for c in classes])
            q = np.setdiff1d(np.arange(len(ya)), sup)
            clf = head().fit(Z[sup], ya[sup])
            field = np.empty((len(ya), len(classes)))
            field[q] = beta * clf.predict_proba(Z[q]) + (1 - beta) * prior[q]
            oh = np.zeros((len(sup), len(classes)))
            oh[np.arange(len(sup)), np.searchsorted(classes, ya[sup])] = 1
            field[sup] = oh
            sc.append(f1_score(ya[q], classes[smoother(field)[q].argmax(1)],
                               average="macro", zero_division=0))
        per[n] = float(np.mean(sc))
    print(f"  {label:<22} " + "  ".join(f"{s}:{per[s]:.3f}" for s in SHOTS)
          + f"   mean {np.mean([per[s] for s in SHOTS]):.4f}")
    return per


def add_rich_spatial(X, names, coords):
    """multi-scale neighbour means + 8-NN neighbour spread for band means."""
    tree = cKDTree(coords)
    base_cols = [names.index(n) for n in names
                 if n.endswith("_mean") and any(n.startswith(b) for b in BANDS)]
    base_cols += [names.index(f"d_{b}_std") for b in BANDS]
    extra, xn = [], []
    for k in (4, 24):
        nn = tree.query(coords, k=k + 1)[1][:, 1:]
        extra.append(X[:, base_cols][nn].mean(axis=1))
        xn += [f"nbr{k}_{names[c]}" for c in base_cols]
    nn8 = tree.query(coords, k=9)[1][:, 1:]
    spread_cols = [names.index(f"{b}_mean") for b in BANDS]
    extra.append(X[:, spread_cols][nn8].std(axis=1))
    xn += [f"nbrspread_{b}" for b in BANDS]
    return np.column_stack([X, *extra]).astype(np.float32), names + xn


def class_conditional_coral(Xm, ym, Xa, rounds=2, eps=1e-4):
    """Iteratively align Madrid to Amsterdam class by class using pseudo-labels."""
    from src.adapt import _sym_inv_sqrt, _sym_sqrt
    classes = np.unique(ym)
    coral = make_coral(Xm, Xa)
    rf = fit_final_rf(coral(Xm), ym, rf_params=dict(
        n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1))
    Xm_al = coral(Xm)
    for _ in range(rounds):
        yhat = rf.predict(Xa)
        Xm_new = np.empty_like(Xm_al)
        for c in classes:
            src = Xm_al[ym == c]
            tgt = Xa[yhat == c]
            if len(tgt) < Xm_al.shape[1] + 2:          # too few pseudo-labels
                Xm_new[ym == c] = src
                continue
            mu_s, mu_t = src.mean(0), tgt.mean(0)
            cs = np.cov(src - mu_s, rowvar=False) + eps * np.eye(src.shape[1])
            ct = np.cov(tgt - mu_t, rowvar=False) + eps * np.eye(src.shape[1])
            A = _sym_inv_sqrt(cs, eps) @ _sym_sqrt(ct, eps)
            Xm_new[ym == c] = (src - mu_s) @ A + mu_t
        Xm_al = Xm_new
        rf = fit_final_rf(Xm_al, ym, rf_params=dict(
            n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1))
    return rf


def main():
    t0 = time.time()
    n_trials = 5 if QUICK else 15
    fb = dict(changepoint=True, spatial=True)

    m = build_city(str(ROOT / "data" / "madrid_train.parquet"), **fb)
    a = build_city(str(ROOT / "data" / "amsterdam_data.parquet"), **fb)
    coords_a = a.pixels.to_numpy(dtype=float)
    coords_m = m.pixels.to_numpy(dtype=float)
    smoother = spatial_smoother(coords_a, k=4)
    print(f"tuned baseline features {a.X.shape}   {n_trials} draws/budget\n")

    # ---- baseline (current pipeline) ----
    prior0 = fit_final_rf(make_coral(m.X, a.X)(m.X), m.y, rf_params=dict(
        n_estimators=300, class_weight="balanced",
        random_state=SEED, n_jobs=-1)).predict_proba(a.X)
    base = few_shot(a.X, a.y, prior0, smoother, a.X.shape[1], "baseline", n_trials)

    # ---- A: richer spatial features ----
    Xm2, names2 = add_rich_spatial(m.X, m.feature_names, coords_m)
    Xa2, _ = add_rich_spatial(a.X, a.feature_names, coords_a)
    prior2 = fit_final_rf(make_coral(Xm2, Xa2)(Xm2), m.y, rf_params=dict(
        n_estimators=300, class_weight="balanced",
        random_state=SEED, n_jobs=-1)).predict_proba(Xa2)
    rich = few_shot(Xa2, a.y, prior2, smoother, Xa2.shape[1],
                    f"rich-spatial (+{Xa2.shape[1]-a.X.shape[1]}f)", n_trials)

    # ---- B: class-conditional CORAL (changes the prior only) ----
    rf_cc = class_conditional_coral(m.X, m.y, a.X)
    prior_cc = rf_cc.predict_proba(a.X)
    from src.evaluate import zero_shot
    print(f"\n  class-cond CORAL zero-shot: {zero_shot(rf_cc, a.X, a.y):.4f}  "
          f"(plain CORAL was ~0.578)")
    cc = few_shot(a.X, a.y, prior_cc, smoother, a.X.shape[1],
                  "class-cond CORAL prior", n_trials)

    # ---- C: both together ----
    rf_cc2 = class_conditional_coral(Xm2, m.y, Xa2)
    both = few_shot(Xa2, a.y, rf_cc2.predict_proba(Xa2), smoother,
                    Xa2.shape[1], "rich + class-cond", n_trials)

    print("\n" + "=" * 78)
    print(f"{'variant':>22} | " + " ".join(f"{s:>7}" for s in SHOTS) + " |   mean   dMean")
    print("-" * 78)
    bm = np.mean([base[s] for s in SHOTS])
    for lbl, r in [("baseline", base), ("rich-spatial", rich),
                   ("class-cond CORAL", cc), ("rich + class-cond", both)]:
        mn = np.mean([r[s] for s in SHOTS])
        print(f"{lbl:>22} | " + " ".join(f"{r[s]:>7.4f}" for s in SHOTS)
              + f" | {mn:.4f}  {mn-bm:+.4f}")

    (ROOT / "results" / "more_feats_scores.json").write_text(json.dumps(
        dict(baseline=base, rich_spatial=rich, class_cond=cc, both=both), indent=2))
    print(f"\nwrote results/more_feats_scores.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
