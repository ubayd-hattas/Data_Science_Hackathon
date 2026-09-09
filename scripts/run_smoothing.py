"""Two untried ideas on top of the tuned pipeline.

A. Spatial smoothing of *predictions* (not features). We already feed
   neighbourhood-averaged features in; the other lever is to average the
   predicted class probabilities over each pixel's k nearest neighbours and then
   re-pick the winner. Age classes cluster geographically, so this denoises
   without adding dimensions. Two flavours:
     - "pred"   : neighbours contribute their predicted probabilities
     - "anchor" : support pixels contribute their TRUE one-hot instead
                  (legitimate — those labels are ours), so known pixels anchor
                  their neighbourhood.

B. Rescuing the 5-labels-per-class case, the only budget that never improved:
     - lower the blend floor so the Madrid prior dominates when the local head
       is near-useless
     - mixup: synthesise extra support points as convex blends of same-class
       pairs, giving the small forest more to chew on

Usage:  python scripts/run_smoothing.py [--quick]
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

from src.adapt import make_coral, zca_whiten
from src.data import build_city
from src.evaluate import fit_final_rf

SEED = 42
QUICK = "--quick" in sys.argv
SHOTS = (5, 25, 50, 100, 200)
CFG = json.loads((ROOT / "results" / "overnight_best.json").read_text())["winning_cfg"]


def shrink_for(n, nf):
    eff = n * 60.0 / max(nf, 1)
    return float(np.clip(CFG["sh_intercept"] - CFG["sh_slope"] * np.log2(max(eff, 2.0)), 0, 1))


def head():
    return RandomForestClassifier(
        n_estimators=200 if QUICK else CFG["n_estimators"],
        max_depth=CFG["max_depth"], min_samples_leaf=CFG["min_samples_leaf"],
        max_features=CFG["max_features"], class_weight="balanced",
        random_state=SEED, n_jobs=-1)


def mixup(Xs, ys, rng, n_new_per_class=40, alpha=0.4):
    """Extra support points: convex blends of same-class pairs."""
    Xa_, ya_ = [Xs], [ys]
    for c in np.unique(ys):
        idx = np.where(ys == c)[0]
        if len(idx) < 2:
            continue
        i = rng.choice(idx, n_new_per_class)
        j = rng.choice(idx, n_new_per_class)
        lam = rng.beta(alpha, alpha, size=(n_new_per_class, 1))
        Xa_.append(lam * Xs[i] + (1 - lam) * Xs[j])
        ya_.append(np.full(n_new_per_class, c))
    return np.vstack(Xa_), np.concatenate(ya_)


def main():
    t0 = time.time()
    feats = dict(changepoint=CFG["features"] in ("changepoint", "both"),
                 spatial=CFG["features"] in ("spatial", "both"))
    m = build_city(str(ROOT / "data" / "madrid_train.parquet"), **feats)
    a = build_city(str(ROOT / "data" / "amsterdam_data.parquet"), **feats)
    Xm, ym, Xa, ya = m.X, m.y, a.X, a.y
    nf, classes = Xa.shape[1], np.unique(ym)
    coords = a.pixels.to_numpy(dtype=np.float64)
    n_trials = 5 if QUICK else 15
    print(f"{CFG['features']} features {Xa.shape}   {n_trials} draws/budget\n")

    coral = make_coral(Xm, Xa)
    prior = fit_final_rf(coral(Xm), ym, rf_params=dict(
        n_estimators=300, class_weight="balanced",
        random_state=SEED, n_jobs=-1)).predict_proba(Xa)

    # neighbour index, computed once
    tree = cKDTree(coords)
    NEI = {k: tree.query(coords, k=k + 1)[1] for k in (4, 8, 16)}

    variants = ["baseline", "smooth-pred-8", "smooth-anchor-4",
                "smooth-anchor-8", "smooth-anchor-16", "low-floor", "mixup"]
    res = {v: {} for v in variants}

    for n in SHOTS:
        Z = zca_whiten(Xa, shrink=shrink_for(n, nf), eps=CFG["eps"])(Xa)
        beta = float(np.clip(n / CFG["beta_div"], CFG["beta_floor"], 0.95))
        beta_low = float(np.clip(n / CFG["beta_div"], 0.12, 0.95))
        rng = np.random.default_rng(SEED)
        acc = {v: [] for v in variants}

        for _ in range(n_trials):
            sup = np.concatenate([
                rng.choice(np.where(ya == c)[0], min(n, (ya == c).sum()),
                           replace=False) for c in classes])
            q = np.setdiff1d(np.arange(len(ya)), sup)

            clf = head().fit(Z[sup], ya[sup])
            p_head = clf.predict_proba(Z[q])
            blend = beta * p_head + (1 - beta) * prior[q]
            acc["baseline"].append(f1_score(ya[q], classes[blend.argmax(1)],
                                            average="macro", zero_division=0))

            # full-length probability field for smoothing
            def field(anchor: bool):
                P = np.empty((len(ya), len(classes)))
                P[q] = blend
                if anchor:                       # support pixels vote their truth
                    oh = np.zeros((len(sup), len(classes)))
                    oh[np.arange(len(sup)), np.searchsorted(classes, ya[sup])] = 1
                    P[sup] = oh
                else:
                    P[sup] = beta * clf.predict_proba(Z[sup]) + (1 - beta) * prior[sup]
                return P

            for tag, k, anchor in (("smooth-pred-8", 8, False),
                                   ("smooth-anchor-4", 4, True),
                                   ("smooth-anchor-8", 8, True),
                                   ("smooth-anchor-16", 16, True)):
                P = field(anchor)
                Ps = P[NEI[k]].mean(axis=1)
                acc[tag].append(f1_score(ya[q], classes[Ps[q].argmax(1)],
                                         average="macro", zero_division=0))

            bl = beta_low * p_head + (1 - beta_low) * prior[q]
            acc["low-floor"].append(f1_score(ya[q], classes[bl.argmax(1)],
                                             average="macro", zero_division=0))

            Xs2, ys2 = mixup(Z[sup], ya[sup], rng)
            pm = head().fit(Xs2, ys2).predict_proba(Z[q])
            bm = beta * pm + (1 - beta) * prior[q]
            acc["mixup"].append(f1_score(ya[q], classes[bm.argmax(1)],
                                         average="macro", zero_division=0))

        for v in variants:
            res[v][n] = float(np.mean(acc[v]))
        print(f"  n={n:>3}  " + "  ".join(f"{v.split('-')[0][:6]}:{res[v][n]:.3f}"
                                          for v in variants))

    print("\n" + "=" * 78)
    print(f"{'variant':>18} | " + " ".join(f"{s:>7}" for s in SHOTS) + "  |   mean")
    print("-" * 78)
    for v in variants:
        row = " ".join(f"{res[v][s]:>7.4f}" for s in SHOTS)
        print(f"{v:>18} | {row}  | {np.mean([res[v][s] for s in SHOTS]):.4f}")
    base = np.array([res["baseline"][s] for s in SHOTS])
    print("-" * 78)
    for v in variants[1:]:
        d = np.array([res[v][s] for s in SHOTS]) - base
        print(f"{v:>18} delta | " + " ".join(f"{x:>+7.4f}" for x in d))

    (ROOT / "results" / "smoothing_scores.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote results/smoothing_scores.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
