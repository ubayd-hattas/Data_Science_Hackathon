"""Last two untested ideas: gradient boosting, and a CORAL + few-shot ensemble.

A. Boosting vs Random Forest, on both paths:
   - Madrid CV + zero-shot  (raw and CORAL features)
   - few-shot head, whitened + adaptive shrink
   HistGradientBoosting is sklearn-builtin (safe on the hub); LightGBM is tried
   too if importable.

B. Ensemble: average the zero-shot CORAL model's class probabilities with the
   few-shot head's, so Madrid knowledge and local Amsterdam labels both vote.
   Weight shifts toward the few-shot head as the label budget grows.

Usage:  python scripts/run_boosting.py [--quick]
"""

import functools
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score

print = functools.partial(print, flush=True)  # noqa: A001

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapt import make_coral, shrink_for_shots, zca_whiten
from src.data import build_city
from src.evaluate import few_shot_classifier_curve, madrid_cv, zero_shot

SEED = 42
QUICK = "--quick" in sys.argv
SHOTS = (5, 25, 50, 100, 200)

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


def rf(): return RandomForestClassifier(
    n_estimators=150 if QUICK else 300, class_weight="balanced",
    random_state=SEED, n_jobs=-1)


def hgb(): return HistGradientBoostingClassifier(
    max_iter=120 if QUICK else 300, learning_rate=0.08,
    max_depth=None, l2_regularization=1.0, random_state=SEED)


def lgbm(): return LGBMClassifier(
    n_estimators=200 if QUICK else 500, learning_rate=0.05, num_leaves=31,
    class_weight="balanced", random_state=SEED, n_jobs=-1, verbosity=-1)


def main() -> None:
    t0 = time.time()
    md = build_city(str(ROOT / "data" / "madrid_train.parquet"))
    am = build_city(str(ROOT / "data" / "amsterdam_data.parquet"))
    Xm, ym, Xa, ya = md.X, md.y, am.X, am.y
    nf = Xa.shape[1]
    classes = np.unique(ym)
    n_trials = 5 if QUICK else 12
    n_rep = 2 if QUICK else 5

    heads = {"RF": rf, "HistGB": hgb}
    if HAS_LGBM:
        heads["LightGBM"] = lgbm
    print(f"heads: {list(heads)}   ({time.time()-t0:.1f}s to load)\n")

    coral = make_coral(Xm, Xa)

    # ── A1: Madrid CV + zero-shot per head ─────────────────────────────
    print("=== Madrid CV + zero-shot ===")
    print(f"{'head':>10} {'MadridCV':>10} {'zs-raw':>8} {'zs-CORAL':>9}")
    for hname, make in heads.items():
        cv = madrid_cv(Xm, ym, n_repeats=n_rep,
                       rf_params=None, seed=SEED) if hname == "RF" else None
        # madrid_cv is RF-specific; do a quick manual CV for the others
        if cv is None:
            from sklearn.model_selection import RepeatedStratifiedKFold
            sk = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=SEED)
            sc = []
            for tr, va in sk.split(Xm, ym):
                m = make().fit(Xm[tr], ym[tr])
                sc.append(f1_score(ym[va], m.predict(Xm[va]), average="macro"))
            cv_mean, cv_std = np.mean(sc), np.std(sc)
        else:
            cv_mean, cv_std = cv.mean, cv.std
        m_raw = make().fit(Xm, ym)
        m_cor = make().fit(coral(Xm), ym)
        print(f"{hname:>10} {cv_mean:>7.3f}+-{cv_std:.3f} "
              f"{zero_shot(m_raw, Xa, ya):>8.3f} {zero_shot(m_cor, Xa, ya):>9.3f}")

    # ── A2: few-shot head comparison (whitened + adaptive shrink) ──────
    print("\n=== Few-shot head (whitened + adaptive shrink) ===")
    fs_res = {}
    for hname, make in heads.items():
        row = {}
        for s in SHOTS:
            w = zca_whiten(Xa, shrink=shrink_for_shots(s, nf))
            r = few_shot_classifier_curve(Xa, ya, make, shots=(s,),
                                          n_trials=n_trials, transform=w, seed=SEED)
            row[s] = r.per_shot[s].mean()
        fs_res[hname] = row
        print(f"  {hname:>10} " + "  ".join(f"{s}:{row[s]:.3f}" for s in SHOTS))

    # ── B: ensemble few-shot head with the CORAL zero-shot model ──────
    print("\n=== Ensemble: CORAL zero-shot proba + few-shot RF proba ===")
    coral_clf = rf().fit(coral(Xm), ym)
    proba_zs_full = coral_clf.predict_proba(Xa)          # (n_ams, 4), fixed
    rng = np.random.default_rng(SEED)
    ens_row = {}
    for s in SHOTS:
        w = zca_whiten(Xa, shrink=shrink_for_shots(s, nf))
        Z = w(Xa)
        beta = np.clip(s / 100.0, 0.25, 0.9)             # trust few-shot more as s grows
        trial = []
        for _ in range(n_trials):
            sup = []
            for c in classes:
                idx = np.where(ya == c)[0]
                sup.extend(rng.choice(idx, min(s, len(idx)), replace=False).tolist())
            sup = np.asarray(sup)
            q = np.where(~np.isin(np.arange(len(ya)), sup))[0]
            head = rf().fit(Z[sup], ya[sup])
            p_fs = head.predict_proba(Z[q])
            p_zs = proba_zs_full[q]
            pred = classes[(beta * p_fs + (1 - beta) * p_zs).argmax(axis=1)]
            trial.append(f1_score(ya[q], pred, average="macro", zero_division=0))
        ens_row[s] = float(np.mean(trial))
    print(f"  {'ensemble':>10} " + "  ".join(f"{s}:{ens_row[s]:.3f}" for s in SHOTS))
    print(f"  {'(RF alone)':>10} " + "  ".join(f"{s}:{fs_res['RF'][s]:.3f}" for s in SHOTS))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    np.savez(out / "boosting_scores.npz", shots=np.array(SHOTS),
             **{f"fs_{h}": np.array([fs_res[h][s] for s in SHOTS]) for h in heads},
             ensemble=np.array([ens_row[s] for s in SHOTS]))
    print(f"\nwrote results/boosting_scores.npz  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
