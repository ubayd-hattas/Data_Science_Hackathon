"""Does using the class *order* help, and does correcting for the class-mix shift?

Two ideas, tested against the current best pipeline:

  A. Ordinal head (Frank & Hall Random Forest) instead of a plain Random Forest
     / nearest-prototype rule.
  B. Label-shift correction: rescale Madrid-trained probabilities to Amsterdam's
     class mix, with the target mix estimated by EM from unlabelled Amsterdam
     (zero-shot) or counted from the support set (few-shot).

Reported for every setting: macro-F1 (the graded metric) and MAE-in-steps (how
far off on the 1-4 scale, on average).

Usage:  python scripts/run_ordinal.py [--quick]
"""

import functools
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

print = functools.partial(print, flush=True)  # noqa: A001

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapt import make_coral, zca_whiten
from src.data import build_city
from src.evaluate import few_shot_classifier_curve, few_shot_curve
from src.ordinal import (
    OrdinalRF, class_prior, estimate_target_prior, mae_in_steps, reweight_posterior,
)

SEED = 42
QUICK = "--quick" in sys.argv
SHOTS = (5, 25, 50, 100, 200)


def main() -> None:
    t0 = time.time()
    madrid = build_city(str(ROOT / "data" / "madrid_train.parquet"))
    ams = build_city(str(ROOT / "data" / "amsterdam_data.parquet"))
    Xm, ym, Xa, ya = madrid.X, madrid.y, ams.X, ams.y
    classes = np.unique(ym)
    print(f"Loaded  Madrid {Xm.shape}  Amsterdam {Xa.shape}   ({time.time()-t0:.1f}s)")

    rf_p = dict(n_estimators=150 if QUICK else 300, class_weight="balanced",
               random_state=SEED, n_jobs=-1)
    prior_m = class_prior(ym, classes)
    prior_a = class_prior(ya, classes)
    print(f"Madrid class mix    {np.round(prior_m, 3)}")
    print(f"Amsterdam class mix {np.round(prior_a, 3)}")

    # ── ZERO-SHOT: plain vs ordinal, raw vs CORAL, +/- EM prior fix ──────
    print("\n=== ZERO-SHOT (no Amsterdam labels) ===")
    coral = make_coral(Xm, Xa)
    feats = {"raw": (Xm, Xa), "CORAL": (coral(Xm), Xa)}
    heads = {"plain-RF": RandomForestClassifier(**rf_p), "ordinal-RF": OrdinalRF(rf_p)}

    print(f"{'features':>8} {'head':>11} {'prior-fix':>10} {'macroF1':>9} {'MAE-steps':>10}")
    for fname, (Xtr, Xte) in feats.items():
        for hname, head in heads.items():
            head.fit(Xtr, ym)
            proba = head.predict_proba(Xte)
            for fix in (False, True):
                p = proba
                if fix:
                    tgt = estimate_target_prior(proba, prior_m)
                    p = reweight_posterior(proba, prior_m, tgt)
                pred = classes[p.argmax(1)]
                print(f"{fname:>8} {hname:>11} {str(fix):>10} "
                      f"{f1_score(ya, pred, average='macro'):>9.4f} "
                      f"{mae_in_steps(ya, pred):>10.4f}")

    # ── FEW-SHOT: prototype vs RF head vs ordinal head, in whitened space ──
    print("\n=== FEW-SHOT (whitened Amsterdam features) ===")
    whiten = zca_whiten(Xa)
    n_trials = 5 if QUICK else 10

    proto = few_shot_curve(Xa, ya, shots=SHOTS, n_trials=n_trials,
                           transform=whiten, seed=SEED)
    rf_head = few_shot_classifier_curve(
        Xa, ya, lambda: RandomForestClassifier(**rf_p), shots=SHOTS,
        n_trials=n_trials, transform=whiten, seed=SEED, extra_metric=mae_in_steps)
    ord_head = few_shot_classifier_curve(
        Xa, ya, lambda: OrdinalRF(rf_p), shots=SHOTS,
        n_trials=n_trials, transform=whiten, seed=SEED, extra_metric=mae_in_steps)

    print(f"{'shots':>6} | {'prototype':>18} | {'RF head':>18} | {'ordinal head':>18}")
    print(f"{'':>6} | {'F1':>8} {'MAE':>8} | {'F1':>8} {'MAE':>8} | {'F1':>8} {'MAE':>8}")
    print("-" * 74)
    for s in SHOTS:
        print(f"{s:>6} | {proto.per_shot[s].mean():>8.4f} {'--':>8} | "
              f"{rf_head.per_shot[s].mean():>8.4f} {rf_head.per_shot_extra[s].mean():>8.4f} | "
              f"{ord_head.per_shot[s].mean():>8.4f} {ord_head.per_shot_extra[s].mean():>8.4f}")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    np.savez(
        out / "ordinal_scores.npz",
        shots=np.array(SHOTS), prior_m=prior_m, prior_a=prior_a,
        proto_f1=proto.means,
        rf_head_f1=rf_head.means, rf_head_mae=np.array([rf_head.per_shot_extra[s].mean() for s in SHOTS]),
        ord_head_f1=ord_head.means, ord_head_mae=np.array([ord_head.per_shot_extra[s].mean() for s in SHOTS]),
    )
    print(f"\nwrote results/ordinal_scores.npz  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
