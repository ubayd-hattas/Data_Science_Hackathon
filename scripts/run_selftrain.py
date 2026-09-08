"""Does self-training on the 25k unlabelled Amsterdam patches help?

Baseline: fit on the support set, predict the rest (adaptive-shrink whitened
space). Self-training: iteratively promote the most confident predictions to
pseudo-labels and refit. Two base classifiers: nearest-prototype and a small RF.

Usage:  python scripts/run_selftrain.py [--quick]
"""

import functools
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

print = functools.partial(print, flush=True)  # noqa: A001

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapt import shrink_for_shots, zca_whiten
from src.data import build_city
from src.evaluate import (
    PrototypeClassifier, few_shot_classifier_curve, few_shot_selftrain_curve,
)

SEED = 42
QUICK = "--quick" in sys.argv
SHOTS = (5, 25, 50, 100, 200)


def main() -> None:
    t0 = time.time()
    ams = build_city(str(ROOT / "data" / "amsterdam_data.parquet"))
    Xa, ya = ams.X, ams.y
    nf = Xa.shape[1]
    n_trials = 5 if QUICK else 12
    rf_p = dict(n_estimators=150 if QUICK else 250, class_weight="balanced",
               random_state=SEED, n_jobs=-1)
    print(f"Amsterdam {Xa.shape}   {n_trials} draws/budget\n")

    heads = {
        "prototype": lambda: PrototypeClassifier(),
        "smallRF": lambda: RandomForestClassifier(**rf_p),
    }

    results = {}
    for hname, make in heads.items():
        for st in (False, True):
            tag = f"{hname}{'+selftrain' if st else ''}"
            tf = time.time()
            per_budget = {}
            for s in SHOTS:
                whiten = zca_whiten(Xa, shrink=shrink_for_shots(s, nf))
                fn = few_shot_selftrain_curve if st else few_shot_classifier_curve
                kw = dict(shots=(s,), n_trials=n_trials, transform=whiten, seed=SEED)
                if st:
                    kw.update(rounds=4, add_frac=0.4)
                fs = fn(Xa, ya, make, **kw)
                per_budget[s] = fs.per_shot[s].mean()
            results[tag] = per_budget
            print(f"  {tag:22s} " +
                  "  ".join(f"{s}:{per_budget[s]:.3f}" for s in SHOTS) +
                  f"   ({time.time()-tf:.0f}s)")

    print("\n" + "=" * 66)
    print(f"{'method':>22} | " + " ".join(f"{s:>6}" for s in SHOTS))
    print("-" * 66)
    for tag, pb in results.items():
        print(f"{tag:>22} | " + " ".join(f"{pb[s]:>6.3f}" for s in SHOTS))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    np.savez(out / "selftrain_scores.npz", shots=np.array(SHOTS),
             **{tag.replace("+", "_"): np.array([pb[s] for s in SHOTS])
                for tag, pb in results.items()})
    print(f"\nwrote results/selftrain_scores.npz  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
