"""Do change-point and spatial-context features improve the pipeline?

Compares four feature sets:
  base            the original 60
  +changepoint    slope / biggest-jump size, direction, timing per band (+24)
  +spatial        neighbourhood-averaged band means and change-wobble (+24)
  +both           84 + 24 = 108

For each: Madrid 5xN CV, zero-shot RF (raw + CORAL), and the few-shot prototype
curve in whitened Amsterdam space.

Usage:  python scripts/run_features.py [--quick]
"""

import functools
import sys
import time
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)  # noqa: A001

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapt import make_coral, zca_whiten
from src.data import build_city
from src.evaluate import few_shot_curve, fit_final_rf, madrid_cv, zero_shot

SEED = 42
QUICK = "--quick" in sys.argv
SHOTS = (5, 25, 50, 100, 200)
FEATURE_SETS = {
    "base":        dict(changepoint=False, spatial=False),
    "+changepoint": dict(changepoint=True, spatial=False),
    "+spatial":     dict(changepoint=False, spatial=True),
    "+both":        dict(changepoint=True, spatial=True),
}


def main() -> None:
    t0 = time.time()
    rf_p = dict(n_estimators=150 if QUICK else 300, class_weight="balanced",
               random_state=SEED, n_jobs=-1)
    n_rep = 2 if QUICK else 5
    n_trials = 5 if QUICK else 10

    rows = []
    for name, opts in FEATURE_SETS.items():
        tf = time.time()
        Xm = build_city(str(ROOT / "data" / "madrid_train.parquet"), **opts)
        Xa = build_city(str(ROOT / "data" / "amsterdam_data.parquet"), **opts)
        print(f"\n[{name}]  {Xm.X.shape[1]} features   (build {time.time()-tf:.1f}s)")

        cv = madrid_cv(Xm.X, Xm.y, n_repeats=n_rep, rf_params=rf_p, seed=SEED)
        rf_raw = fit_final_rf(Xm.X, Xm.y, rf_params=rf_p)
        f1_zero = zero_shot(rf_raw, Xa.X, Xa.y)
        coral = make_coral(Xm.X, Xa.X)
        rf_cor = fit_final_rf(coral(Xm.X), Xm.y, rf_params=rf_p)
        f1_cor = zero_shot(rf_cor, Xa.X, Xa.y)

        fs = few_shot_curve(Xa.X, Xa.y, shots=SHOTS, n_trials=n_trials,
                            transform=zca_whiten(Xa.X), seed=SEED)
        print(f"    Madrid CV {cv.mean:.4f}+/-{cv.std:.4f}   "
              f"zero-shot raw {f1_zero:.4f}  CORAL {f1_cor:.4f}")
        print(f"    few-shot whitened: " +
              "  ".join(f"{s}:{fs.per_shot[s].mean():.3f}" for s in SHOTS))
        rows.append((name, Xm.X.shape[1], cv.mean, f1_zero, f1_cor, fs.means))

    print("\n" + "=" * 78)
    print(f"{'set':>13} {'feats':>6} {'MadridCV':>9} {'zs-raw':>8} {'zs-CORAL':>9} "
          f"{'fs@25':>7} {'fs@100':>7} {'fs@200':>7}")
    print("-" * 78)
    for name, nf, cvm, zr, zc, fsm in rows:
        print(f"{name:>13} {nf:>6} {cvm:>9.4f} {zr:>8.4f} {zc:>9.4f} "
              f"{fsm[1]:>7.3f} {fsm[3]:>7.3f} {fsm[4]:>7.3f}")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    np.savez(out / "feature_scores.npz",
             names=np.array([r[0] for r in rows]),
             nfeat=np.array([r[1] for r in rows]),
             madrid_cv=np.array([r[2] for r in rows]),
             zs_raw=np.array([r[3] for r in rows]),
             zs_coral=np.array([r[4] for r in rows]),
             fs_means=np.stack([r[5] for r in rows]), shots=np.array(SHOTS))
    print(f"\nwrote results/feature_scores.npz  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
