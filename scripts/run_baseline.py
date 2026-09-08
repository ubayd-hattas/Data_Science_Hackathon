"""Baseline + domain-adaptation comparison for Madrid -> Amsterdam transfer.

Reports:
  [1] Madrid 5xN CV                          in-city reference
  [2] Zero-shot RF -> Amsterdam              raw domain gap
      2b. + CORAL-aligned Madrid features    (still no Amsterdam labels)
  [3] Few-shot prototype curve, several feature treatments:
        raw            reuse the Madrid StandardScaler  (== Notebook 4)
        rescale        match per-feature mean/var to Amsterdam
        zca            transductively whiten Amsterdam by its own covariance
        embedding      Madrid-trained triplet embedding
        embedding+zca  whiten the embedding space

All treatments are unsupervised w.r.t. Amsterdam labels: the support/query split
is the only place a target label is touched.

Usage:  python scripts/run_baseline.py [--quick] [--no-embedding]
"""

import functools
import sys
import time
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)  # noqa: A001

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapt import make_coral, rescale_to, zca_whiten
from src.data import build_city
from src.embedding import TripletEmbedding
from src.evaluate import (
    FewShotResult, few_shot_curve, fit_final_rf, madrid_cv, plot_curve,
    summary_table, zero_shot,
)

SEED = 42
QUICK = "--quick" in sys.argv
NO_EMB = "--no-embedding" in sys.argv


def main() -> None:
    t0 = time.time()
    print("Loading + feature engineering ...")
    madrid = build_city(str(ROOT / "data" / "madrid_train.parquet"))
    ams = build_city(str(ROOT / "data" / "amsterdam_data.parquet"))
    Xm, ym, Xa, ya = madrid.X, madrid.y, ams.X, ams.y
    print(f"  Madrid {Xm.shape}  Amsterdam {Xa.shape}   ({time.time()-t0:.1f}s)")

    n_rep = 3 if QUICK else 5
    n_tree = 200 if QUICK else 300
    rf_p = dict(n_estimators=n_tree, class_weight="balanced",
               random_state=SEED, n_jobs=-1)

    # [1] Madrid CV --------------------------------------------------------
    print(f"\n[1] Madrid 5x{n_rep} CV ({n_tree} trees) ...")
    cv = madrid_cv(Xm, ym, n_repeats=n_rep, rf_params=rf_p, seed=SEED)
    print("   ", cv)

    # [2] Zero-shot, raw + CORAL ----------------------------------------
    print("\n[2] Zero-shot RF -> Amsterdam ...")
    rf_raw = fit_final_rf(Xm, ym, rf_params=rf_p)
    f1_zero = zero_shot(rf_raw, Xa, ya)
    coral = make_coral(Xm, Xa)
    rf_coral = fit_final_rf(coral(Xm), ym, rf_params=rf_p)
    f1_zero_coral = zero_shot(rf_coral, Xa, ya)
    print(f"    raw            {f1_zero:.4f}")
    print(f"    CORAL-aligned  {f1_zero_coral:.4f}")

    # [3] Few-shot prototype curves -----------------------------------
    # Prototype distance is scale-sensitive, so every variant standardises the
    # 60 features; they differ only in *whose* statistics do it.
    print("\n[3] Few-shot prototype transfer ...")
    mu_m, sd_m = Xm.mean(0), Xm.std(0) + 1e-8       # Madrid stats (== Notebook 4)
    mu_a, sd_a = Xa.mean(0), Xa.std(0) + 1e-8       # Amsterdam stats (transductive)

    coral_a2m = make_coral(Xa, Xm)          # reshape Amsterdam to Madrid's shape
    whiten_ams = zca_whiten(Xa)             # decorrelate by Amsterdam's own cov
    whiten_mad = zca_whiten(Xm)             # decorrelate by Madrid's cov (transfer)

    variants: dict[str, FewShotResult] = {}
    variants["madrid_scale"] = few_shot_curve((Xa - mu_m) / sd_m, ya, seed=SEED)
    variants["ams_scale"] = few_shot_curve((Xa - mu_a) / sd_a, ya, seed=SEED)
    variants["zca_ams"] = few_shot_curve(Xa, ya, transform=whiten_ams, seed=SEED)
    variants["zca_mad"] = few_shot_curve(Xa, ya, transform=whiten_mad, seed=SEED)
    variants["coral+zca"] = few_shot_curve(
        Xa, ya, transform=lambda X: whiten_mad(coral_a2m(X)), seed=SEED
    )

    if not NO_EMB:
        te = time.time()
        emb = TripletEmbedding(dim=16, epochs=15 if QUICK else 60,
                               semi_hard=False, seed=SEED).fit(Xm, ym)
        print(f"    embedding trained in {time.time()-te:.1f}s "
              f"(loss {emb.history_[0]:.3f} -> {emb.history_[-1]:.3f})")
        Za = emb.transform(Xa)
        variants["embedding"] = few_shot_curve(Za, ya, transform=None, seed=SEED)
        variants["embed+zca"] = few_shot_curve(
            Za, ya, transform=zca_whiten(Za), seed=SEED
        )

    # --- comparison table ------------------------------------------
    names = list(variants)
    shots = variants["madrid_scale"].shots
    print("\n" + "=" * (12 + 15 * len(names)))
    print("FEW-SHOT macro-F1  (mean over 10 draws)")
    print("=" * (12 + 15 * len(names)))
    print(f"{'shots':>6} |" + "".join(f" {n:>13} |" for n in names))
    print("-" * (12 + 15 * len(names)))
    for s in shots:
        print(f"{s:>6} |" + "".join(
            f" {variants[n].per_shot[s].mean():>13.4f} |" for n in names))
    print("-" * (12 + 15 * len(names)))
    print(f"  Madrid CV {cv.mean:.3f}   zero-shot raw {f1_zero:.3f}   "
          f"zero-shot CORAL {f1_zero_coral:.3f}")

    # --- deliverables --------------------------------------------
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    best = max(names, key=lambda n: variants[n].means[-1])
    (out / "summary.md").write_text(
        f"# Transfer results\n\n"
        f"Madrid CV: {cv.mean:.3f} +/- {cv.std:.3f}\n\n"
        f"Zero-shot: raw {f1_zero:.3f}, CORAL {f1_zero_coral:.3f}\n\n"
        + summary_table(cv, f1_zero, variants[best]).replace(
            "Few-shot,", f"Few-shot [{best}],")
    )
    try:
        for n, fs in variants.items():
            plot_curve(fs, f1_zero, cv, str(out / f"curve_{n.replace('+','_')}.png"))
    except ModuleNotFoundError as exc:
        print(f"    (skipped plots: {exc})")
    np.savez(
        out / "scores.npz",
        shots=np.array(shots), cv_mean=cv.mean, cv_std=cv.std,
        f1_zero=f1_zero, f1_zero_coral=f1_zero_coral,
        **{f"{n.replace('+','_')}_means": variants[n].means for n in names},
        **{f"{n.replace('+','_')}_stds": variants[n].stds for n in names},
    )
    print(f"\nbest at 200 shots: '{best}' ({variants[best].means[-1]:.4f})")
    print(f"wrote results/  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
