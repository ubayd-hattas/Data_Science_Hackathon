"""Reproduce the Notebook 4 result and test whether the Madrid embedding helps.

Runs four things and prints a comparison:
  1. Madrid 5x5 CV                          (headline in-city estimate)
  2. Zero-shot RF -> Amsterdam              (raw domain gap)
  3. Few-shot prototypes, RAW features      (what Notebook 4 actually does)
  4. Few-shot prototypes, Madrid EMBEDDING  (what the approach note specifies)

Plus the diagnostic: (3) with the standardiser refitted on Amsterdam. If that
barely moves the curve, nothing Madrid-trained is reaching Amsterdam in the
raw-feature version, and (4) is the fix.

Usage:  python scripts/run_baseline.py
"""

import functools
import sys
import time
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)  # noqa: A001  (see progress live)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUICK = "--quick" in sys.argv  # smaller CV + fewer epochs for a fast first look

from src.data import build_city
from src.embedding import TripletEmbedding
from src.evaluate import (
    few_shot_curve, fit_final_rf, madrid_cv, plot_curve, summary_table, zero_shot,
)

SEED = 42


def main() -> None:
    t0 = time.time()
    print("Loading + feature engineering ...")
    madrid = build_city(str(ROOT / "data" / "madrid_train.parquet"))
    ams = build_city(str(ROOT / "data" / "amsterdam_data.parquet"))
    print(f"  Madrid {madrid.X.shape}  Amsterdam {ams.X.shape}   ({time.time()-t0:.1f}s)")

    # 1. Madrid CV -----------------------------------------------------------
    n_rep = 2 if QUICK else 5
    rf_p = dict(n_estimators=150 if QUICK else 300, class_weight="balanced",
                random_state=SEED, n_jobs=-1)
    print(f"\n[1] Madrid 5x{n_rep} cross-validation ...")
    cv = madrid_cv(madrid.X, madrid.y, n_repeats=n_rep, rf_params=rf_p, seed=SEED)
    print("   ", cv)

    # 2. Zero-shot ---------------------------------------------------------
    print("\n[2] Zero-shot RF -> Amsterdam ...")
    rf = fit_final_rf(madrid.X, madrid.y, rf_params=rf_p)
    f1_zero = zero_shot(rf, ams.X, ams.y)
    print(f"    macro-F1: {f1_zero:.4f}")

    # 3. Few-shot, raw features (Notebook 4) -------------------------------
    print("\n[3] Few-shot prototypes on RAW features ...")
    fs_raw = few_shot_curve(ams.X, ams.y, transform=None, seed=SEED)
    print(fs_raw.table())

    # 3b. diagnostic: does the raw-feature version use Madrid at all? -----
    #     Notebook 3 standardises with a scaler fitted on Madrid. Refit it on
    #     Amsterdam instead and see if the curve changes.
    mu_a, sd_a = ams.X.mean(0), ams.X.std(0) + 1e-8
    fs_raw_amsscale = few_shot_curve(
        (ams.X - mu_a) / sd_a, ams.y, transform=None, seed=SEED
    )

    # 4. Few-shot, Madrid-trained embedding ------------------------------
    print("\n[4] Few-shot prototypes in a Madrid-trained triplet embedding ...")
    te = time.time()
    emb = TripletEmbedding(dim=16, epochs=15 if QUICK else 60,
                           semi_hard=False, seed=SEED).fit(madrid.X, madrid.y)
    print(f"    embedding trained in {time.time()-te:.1f}s   "
          f"(final triplet loss {emb.history_[-1]:.4f})")
    fs_emb = few_shot_curve(ams.X, ams.y, transform=emb.transform, seed=SEED)
    print(fs_emb.table())

    # --- comparison -----------------------------------------------------
    print("\n" + "=" * 64)
    print("COMPARISON  (macro-F1, mean over 10 draws)")
    print("=" * 64)
    print(f"{'shots':>6} | {'raw (NB4)':>12} | {'raw, AMS-scale':>15} | {'MAD embedding':>14}")
    print("-" * 64)
    for s in fs_raw.shots:
        print(f"{s:>6} | {fs_raw.per_shot[s].mean():>12.4f} | "
              f"{fs_raw_amsscale.per_shot[s].mean():>15.4f} | "
              f"{fs_emb.per_shot[s].mean():>14.4f}")
    print("-" * 64)
    d = abs(fs_raw.means - fs_raw_amsscale.means).max()
    print(f"max |raw - raw(AMS-scale)| = {d:.4f}  "
          f"->  {'Madrid scaler is doing ~nothing' if d < 0.01 else 'Madrid scaler matters'}")

    # --- deliverables -------------------------------------------------
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "summary_raw.md").write_text(summary_table(cv, f1_zero, fs_raw))
    (out / "summary_embedding.md").write_text(summary_table(cv, f1_zero, fs_emb))
    try:
        plot_curve(fs_raw, f1_zero, cv, str(out / "curve_raw.png"))
        plot_curve(fs_emb, f1_zero, cv, str(out / "curve_embedding.png"))
    except ModuleNotFoundError as exc:
        print(f"    (skipped plots: {exc})")
    np.savez(
        out / "scores.npz",
        cv_fold_f1=cv.fold_f1, f1_zero=f1_zero,
        shots=np.array(fs_raw.shots),
        raw_means=fs_raw.means, raw_stds=fs_raw.stds,
        raw_amsscale_means=fs_raw_amsscale.means,
        emb_means=fs_emb.means, emb_stds=fs_emb.stds,
    )
    print(f"\nwrote results/ ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
