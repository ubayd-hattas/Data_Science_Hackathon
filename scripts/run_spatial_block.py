"""Spatial-block-disjoint few-shot evaluation — robustness check for the audit.

The deliverable curve draws the support set by simple random sampling, so
~80% of support pixels sit next to a scored query pixel
(docs/AMSTERDAM_SPATIAL_ADJACENCY_AUDIT.md). Landsat pixels are spatially
autocorrelated, so that makes the reported score a little optimistic relative
to the real task: "label a neighbourhood you have no labels in".

This script re-runs the *exact* pipeline (tuned config, class-conditional
CORAL prior, adaptive whitening, blend, distance-weighted smoothing) but the
support set is drawn only from map tiles that are >= 2 tiles away from the
tile being scored. No support pixel is anywhere near a query pixel, so the
number this prints is the adjacency-free lower bound.

    python scripts/run_spatial_block.py            # budgets 25/50/100, grid 4x4
    python scripts/run_spatial_block.py --quick    # fewer trees, budget 50 only

Writes results/spatial_block_eval.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_city
from src.adapt import zca_whiten, spatial_smoother, class_conditional_coral
from src.evaluate import fit_final_rf

DATA = ROOT / "data"
SEED = 42
QUICK = "--quick" in sys.argv
GRID = 4                      # 4x4 = 16 map tiles
# query tile must be >= MIN_TILE_GAP tiles from every support tile.
#   2 = fully disjoint (a whole buffer tile between them)
#   1 = support may sit in a tile touching the query tile, but not inside it
MIN_TILE_GAP = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--gap=")), 2)
BUDGETS = (50,) if QUICK else (25, 50, 100, 200)
N_SEEDS = 2 if QUICK else 3  # support-draw seeds per query tile


def load_config():
    best = ROOT / "results" / "overnight_best.json"
    if not best.exists():
        raise SystemExit("results/overnight_best.json missing — need the tuned config")
    b = json.loads(best.read_text())["winning_cfg"]
    feats = dict(changepoint=b["features"] in ("changepoint", "both"),
                 spatial=b["features"] in ("spatial", "both"))
    rf_kw = dict(n_estimators=b["n_estimators"], max_depth=b["max_depth"],
                 min_samples_leaf=b["min_samples_leaf"], max_features=b["max_features"])
    return b, feats, rf_kw


def tile_index(coords: np.ndarray, grid: int) -> np.ndarray:
    """Assign each pixel to one of grid*grid tiles by quantile bins on each axis."""
    out = np.zeros(len(coords), dtype=int)
    for axis in (0, 1):
        v = coords[:, axis]
        edges = np.quantile(v, np.linspace(0, 1, grid + 1))
        edges[-1] += 1.0
        b = np.clip(np.digitize(v, edges[1:-1]), 0, grid - 1)
        out = out * grid + b if axis else b
    return out


def main() -> None:
    b, feats, rf_kw = load_config()
    sh_int, sh_slp = b["sh_intercept"], b["sh_slope"]
    beta_div, beta_floor = b["beta_div"], b["beta_floor"]
    eps = b["eps"]
    rf_params = dict(**rf_kw, class_weight="balanced", random_state=SEED, n_jobs=-1)
    if QUICK:
        rf_params["n_estimators"] = min(rf_params["n_estimators"], 150)

    print("config:", b["features"], "features |", rf_params["n_estimators"], "trees")

    t0 = time.time()
    madrid = build_city(str(DATA / "madrid_train.parquet"), **feats)
    ams = build_city(str(DATA / "amsterdam_data.parquet"), **feats)
    Xm, ym = madrid.X, madrid.y
    Xa, ya = ams.X, ams.y
    classes = np.unique(ym)
    nf = Xa.shape[1]
    coords = ams.pixels.to_numpy(dtype=float)
    print(f"Amsterdam {Xa.shape}  ({time.time()-t0:.0f}s to build features)")

    # ---- Stage-1 prior: class-conditional CORAL (unlabelled target only) ----
    cache = ROOT / "results" / f"_prior_cache_{b['features']}_{'q' if QUICK else 'f'}.npy"
    if cache.exists():
        prior_proba = np.load(cache)
        print(f"class-conditional CORAL prior loaded from cache {cache.name}")
    else:
        t0 = time.time()
        rf_cc = class_conditional_coral(
            Xm, ym, Xa, lambda: RandomForestClassifier(**rf_params), rounds=2)
        prior_proba = rf_cc.predict_proba(Xa)
        np.save(cache, prior_proba)
        print(f"class-conditional CORAL prior ready  ({time.time()-t0:.0f}s)")

    smoother = spatial_smoother(coords, k=8)
    tiles = tile_index(coords, GRID)
    tile_ids = np.unique(tiles)
    print(f"{len(tile_ids)} tiles, sizes {np.bincount(tiles).min()}-{np.bincount(tiles).max()} px\n")

    def shrink_for(n):
        eff = n * 60.0 / max(nf, 1)
        return float(np.clip(sh_int - sh_slp * np.log2(max(eff, 2.0)), 0.0, 1.0))

    # zero-shot prior scored on each held-out tile (same denominator as the few-shot folds)
    prior_smoothed = smoother(prior_proba)
    prior_tile = {}
    for qt in tile_ids:
        q = np.where(tiles == qt)[0]
        if len(q) >= 40:
            prior_tile[int(qt)] = f1_score(
                ya[q], classes[prior_smoothed[q].argmax(axis=1)],
                average="macro", zero_division=0)
    pv = np.array(list(prior_tile.values()))
    print(f"  zero-shot prior, per-tile      {pv.mean():.4f} +/- {pv.std():.4f}"
          f"   ({len(pv)} tiles)\n")

    results = {"zero_shot_per_tile": dict(mean=float(pv.mean()), std=float(pv.std()),
                                          n_tiles=int(len(pv)))}
    for n in BUDGETS:
        beta = float(np.clip(n / beta_div, beta_floor, 0.95))
        white = zca_whiten(Xa, shrink=shrink_for(n), eps=eps)
        Z = white(Xa)
        scores, deltas, n_folds, skipped = [], [], 0, 0

        for qt in tile_ids:
            qx, qy = divmod(qt, GRID)
            # support-eligible tiles: Chebyshev distance >= MIN_TILE_GAP from query tile
            elig = [t for t in tile_ids
                    if max(abs((t // GRID) - qx), abs((t % GRID) - qy)) >= MIN_TILE_GAP]
            pool = np.where(np.isin(tiles, elig))[0]
            query = np.where(tiles == qt)[0]
            if len(query) < 40:
                continue
            # enough of every class in the support pool?
            if min((ya[pool] == c).sum() for c in classes) < n:
                skipped += 1
                continue

            for s in range(N_SEEDS):
                rng = np.random.default_rng(SEED + 1000 * int(qt) + s)
                support = np.concatenate([
                    rng.choice(pool[ya[pool] == c], n, replace=False) for c in classes
                ])
                head = RandomForestClassifier(**rf_params).fit(Z[support], ya[support])
                field = beta * head.predict_proba(Z) + (1 - beta) * prior_proba
                pred = classes[smoother(field)[query].argmax(axis=1)]
                f = f1_score(ya[query], pred, average="macro", zero_division=0)
                scores.append(f)
                deltas.append(f - prior_tile[int(qt)])   # vs zero-shot on the same tile
                n_folds += 1

        scores, deltas = np.asarray(scores), np.asarray(deltas)
        results[n] = dict(mean=float(scores.mean()), std=float(scores.std()),
                          vs_zero_shot=float(deltas.mean()),
                          n_folds=int(n_folds), tiles_skipped=int(skipped))
        print(f"  n={n:>3}  spatial-block macro-F1 {scores.mean():.4f} +/- {scores.std():.4f}"
              f"   (vs zero-shot same tiles: {deltas.mean():+.4f};"
              f" {n_folds} folds, {skipped} skipped)")

    out = dict(
        grid=GRID, min_tile_gap=MIN_TILE_GAP, seed=SEED, n_seeds=N_SEEDS,
        quick=QUICK, features=b["features"], per_budget=results,
        note="support drawn only from tiles >= MIN_TILE_GAP tiles from the scored tile; "
             "no support pixel is adjacent to any query pixel",
    )
    (ROOT / "results").mkdir(exist_ok=True)
    fname = f"spatial_block_eval_gap{MIN_TILE_GAP}{'_quick' if QUICK else ''}.json"
    (ROOT / "results" / fname).write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/{fname}")
    print("compare against the random-split curve in results/deliverable_table.csv")


if __name__ == "__main__":
    main()
