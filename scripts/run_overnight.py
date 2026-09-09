"""Overnight nested search over feature set + pipeline hyperparameters.

Honest protocol — no tuning on the reporting data:

  1. Split Amsterdam pixels 50/50, stratified by class, fixed seed:
       TUNE  = configs are scored here
       TEST  = the single winning config is evaluated here, once
  2. Random-search ~N_CONFIGS pipelines. Each config chooses:
       - feature set        base / +changepoint / +spatial / +both
       - RF depth, leaf size, max_features, tree count
       - shrinkage-whitening formula coefficients (intercept, slope)
       - ensemble weight schedule (beta divisor, floor)
       - PCA dimensionality before whitening (or none)
  3. Score = mean few-shot macro-F1 across 5/25/50/100/200 labels/class on TUNE
     (10 draws), with a small bonus weight on the 25-label point (low-data prize).
  4. Re-run the best config on TEST with 30 draws -> the number to trust.

Checkpoints results/overnight_progress.jsonl after every config, so an
interrupted run keeps its work. Final summary -> results/overnight_best.json.

    python scripts/run_overnight.py            # full (~4-8 h)
    python scripts/run_overnight.py --configs 40 --quick   # short dry run
"""

import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

print = functools.partial(print, flush=True)  # noqa: A001

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapt import make_coral, zca_whiten
from src.data import build_city
from src.evaluate import fit_final_rf, madrid_cv, zero_shot

SEED = 42
QUICK = "--quick" in sys.argv
N_CONFIGS = 40 if QUICK else int(
    sys.argv[sys.argv.index("--configs") + 1] if "--configs" in sys.argv else 500
)
SHOTS = (5, 25, 50, 100, 200)
N_DRAWS_TUNE = 6 if QUICK else 10
N_DRAWS_TEST = 10 if QUICK else 30

FEATURE_SETS = {
    "base": dict(changepoint=False, spatial=False),
    "changepoint": dict(changepoint=True, spatial=False),
    "spatial": dict(changepoint=False, spatial=True),
    "both": dict(changepoint=True, spatial=True),
}

PROG = ROOT / "results" / "overnight_progress.jsonl"
BEST = ROOT / "results" / "overnight_best.json"


def shrink_level(n, nf, intercept, slope):
    eff = n * 60.0 / max(nf, 1)
    return float(np.clip(intercept - slope * np.log2(max(eff, 2.0)), 0.0, 1.0))


def few_shot_score(Xa_te, ya_te, prior_te, cfg, nf, rng, n_draws):
    """Mean macro-F1 across budgets for one config on the given target split."""
    classes = np.unique(ya_te)
    beta_div, beta_floor = cfg["beta_div"], cfg["beta_floor"]
    per = {}
    for n in SHOTS:
        sh = shrink_level(n, nf, cfg["sh_intercept"], cfg["sh_slope"])
        Z = zca_whiten(Xa_te, shrink=sh, eps=cfg["eps"])(Xa_te)
        if cfg["pca"]:
            Z = PCA(n_components=min(cfg["pca"], Z.shape[1]),
                    random_state=SEED).fit_transform(Z)
        beta = float(np.clip(n / beta_div, beta_floor, 0.95))
        scores = []
        for _ in range(n_draws):
            sup = np.concatenate([
                rng.choice(np.where(ya_te == c)[0], min(n, (ya_te == c).sum()),
                           replace=False) for c in classes])
            q = np.setdiff1d(np.arange(len(ya_te)), sup)
            head = RandomForestClassifier(
                n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                min_samples_leaf=cfg["min_samples_leaf"],
                max_features=cfg["max_features"], class_weight="balanced",
                random_state=SEED, n_jobs=-1).fit(Z[sup], ya_te[sup])
            blend = beta * head.predict_proba(Z[q]) + (1 - beta) * prior_te[q]
            scores.append(f1_score(ya_te[q], classes[blend.argmax(1)],
                                   average="macro", zero_division=0))
        per[n] = float(np.mean(scores))
    # low-data prize gets 1.5x weight
    w = np.array([1, 1.5, 1, 1, 1])
    return float(np.average([per[n] for n in SHOTS], weights=w)), per


def _pick(rng, options):
    """Sample one element from a heterogeneous list without numpy coercion."""
    return options[int(rng.integers(len(options)))]


def sample_cfg(rng):
    return dict(
        features=_pick(rng, list(FEATURE_SETS)),
        n_estimators=_pick(rng, [200, 350, 500]),
        max_depth=_pick(rng, [None, 14, 22, 34]),
        min_samples_leaf=_pick(rng, [1, 2, 4]),
        max_features=_pick(rng, ["sqrt", 0.3, 0.5]),
        sh_intercept=_pick(rng, [1.2, 1.44, 1.7]),
        sh_slope=_pick(rng, [0.15, 0.188, 0.23]),
        beta_div=_pick(rng, [30, 50, 70]),
        beta_floor=_pick(rng, [0.3, 0.4, 0.5]),
        eps=_pick(rng, [1e-6, 1e-5, 1e-4]),
        pca=_pick(rng, [0, 0, 20, 30, 40]),
    )


def cfg_key(c):
    return json.dumps({k: (v if not isinstance(v, np.generic) else v.item())
                       for k, v in c.items()}, sort_keys=True, default=str)


def main():
    t0 = time.time()
    (ROOT / "results").mkdir(exist_ok=True)

    # build every feature set once, cache
    print("building feature sets ...")
    cache = {}
    for name, opts in FEATURE_SETS.items():
        m = build_city(str(ROOT / "data" / "madrid_train.parquet"), **opts)
        a = build_city(str(ROOT / "data" / "amsterdam_data.parquet"), **opts)
        # honest split of Amsterdam
        idx_tune, idx_test = train_test_split(
            np.arange(len(a.y)), test_size=0.5, stratify=a.y, random_state=SEED)
        coral = make_coral(m.X, a.X)
        rf_prior = fit_final_rf(coral(m.X), m.y, rf_params=dict(
            n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1))
        prior = rf_prior.predict_proba(a.X)
        cache[name] = dict(
            Xm=m.X, ym=m.y, Xa=a.X, ya=a.y, nf=a.X.shape[1],
            tune=idx_tune, test=idx_test, prior=prior,
            f1_zero_raw=zero_shot(fit_final_rf(m.X, m.y), a.X, a.y),
            f1_zero_coral=zero_shot(rf_prior, a.X, a.y),
        )
        print(f"  {name:11s} {m.X.shape[1]} feats  "
              f"zs raw {cache[name]['f1_zero_raw']:.3f}  "
              f"zs CORAL {cache[name]['f1_zero_coral']:.3f}")

    # Madrid CV once on base (reference, config-independent) — cached across restarts
    CV_CACHE = ROOT / "results" / "overnight_madrid_cv.json"
    if CV_CACHE.exists():
        _c = json.loads(CV_CACHE.read_text())
        cv = type("CV", (), {"mean": _c["mean"], "std": _c["std"]})()
        print(f"Madrid CV {cv.mean:.4f} +/- {cv.std:.4f}  (cached)\n")
    else:
        cv = madrid_cv(cache["base"]["Xm"], cache["base"]["ym"],
                       n_repeats=2 if QUICK else 5, seed=SEED)
        CV_CACHE.write_text(json.dumps({"mean": cv.mean, "std": cv.std}))
        print(f"Madrid CV {cv.mean:.4f} +/- {cv.std:.4f}\n")

    rng = np.random.default_rng(SEED)
    seen, results = set(), []
    if PROG.exists():
        for line in PROG.read_text().splitlines():
            r = json.loads(line)
            seen.add(r["key"]); results.append(r)
        print(f"resuming: {len(results)} configs already scored")

    with PROG.open("a") as fh:
        while len([r for r in results]) < N_CONFIGS:
            cfg = sample_cfg(rng)
            k = cfg_key(cfg)
            if k in seen:
                continue
            seen.add(k)
            c = cache[cfg["features"]]
            tune_score, per = few_shot_score(
                c["Xa"][c["tune"]], c["ya"][c["tune"]], c["prior"][c["tune"]],
                cfg, c["nf"], np.random.default_rng(SEED), N_DRAWS_TUNE)
            rec = dict(key=k, cfg={kk: (vv.item() if isinstance(vv, np.generic) else vv)
                                    for kk, vv in cfg.items()},
                       tune_score=tune_score, tune_per_budget=per,
                       elapsed=round(time.time() - t0, 1))
            results.append(rec)
            fh.write(json.dumps(rec, default=str) + "\n"); fh.flush()
            best = max(results, key=lambda r: r["tune_score"])
            print(f"[{len(results):4d}/{N_CONFIGS}] {cfg['features']:11s} "
                  f"tune {tune_score:.4f}   best {best['tune_score']:.4f} "
                  f"({best['cfg']['features']})   {rec['elapsed']/60:.0f}m")

    # ── evaluate the winner on TEST ────────────────────────────────
    best = max(results, key=lambda r: r["tune_score"])
    cfg = best["cfg"]
    c = cache[cfg["features"]]
    print(f"\nWINNER (tune {best['tune_score']:.4f}): {json.dumps(cfg)}")
    test_score, test_per = few_shot_score(
        c["Xa"][c["test"]], c["ya"][c["test"]], c["prior"][c["test"]],
        cfg, c["nf"], np.random.default_rng(SEED + 1), N_DRAWS_TEST)
    print(f"TEST mean {test_score:.4f}")
    for n in SHOTS:
        print(f"  {n:>3}/class  {test_per[n]:.4f}")

    BEST.write_text(json.dumps(dict(
        madrid_cv=cv.mean, madrid_cv_std=cv.std,
        zero_shot_raw=c["f1_zero_raw"], zero_shot_coral=c["f1_zero_coral"],
        winning_cfg=cfg, tune_score=best["tune_score"],
        test_score=test_score, test_per_budget=test_per,
        n_configs=len(results), hours=round((time.time() - t0) / 3600, 2),
    ), indent=2, default=str))
    print(f"\nwrote {BEST}  ({(time.time()-t0)/3600:.1f} h, {len(results)} configs)")


if __name__ == "__main__":
    main()
