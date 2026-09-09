"""Take the best config from the (partial) overnight search and score it honestly.

Rebuilds the winning feature set, re-does the seeded 50/50 Amsterdam split, and
evaluates the winner on the untouched TEST half with 30 draws per budget. Writes
results/overnight_best.json in the same shape the full run would.
"""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.adapt import make_coral, zca_whiten
from src.data import build_city
from src.evaluate import fit_final_rf, zero_shot

SEED = 42
SHOTS = (5, 25, 50, 100, 200)
FEATURE_SETS = {"base": dict(changepoint=False, spatial=False),
                "changepoint": dict(changepoint=True, spatial=False),
                "spatial": dict(changepoint=False, spatial=True),
                "both": dict(changepoint=True, spatial=True)}

rows = [json.loads(l) for l in (ROOT/"results"/"overnight_progress.jsonl").read_text().splitlines()]
best = max(rows, key=lambda r: r["tune_score"])
cfg = best["cfg"]
print(f"best of {len(rows)} configs  tune {best['tune_score']:.4f}\n{json.dumps(cfg, indent=2)}")

t0 = time.time()
m = build_city(str(ROOT/"data"/"madrid_train.parquet"), **FEATURE_SETS[cfg["features"]])
a = build_city(str(ROOT/"data"/"amsterdam_data.parquet"), **FEATURE_SETS[cfg["features"]])
idx_tune, idx_test = train_test_split(np.arange(len(a.y)), test_size=0.5,
                                      stratify=a.y, random_state=SEED)
coral = make_coral(m.X, a.X)
rf_prior = fit_final_rf(coral(m.X), m.y, rf_params=dict(
    n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1))
prior = rf_prior.predict_proba(a.X)
f1_zero_raw = zero_shot(fit_final_rf(m.X, m.y), a.X, a.y)
f1_zero_coral = zero_shot(rf_prior, a.X, a.y)

def shrink(n, nf):
    eff = n * 60.0 / max(nf, 1)
    return float(np.clip(cfg["sh_intercept"] - cfg["sh_slope"]*np.log2(max(eff, 2.0)), 0, 1))

Xte, yte, pte = a.X[idx_test], a.y[idx_test], prior[idx_test]
nf, classes, rng = a.X.shape[1], np.unique(yte), np.random.default_rng(SEED+1)
per = {}
for n in SHOTS:
    Z = zca_whiten(Xte, shrink=shrink(n, nf), eps=cfg["eps"])(Xte)
    if cfg["pca"]:
        Z = PCA(n_components=min(cfg["pca"], Z.shape[1]), random_state=SEED).fit_transform(Z)
    beta = float(np.clip(n/cfg["beta_div"], cfg["beta_floor"], 0.95))
    sc = []
    for _ in range(30):
        sup = np.concatenate([rng.choice(np.where(yte==c)[0], min(n,(yte==c).sum()),
                                         replace=False) for c in classes])
        q = np.setdiff1d(np.arange(len(yte)), sup)
        h = RandomForestClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"], max_features=cfg["max_features"],
            class_weight="balanced", random_state=SEED, n_jobs=-1).fit(Z[sup], yte[sup])
        blend = beta*h.predict_proba(Z[q]) + (1-beta)*pte[q]
        sc.append(f1_score(yte[q], classes[blend.argmax(1)], average="macro", zero_division=0))
    per[n] = [float(np.mean(sc)), float(np.std(sc))]
    print(f"  {n:>3}/class  {per[n][0]:.4f} +/- {per[n][1]:.4f}")

out = dict(winning_cfg=cfg, tune_score=best["tune_score"], n_configs=len(rows),
           partial=True, zero_shot_raw=f1_zero_raw, zero_shot_coral=f1_zero_coral,
           test_per_budget={str(k): v[0] for k, v in per.items()},
           test_per_budget_std={str(k): v[1] for k, v in per.items()},
           test_mean=float(np.mean([per[n][0] for n in SHOTS])))
(ROOT/"results"/"overnight_best.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote results/overnight_best.json  ({time.time()-t0:.0f}s)")
