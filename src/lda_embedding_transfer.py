"""M7 candidate: a genuinely Madrid-learned embedding that should stay stable
at LOW label counts, unlike RF-leaf-proximity (needs >=50/class to be stable).

Linear Discriminant Analysis is fit on Madrid (X, y) -- a closed-form supervised
projection that finds the linear directions maximising between-class variance
relative to within-class variance. Unlike the organiser's raw-feature
prototypes, this embedding is genuinely learned from Madrid's labelled
structure, not just Euclidean distance in the unprocessed 60-feature space.
Unlike RF-leaf-proximity, it needs no per-class mode over hundreds of trees,
so it should not collapse at n=5.

Target to beat: 0.599 @ 25/class (organiser baseline, raw-feature prototype).
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from organiser_baseline import feature_columns, prototype_predict

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "features_cache.pkl"
SHOT_SIZES = [5, 25, 50, 100, 200]
N_TRIALS = 10
SEED = 42

BASELINE_RAW = {5: 0.5437, 25: 0.5960, 50: 0.6067, 100: 0.6200, 200: 0.6215}
BASELINE_ORGANISER = {5: 0.5437, 25: 0.5986, 50: 0.6079, 100: 0.6129, 200: 0.6150}


def load_features():
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def few_shot_sweep(X, y, shots=SHOT_SIZES, n_trials=N_TRIALS, seed=SEED):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    results = {}
    for n in shots:
        f1s = []
        for _ in range(n_trials):
            support_idx = []
            for c in classes:
                idx = np.where(y == c)[0]
                support_idx.extend(rng.choice(idx, min(n, len(idx)), replace=False).tolist())
            support_idx = np.array(support_idx)
            query_mask = np.ones(len(y), dtype=bool)
            query_mask[support_idx] = False
            pred = prototype_predict(X[support_idx], y[support_idx], X[query_mask])
            f1s.append(f1_score(y[query_mask], pred, average="macro", zero_division=0))
        results[n] = (float(np.mean(f1s)), float(np.std(f1s)))
    return results


def main():
    started = time.time()
    madrid, amsterdam = load_features()
    feature_names, _ = feature_columns()

    X_madrid_raw = madrid[feature_names].to_numpy(dtype=np.float64)
    y_madrid = madrid["age_class"].to_numpy(dtype=int)
    X_amsterdam_raw = amsterdam[feature_names].to_numpy(dtype=np.float64)
    y_amsterdam = amsterdam["age_class"].to_numpy(dtype=int)

    scaler = StandardScaler().fit(X_madrid_raw)
    X_madrid = scaler.transform(X_madrid_raw)
    X_amsterdam = scaler.transform(X_amsterdam_raw)

    print("Fitting LDA embedding on Madrid (supervised, closed-form)...", flush=True)
    t0 = time.time()
    lda = LinearDiscriminantAnalysis(n_components=3)
    lda.fit(X_madrid, y_madrid)
    print(f"  done in {time.time()-t0:.1f}s, explained_variance_ratio={lda.explained_variance_ratio_}", flush=True)

    madrid_embedded = lda.transform(X_madrid)
    amsterdam_embedded = lda.transform(X_amsterdam)

    # Sanity check: does the Madrid-learned embedding even separate Madrid's
    # own classes well via nearest-centroid (upper bound on how good this
    # embedding could possibly be for Amsterdam)?
    madrid_self_pred = np.array([
        np.unique(y_madrid)[np.argmin(np.linalg.norm(
            row - np.stack([madrid_embedded[y_madrid == c].mean(axis=0) for c in np.unique(y_madrid)]), axis=1))]
        for row in madrid_embedded[:5000]  # subsample for speed, just a sanity check
    ])
    madrid_self_f1 = f1_score(y_madrid[:5000], madrid_self_pred, average="macro")
    print(f"  sanity check -- Madrid self nearest-centroid F1 (subsample): {madrid_self_f1:.4f}", flush=True)

    print("\n=== LDA-embedding nearest-prototype transfer (Amsterdam) ===", flush=True)
    t0 = time.time()
    lda_results = few_shot_sweep(amsterdam_embedded, y_amsterdam)
    for n, (m, s) in lda_results.items():
        beats_organiser = "BEATS" if m > BASELINE_ORGANISER[n] else "below"
        print(f"  {n:4d}/class: {m:.4f} +/- {s:.4f}   ({beats_organiser} organiser {BASELINE_ORGANISER[n]:.4f})", flush=True)
    print(f"({time.time()-t0:.1f}s)", flush=True)

    results = {
        "explained_variance_ratio": lda.explained_variance_ratio_.tolist(),
        "madrid_self_nearest_centroid_f1_subsample": float(madrid_self_f1),
        "lda_embedding_prototype": {str(n): {"mean": m, "std": s} for n, (m, s) in lda_results.items()},
        "baseline_organiser": BASELINE_ORGANISER,
        "elapsed_seconds": time.time() - started,
    }
    out_path = ROOT / "outputs" / "lda_embedding_transfer_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
