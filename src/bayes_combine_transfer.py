"""M7 candidate 2: Bayesian combination of the prior-corrected zero-shot RF
(informed by all of Madrid, zero Amsterdam labels) with the few-shot raw-
feature prototype (informed only by n Amsterdam labels).

Different from the earlier failed hybrid (which z-scored two similarity
scores and summed them -- that shifted decision boundaries badly). Here both
signals are proper class probability distributions over the same 4 classes,
so we combine them the principled way: posterior proportional to
prior * likelihood, elementwise, renormalised. The zero-shot RF prior should
stabilise predictions when few-shot evidence is weak (n=5), while the
few-shot likelihood should dominate as support grows and prototype distances
become more reliable.

Target to beat: 0.599 @ 25/class (organiser baseline).
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from organiser_baseline import feature_columns

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "features_cache.pkl"
RF_PARAMS = dict(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
SHOT_SIZES = [5, 25, 50, 100, 200]
N_TRIALS = 10
SEED = 42
BASELINE_ORGANISER = {5: 0.5437, 25: 0.5986, 50: 0.6079, 100: 0.6129, 200: 0.6150}


def load_features():
    with open(CACHE, "rb") as f:
        return pickle.load(f)


def prior_corrected_proba(rf, X, madrid_prior):
    proba = rf.predict_proba(X)
    classes = rf.classes_
    train_p = np.array([madrid_prior[c] for c in classes])
    uniform_p = np.full(len(classes), 1.0 / len(classes))
    adjusted = proba * (uniform_p / train_p)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted, classes


def prototype_softmax_proba(X_support, y_support, X_query, temperature=1.0):
    classes = np.unique(y_support)
    prototypes = np.stack([X_support[y_support == c].mean(axis=0) for c in classes])
    dist = np.linalg.norm(X_query[:, None, :] - prototypes[None, :, :], axis=2)
    neg_d = -dist / temperature
    neg_d -= neg_d.max(axis=1, keepdims=True)
    exp_d = np.exp(neg_d)
    return exp_d / exp_d.sum(axis=1, keepdims=True), classes


def bayes_combine_predict(zero_shot_proba_all, X_support, y_support, X_query, query_idx_in_full, eps=1e-6):
    few_shot_proba, classes = prototype_softmax_proba(X_support, y_support, X_query)
    prior = zero_shot_proba_all[query_idx_in_full]  # aligned to same class order (1,2,3,4)
    combined = prior * few_shot_proba + eps
    combined /= combined.sum(axis=1, keepdims=True)
    return classes[combined.argmax(axis=1)]


def few_shot_sweep_bayes(X_amsterdam, y_amsterdam, zero_shot_proba, shots=SHOT_SIZES, n_trials=N_TRIALS, seed=SEED):
    rng = np.random.default_rng(seed)
    classes = np.unique(y_amsterdam)
    results = {}
    for n in shots:
        f1s = []
        for _ in range(n_trials):
            support_idx = []
            for c in classes:
                idx = np.where(y_amsterdam == c)[0]
                support_idx.extend(rng.choice(idx, min(n, len(idx)), replace=False).tolist())
            support_idx = np.array(support_idx)
            query_mask = np.ones(len(y_amsterdam), dtype=bool)
            query_mask[support_idx] = False
            query_idx = np.where(query_mask)[0]
            pred = bayes_combine_predict(
                zero_shot_proba, X_amsterdam[support_idx], y_amsterdam[support_idx],
                X_amsterdam[query_mask], query_idx,
            )
            f1s.append(f1_score(y_amsterdam[query_mask], pred, average="macro", zero_division=0))
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

    print("Fitting final Random Forest on all Madrid...", flush=True)
    t0 = time.time()
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_madrid, y_madrid)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    madrid_prior = {c: (y_madrid == c).mean() for c in np.unique(y_madrid)}
    zero_shot_proba, classes = prior_corrected_proba(rf, X_amsterdam, madrid_prior)
    zero_shot_f1 = f1_score(y_amsterdam, classes[zero_shot_proba.argmax(axis=1)], average="macro")
    print(f"Zero-shot (prior-corrected) macro-F1: {zero_shot_f1:.4f}", flush=True)

    print("\n=== Bayes-combined (zero-shot prior x few-shot likelihood) ===", flush=True)
    t0 = time.time()
    bayes_results = few_shot_sweep_bayes(X_amsterdam, y_amsterdam, zero_shot_proba)
    for n, (m, s) in bayes_results.items():
        beats = "BEATS" if m > BASELINE_ORGANISER[n] else "below"
        print(f"  {n:4d}/class: {m:.4f} +/- {s:.4f}   ({beats} organiser {BASELINE_ORGANISER[n]:.4f})", flush=True)
    print(f"({time.time()-t0:.1f}s)", flush=True)

    results = {
        "zero_shot_prior_corrected_f1": float(zero_shot_f1),
        "bayes_combined": {str(n): {"mean": m, "std": s} for n, (m, s) in bayes_results.items()},
        "baseline_organiser": BASELINE_ORGANISER,
        "elapsed_seconds": time.time() - started,
    }
    out_path = ROOT / "outputs" / "bayes_combine_transfer_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
