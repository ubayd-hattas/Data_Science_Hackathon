"""Final submission pipeline: Stage 1 (Madrid) training + Stage 2 (Amsterdam)
adaptation, parameterised by n = labelled Amsterdam samples per class.

Consolidates everything validated in this project:
  - Stage 1: RandomForestClassifier(500 trees, class_weight=balanced) on all
    Madrid, scaler fit on Madrid only. Reported CV uses a spatial-block
    GroupKFold (not random StratifiedKFold) with a per-fold-fit scaler, because
    random CV measurably overstates performance here (spatial_leakage_eval.py:
    0.588 spatial vs 0.618 random, -0.030 macro-F1 -- see EXPERIMENT_LOG.md and
    the war book for the full audit trail).
  - Stage 2 zero-shot: prior-shift correction to a uniform target prior before
    any Amsterdam label is used (rf_leaf_transfer.py sibling experiment:
    0.340 -> 0.473 raw zero-shot F1, using zero Amsterdam labels).
  - Stage 2 few-shot adaptation: a two-tier regime switch, arrived at after
    two failed alternatives (see EXPERIMENT_LOG / war book for the full trail):
      * A naive z-scored blend of raw-feature and RF-leaf scores made things
        WORSE at every shot size (hybrid_transfer.py) -- score fusion by
        column-wise normalisation is not safe here.
      * A supervised LDA embedding (3 components, closed-form) underperformed
        raw features at every shot size (lda_embedding_transfer.py) -- too
        aggressive a compression of the 60-feature space.
    What actually works, at every one of the five required checkpoints:
      * n <= 50/class: Bayesian combination of the prior-corrected zero-shot
        RF probability (informed by ALL of Madrid, zero Amsterdam labels) with
        the few-shot prototype's softmax likelihood (bayes_combine_transfer.py).
        This is a genuine posterior-propto-prior-times-likelihood combination,
        not a naive score blend, and it is what actually beats the organiser
        baseline at n=5 (0.559 vs 0.544) where every other method struggled.
      * n > 50/class: RF-leaf-proximity nearest-prototype (rf.apply() modal
        leaf agreement) -- a representation genuinely learned from the Madrid
        forest's structure, not just Euclidean averaging, and the strongest
        method once there is enough support to stabilise it
        (rf_leaf_transfer.py: peaks at 0.622 vs organiser's 0.615 @ 200/class).

Usage:
    python final_pipeline.py --n 25                 # one adaptation run, n=25/class
    python final_pipeline.py --sweep                # full required 5/25/50/100/200 sweep
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from organiser_baseline import feature_columns, preprocess_city, prototype_predict

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CACHE = ROOT / "outputs" / "features_cache.pkl"
MODEL_PATH = ROOT / "outputs" / "stage1_model.pkl"
RF_PARAMS = dict(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
REQUIRED_SHOTS = [5, 25, 50, 100, 200]
REGIME_SWITCH_N = 50  # raw-feature prototype below this, RF-leaf-proximity from here up
SEED = 42


# --------------------------------------------------------------------------- #
# Data / Stage 1
# --------------------------------------------------------------------------- #

def load_features():
    if FEATURES_CACHE.exists():
        with open(FEATURES_CACHE, "rb") as f:
            return pickle.load(f)
    madrid, _ = preprocess_city(ROOT / "Data" / "madrid_train.parquet")
    amsterdam, _ = preprocess_city(ROOT / "Data" / "amsterdam_data.parquet")
    FEATURES_CACHE.parent.mkdir(exist_ok=True)
    with open(FEATURES_CACHE, "wb") as f:
        pickle.dump((madrid, amsterdam), f)
    return madrid, amsterdam


def spatial_blocks(px, py, block_size_px=17):
    return (px // block_size_px).astype(str) + "_" + (py // block_size_px).astype(str)


def stage1_cross_validate(X_raw, y, px, py, n_splits=5):
    """Spatial-block CV with a per-fold-fit scaler: our validated, leakage-safe
    internal estimate of Madrid performance (see docstring)."""
    groups = spatial_blocks(px, py)
    gkf = GroupKFold(n_splits=n_splits)
    scores, confusion_total = [], np.zeros((4, 4), dtype=np.int64)
    for train_idx, val_idx in gkf.split(X_raw, y, groups=groups):
        scaler = StandardScaler().fit(X_raw[train_idx])
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(scaler.transform(X_raw[train_idx]), y[train_idx])
        pred = rf.predict(scaler.transform(X_raw[val_idx]))
        scores.append(f1_score(y[val_idx], pred, average="macro"))
        confusion_total += confusion_matrix(y[val_idx], pred, labels=[1, 2, 3, 4])
    return np.array(scores), confusion_total


def stage1_train_final(X_raw, y):
    """The actual deployed Stage 1 model: fit on ALL Madrid data (not a CV
    fold) -- this is the deliverable, not an internal estimate."""
    scaler = StandardScaler().fit(X_raw)
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(scaler.transform(X_raw), y)
    return rf, scaler


# --------------------------------------------------------------------------- #
# Stage 2: zero-shot prior correction
# --------------------------------------------------------------------------- #

def prior_corrected_zero_shot_proba(rf, X_amsterdam, madrid_prior):
    """Returns (adjusted_proba, classes). Used both to report a raw zero-shot
    F1 and as the prior for the Bayesian few-shot combination below."""
    proba = rf.predict_proba(X_amsterdam)
    classes = rf.classes_
    train_p = np.array([madrid_prior[c] for c in classes])
    uniform_p = np.full(len(classes), 1.0 / len(classes))
    adjusted = proba * (uniform_p / train_p)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted, classes


# --------------------------------------------------------------------------- #
# Stage 2: few-shot adaptation, regime switch
# --------------------------------------------------------------------------- #

def leaf_prototype_predict(leaves_support, y_support, leaves_query):
    classes = np.unique(y_support)
    n_trees = leaves_support.shape[1]
    class_modes = np.zeros((len(classes), n_trees), dtype=leaves_support.dtype)
    for i, c in enumerate(classes):
        rows = leaves_support[y_support == c]
        for t in range(n_trees):
            vals, counts = np.unique(rows[:, t], return_counts=True)
            class_modes[i, t] = vals[np.argmax(counts)]
    agreement = np.stack(
        [(leaves_query == class_modes[i]).mean(axis=1) for i in range(len(classes))], axis=1
    )
    return classes[agreement.argmax(axis=1)]


def prototype_softmax_proba(X_support, y_support, X_query, temperature=1.0):
    classes = np.unique(y_support)
    prototypes = np.stack([X_support[y_support == c].mean(axis=0) for c in classes])
    dist = np.linalg.norm(X_query[:, None, :] - prototypes[None, :, :], axis=2)
    neg_d = -dist / temperature
    neg_d -= neg_d.max(axis=1, keepdims=True)
    exp_d = np.exp(neg_d)
    return exp_d / exp_d.sum(axis=1, keepdims=True), classes


def bayes_combine_predict(zero_shot_proba_query, X_support, y_support, X_query, eps=1e-6):
    """Posterior propto prior (zero-shot RF, informed by all of Madrid) times
    likelihood (few-shot prototype softmax, informed by n Amsterdam labels)."""
    few_shot_proba, classes = prototype_softmax_proba(X_support, y_support, X_query)
    combined = zero_shot_proba_query * few_shot_proba + eps
    combined /= combined.sum(axis=1, keepdims=True)
    return classes[combined.argmax(axis=1)]


def adapt_and_predict(n_per_class, X_support, y_support, X_query,
                       leaves_support=None, leaves_query=None, zero_shot_proba_query=None):
    """The parameterised Stage-2 adaptation procedure, submission-ready.

    n_per_class selects the regime (both tiers tested to beat, or tie, the
    organiser baseline at every required checkpoint -- see module docstring):
      - n <= REGIME_SWITCH_N: Bayesian combination of the zero-shot RF prior
        (predicted, per query pixel) with the few-shot prototype likelihood.
      - n >  REGIME_SWITCH_N: RF-leaf-proximity nearest-prototype, using the
        Stage 1 forest's rf.apply() leaf assignments for support/query.
    """
    if n_per_class > REGIME_SWITCH_N:
        if leaves_support is None or leaves_query is None:
            raise ValueError("leaf assignments required for n > regime switch")
        return leaf_prototype_predict(leaves_support, y_support, leaves_query)
    if zero_shot_proba_query is None:
        raise ValueError("zero_shot_proba_query required for n <= regime switch")
    return bayes_combine_predict(zero_shot_proba_query, X_support, y_support, X_query)


def few_shot_sweep(X_amsterdam, leaves_amsterdam, y_amsterdam, zero_shot_proba, shots, n_trials=10, seed=SEED):
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
            pred = adapt_and_predict(
                n, X_amsterdam[support_idx], y_amsterdam[support_idx], X_amsterdam[query_mask],
                leaves_support=leaves_amsterdam[support_idx], leaves_query=leaves_amsterdam[query_mask],
                zero_shot_proba_query=zero_shot_proba[query_mask],
            )
            f1s.append(f1_score(y_amsterdam[query_mask], pred, average="macro", zero_division=0))
        results[n] = {"mean": float(np.mean(f1s)), "std": float(np.std(f1s)), "scores": f1s}
        print(f"  {n:4d}/class: {results[n]['mean']:.4f} +/- {results[n]['std']:.4f}", flush=True)
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="Amsterdam labels per class for a single adaptation run")
    parser.add_argument("--sweep", action="store_true", help="run the full required 5/25/50/100/200 sweep")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.n is None and not args.sweep:
        parser.error("pass --n <int> for a single run, or --sweep for the full required sweep")

    started = time.time()
    madrid, amsterdam = load_features()
    feature_names, _ = feature_columns()

    X_madrid_raw = madrid[feature_names].to_numpy(dtype=np.float64)
    y_madrid = madrid["age_class"].to_numpy(dtype=int)
    px_madrid = madrid["px_key"].to_numpy(dtype=np.int64)
    py_madrid = madrid["py_key"].to_numpy(dtype=np.int64)
    X_amsterdam_raw = amsterdam[feature_names].to_numpy(dtype=np.float64)
    y_amsterdam = amsterdam["age_class"].to_numpy(dtype=int)

    print("=== Stage 1: spatial-block CV (leakage-safe internal estimate) ===", flush=True)
    t0 = time.time()
    cv_scores, confusion_total = stage1_cross_validate(X_madrid_raw, y_madrid, px_madrid, py_madrid)
    print(f"Spatial CV macro-F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}  ({time.time()-t0:.1f}s)", flush=True)
    confusion_props = (confusion_total / confusion_total.sum(axis=1, keepdims=True)).tolist()

    print("\n=== Stage 1: train final model on all Madrid ===", flush=True)
    t0 = time.time()
    rf, scaler = stage1_train_final(X_madrid_raw, y_madrid)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": rf, "scaler": scaler, "feature_names": feature_names}, f)
    print(f"  saved Stage 1 model to {MODEL_PATH}", flush=True)

    X_amsterdam = scaler.transform(X_amsterdam_raw)
    madrid_prior = {c: (y_madrid == c).mean() for c in np.unique(y_madrid)}

    print("\n=== Stage 2: prior-corrected zero-shot ===", flush=True)
    zero_shot_proba, zs_classes = prior_corrected_zero_shot_proba(rf, X_amsterdam, madrid_prior)
    zero_f1 = f1_score(y_amsterdam, zs_classes[zero_shot_proba.argmax(axis=1)], average="macro")
    print(f"Zero-shot (uniform-prior-corrected) macro-F1: {zero_f1:.4f}", flush=True)

    leaves_amsterdam = rf.apply(X_amsterdam)

    shots = REQUIRED_SHOTS if args.sweep else [args.n]
    print(f"\n=== Stage 2: adaptation sweep, n={shots} ===", flush=True)
    t0 = time.time()
    sweep_results = few_shot_sweep(X_amsterdam, leaves_amsterdam, y_amsterdam, zero_shot_proba, shots, n_trials=args.trials, seed=args.seed)
    print(f"({time.time()-t0:.1f}s)", flush=True)

    output = {
        "seed": args.seed,
        "stage1_spatial_cv": {"mean": float(cv_scores.mean()), "std": float(cv_scores.std()),
                               "scores": cv_scores.tolist(),
                               "confusion_matrix_row_proportions": confusion_props},
        "stage1_model_path": str(MODEL_PATH),
        "zero_shot_prior_corrected_macro_f1": float(zero_f1),
        "adaptation_sweep": {str(n): sweep_results[n] for n in shots},
        "regime_switch_n": REGIME_SWITCH_N,
        "elapsed_seconds": time.time() - started,
    }
    suffix = "sweep" if args.sweep else f"n{args.n}"
    out_path = ROOT / "outputs" / f"final_pipeline_{suffix}.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
