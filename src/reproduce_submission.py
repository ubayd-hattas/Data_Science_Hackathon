"""Reproduce the frozen final model, evaluations, artifacts, figures, and smoke tests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from final_pipeline import (
    CLASS_ORDER,
    EVALUATION_SEED,
    RF_PARAMS,
    VALID_BUDGETS,
    adapt,
    extract_spectral_features,
    feature_columns,
    fitted_scaler_from_statistics,
    load_stage1,
    predict,
    train_stage1,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "final_submission"
MODEL_PATH = ROOT / "artifacts" / "stage1_madrid_rf.pkl"
CACHE_PATH = ROOT / "outputs" / "features_60.npz"
SPLIT_PATH = ROOT / "outputs" / "amsterdam_fixed_splits.npz"
RESULT_PATH = OUTPUT / "final_results.json"
DETAIL_PATH = OUTPUT / "amsterdam_trial_details.npz"
HISTORICAL = {5: (0.5592, 0.0316), 25: (0.5976, 0.0235), 50: (0.6086, 0.0095), 100: (0.6200, 0.0064), 200: (0.6215, 0.0047)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unscaled_frame(scaled: np.ndarray, ids: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> pd.DataFrame:
    raw = scaled.astype(np.float64) * scale + mean
    frame = pd.DataFrame(raw, columns=feature_columns())
    frame.insert(0, "py_key", ids[:, 1])
    frame.insert(0, "px_key", ids[:, 0])
    return frame


def per_class_f1(counts: np.ndarray) -> list[float]:
    values = []
    for i in range(4):
        denominator = counts[i, :].sum() + counts[:, i].sum()
        values.append(float(2 * counts[i, i] / denominator) if denominator else 0.0)
    return values


def madrid_cv(scaled: np.ndarray, labels: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> dict:
    raw = scaled.astype(np.float64) * scale + mean
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=EVALUATION_SEED)
    scores, fold_counts = [], []
    total = np.zeros((4, 4), dtype=np.int64)
    for number, (train_idx, validation_idx) in enumerate(splitter.split(raw, labels), 1):
        scaler = StandardScaler().fit(raw[train_idx])
        forest = RandomForestClassifier(**RF_PARAMS)
        forest.fit(scaler.transform(raw[train_idx]).astype(np.float32), labels[train_idx])
        prediction = forest.predict(scaler.transform(raw[validation_idx]).astype(np.float32))
        score = f1_score(labels[validation_idx], prediction, labels=CLASS_ORDER, average="macro", zero_division=0)
        counts = confusion_matrix(labels[validation_idx], prediction, labels=CLASS_ORDER)
        scores.append(float(score)); fold_counts.append(counts.tolist()); total += counts
        print(f"Madrid CV {number:02d}/25: {score:.6f}", flush=True)
    return {
        "protocol": "5-fold x 5-repeat stratified random CV; scaler fitted within every fold",
        "scores": scores,
        "mean": float(np.mean(scores)),
        "population_sd": float(np.std(scores)),
        "fold_confusion_counts": fold_counts,
        "confusion_counts": total.tolist(),
        "confusion_row_proportions": (total / total.sum(axis=1, keepdims=True)).tolist(),
        "per_class_f1_from_pooled_counts": per_class_f1(total),
    }


def evaluate_amsterdam(package: dict, x_scaled: np.ndarray, labels: np.ndarray, ids: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    features = unscaled_frame(x_scaled, ids, mean, scale)
    arrays: dict[str, np.ndarray] = {}
    results = {}
    with np.load(SPLIT_PATH, allow_pickle=False) as split_file:
        for n in VALID_BUDGETS:
            trials = []
            for trial in range(10):
                support_idx = split_file[f"support_{n}_{trial}"]
                query_idx = split_file[f"query_{n}_{trial}"]
                support = features.iloc[support_idx].reset_index(drop=True)
                query = features.iloc[query_idx].reset_index(drop=True)
                state = adapt(package, support, labels[support_idx], n)
                prediction_frame = predict(package, state, query)
                prediction = prediction_frame["predicted_age_class"].to_numpy(dtype=np.int64)
                counts = confusion_matrix(labels[query_idx], prediction, labels=CLASS_ORDER)
                score = f1_score(labels[query_idx], prediction, labels=CLASS_ORDER, average="macro", zero_division=0)
                arrays[f"support_indices_{n}_{trial}"] = support_idx
                arrays[f"query_indices_{n}_{trial}"] = query_idx
                arrays[f"support_pixel_ids_{n}_{trial}"] = ids[support_idx]
                arrays[f"query_pixel_ids_{n}_{trial}"] = ids[query_idx]
                arrays[f"predictions_{n}_{trial}"] = prediction.astype(np.int8)
                trials.append({
                    "trial": trial,
                    "macro_f1": float(score),
                    "confusion_counts": counts.tolist(),
                    "per_class_f1": per_class_f1(counts),
                    "support_array_keys": [f"support_indices_{n}_{trial}", f"support_pixel_ids_{n}_{trial}"],
                    "query_array_keys": [f"query_indices_{n}_{trial}", f"query_pixel_ids_{n}_{trial}"],
                    "prediction_array_key": f"predictions_{n}_{trial}",
                })
            scores = [t["macro_f1"] for t in trials]
            results[str(n)] = {
                "branch": "score_product" if n <= 50 else "modal_leaf_agreement",
                "mean": float(np.mean(scores)),
                "population_sd": float(np.std(scores)),
                "scores": scores,
                "historical_reported_mean": HISTORICAL[n][0],
                "historical_reported_population_sd": HISTORICAL[n][1],
                "difference_from_historical_mean": float(np.mean(scores) - HISTORICAL[n][0]),
                "trials": trials,
            }
            print(f"Amsterdam {n:3d}/class: {np.mean(scores):.6f} +/- {np.std(scores):.6f}", flush=True)
    return results, arrays


def benchmark_values() -> dict:
    consolidated = json.loads((ROOT / "outputs" / "consolidated_results.json").read_text(encoding="utf-8"))
    candidates = consolidated["final_candidate_methods"]
    mapping = {
        "raw_prototype": candidates["raw_60_prototype"],
        "amsterdam_support_logistic": candidates["raw_60_logistic"],
        "madrid_triplet_prototype": candidates["madrid_triplet_embedding_prototype"],
        "madrid_triplet_logistic": candidates["madrid_triplet_embedding_logistic"],
    }
    return {
        name: {
            str(n): {"mean": float(values[str(n)]["macro_f1_mean"]), "population_sd": float(values[str(n)]["macro_f1_std_population"])}
            for n in VALID_BUDGETS
        }
        for name, values in mapping.items()
    }


def make_figures(results: dict, madrid: dict) -> None:
    figure_dir = OUTPUT / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    n = np.array(VALID_BUDGETS); means = np.array([results[str(v)]["mean"] for v in n]); sd = np.array([results[str(v)]["population_sd"] for v in n])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(np.log2(n), means, yerr=sd, marker="o", capsize=4, color="#175a8a", linewidth=2)
    ax.set_xticks(np.log2(n), [str(v) for v in n]); ax.set_xlabel("Labelled Amsterdam pixels per class (log2 spacing)"); ax.set_ylabel("Macro F1")
    ax.set_title("Frozen Madrid RF transfer on internal Amsterdam trials"); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(figure_dir / "amsterdam_f1_vs_log2_labels.png", dpi=180); plt.close(fig)

    cm = np.sum([np.asarray(t["confusion_counts"]) for t in results["25"]["trials"]], axis=0)
    row = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 5.0)); image = ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4): ax.text(j, i, f"{row[i,j]:.2f}\n({cm[i,j]:,})", ha="center", va="center", color="white" if row[i,j] > .55 else "black", fontsize=8)
    ax.set_xticks(range(4), ["C1", "C2", "C3", "C4"]); ax.set_yticks(range(4), ["C1", "C2", "C3", "C4"]); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Amsterdam score-product transfer at 25/class\npooled internal trial appearances"); fig.colorbar(image, ax=ax, label="Row proportion"); fig.tight_layout()
    fig.savefig(figure_dir / "amsterdam_confusion_25.png", dpi=180); plt.close(fig)

    cm = np.asarray(madrid["confusion_counts"]); row = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.4, 4.8)); image = ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4): ax.text(j, i, f"{row[i,j]:.2f}", ha="center", va="center", color="white" if row[i,j] > .55 else "black")
    ax.set_xticks(range(4), ["C1", "C2", "C3", "C4"]); ax.set_yticks(range(4), ["C1", "C2", "C3", "C4"]); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Madrid random CV pooled confusion proportions"); fig.colorbar(image, ax=ax); fig.tight_layout()
    fig.savefig(figure_dir / "madrid_cv_confusion.png", dpi=180); plt.close(fig)


def smoke_case(model_path: Path, cache_path: Path, n: int, trial: int = 0) -> dict:
    package = load_stage1(model_path)
    with np.load(cache_path, allow_pickle=False) as data, np.load(SPLIT_PATH, allow_pickle=False) as splits:
        x, y, ids = data["amsterdam_x"], data["amsterdam_y"], data["amsterdam_pixel_keys"]
        frame = unscaled_frame(x, ids, data["scaler_mean"], data["scaler_scale"])
        support_idx, query_idx = splits[f"support_{n}_{trial}"], splits[f"query_{n}_{trial}"]
        support, query = frame.iloc[support_idx].reset_index(drop=True), frame.iloc[query_idx].reset_index(drop=True)
        assert "age_class" not in query and "weighted_mean_year" not in query
        state = adapt(package, support, y[support_idx], n)
        first = predict(package, state, query)
        second = predict(package, state, query)
        deterministic = first.equals(second)
        score = f1_score(y[query_idx], first["predicted_age_class"], labels=CLASS_ORDER, average="macro", zero_division=0)
        return {"n": n, "trial": trial, "branch": state.branch, "query_rows": len(query), "query_columns": list(query.columns), "query_labels_available_to_predict": False, "deterministic_repeat": deterministic, "macro_f1_scored_after_prediction": float(score), "prediction_sha256": hashlib.sha256(first.to_csv(index=False).encode()).hexdigest()}


def raw_unlabelled_extraction_check() -> dict:
    with np.load(CACHE_PATH, allow_pickle=False) as cache:
        selected = cache["amsterdam_pixel_keys"][:4]
    raw = pd.read_parquet(ROOT / "Data" / "amsterdam_data.parquet")
    keep = np.zeros(len(raw), dtype=bool)
    for px, py in selected: keep |= raw["px_key"].eq(px).to_numpy() & raw["py_key"].eq(py).to_numpy()
    raw = raw.loc[keep].drop(columns=[c for c in ["weighted_mean_year", "age_class"] if c in raw.columns])
    output = extract_spectral_features(raw)
    return {"pixels": len(output), "input_target_columns": [], "output_target_columns": [c for c in output if c in {"weighted_mean_year", "age_class"}], "feature_count": len(feature_columns()), "passed": len(output) == 4 and not any(c in output for c in ["weighted_mean_year", "age_class"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-child", action="store_true")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()
    if args.smoke_child:
        print(json.dumps(smoke_case(args.model, args.cache, args.n), sort_keys=True)); return

    started = time.time(); OUTPUT.mkdir(parents=True, exist_ok=True); MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    source_files = [ROOT / "src" / "final_pipeline.py", Path(__file__), CACHE_PATH, SPLIT_PATH, ROOT / "Data" / "madrid_train.parquet", ROOT / "Data" / "amsterdam_data.parquet"]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_files}
    with np.load(CACHE_PATH, allow_pickle=False) as cache:
        mx, my = cache["madrid_x"], cache["madrid_y"]
        ax, ay, aids = cache["amsterdam_x"], cache["amsterdam_y"], cache["amsterdam_pixel_keys"]
        names, mean, scale = cache["feature_names"].tolist(), cache["scaler_mean"], cache["scaler_scale"]
    if names != feature_columns(): raise RuntimeError("cached feature order does not match frozen code")
    scaler = fitted_scaler_from_statistics(mean, scale, len(my))
    package = train_stage1(mx, my, MODEL_PATH, fitted_scaler=scaler, features_are_scaled=True, source_hashes=hashes)
    reloaded = load_stage1(MODEL_PATH)
    madrid = madrid_cv(mx, my, mean, scale)
    amsterdam, arrays = evaluate_amsterdam(reloaded, ax, ay, aids, mean, scale)
    np.savez_compressed(DETAIL_PATH, **arrays)
    raw_check = raw_unlabelled_extraction_check()
    smoke = [smoke_case(MODEL_PATH, CACHE_PATH, 25), smoke_case(MODEL_PATH, CACHE_PATH, 200)]
    child = subprocess.run([sys.executable, str(Path(__file__)), "--smoke-child", "--model", str(MODEL_PATH), "--cache", str(CACHE_PATH), "--n", "200"], check=True, text=True, capture_output=True)
    fresh_process = json.loads(child.stdout.strip().splitlines()[-1])
    make_figures(amsterdam, madrid)
    result = {
        "scope": "authoritative reproduced final-submission results; Amsterdam values are internal development results, not organiser held-out scores",
        "method": "frozen two-tier Madrid Random Forest transfer",
        "primary_submission": {"amsterdam": amsterdam},
        "development_benchmarks": benchmark_values(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__, "scikit_learn": sklearn.__version__, "pyarrow": pyarrow.__version__, "platform": platform.platform()},
        "source_and_input_sha256": hashes,
        "model_artifact": {"path": str(MODEL_PATH.relative_to(ROOT)), "sha256": sha256(MODEL_PATH), "reload_verified": package["metadata"]["configuration_sha256"] == reloaded["metadata"]["configuration_sha256"], "metadata": reloaded["metadata"]},
        "madrid_stage1_random_cv": madrid,
        "amsterdam_internal_development": amsterdam,
        "trial_array_artifact": str(DETAIL_PATH.relative_to(ROOT)),
        "development_benchmarks_not_primary_submission": benchmark_values(),
        "smoke_tests": smoke,
        "fresh_process_smoke_test": fresh_process,
        "unlabelled_raw_feature_extraction_check": raw_check,
        "artifact_hashes": {"stage1_model": sha256(MODEL_PATH), "trial_details": sha256(DETAIL_PATH)},
        "elapsed_seconds": time.time() - started,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
