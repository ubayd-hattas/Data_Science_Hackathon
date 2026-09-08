"""Faithful, non-notebook reproduction of organiser preprocessing and baseline."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
BASE_BANDS = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
INDEX_COLS = ["NDVI", "NDBI", "UI", "MNDWI", "BSI"]
BLUE_MAX = 15_000
CLASS_EVENTS = {"Amsterdam": 1945, "Madrid": 1960}
EARLY_CUTOFF = 2003
LATE_START = 2004
PID = ["px_key", "py_key"]


def assign_age_class(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["age_class"] = pd.NA
    for city, event in CLASS_EVENTS.items():
        mask = out["city"] == city
        if mask.any():
            out.loc[mask, "age_class"] = pd.cut(
                out.loc[mask, "weighted_mean_year"],
                bins=[-np.inf, event, 1984, 2004, np.inf],
                labels=[1, 2, 3, 4],
            ).astype("Int64")
    return out


def get_best_obs_band(data: pd.DataFrame, band: str) -> np.ndarray:
    values = np.full(len(data), np.nan)
    for slot in [1, 2, 3]:
        band_col = f"{band}_{slot}"
        qa_col = f"qa_valid_{slot}"
        if band_col not in data.columns or qa_col not in data.columns:
            continue
        valid = data[qa_col].fillna(False).values
        values = np.where(
            np.isnan(values) & valid,
            data[band_col].fillna(np.nan).values,
            values,
        )
    return values


def make_flat_view(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    keep = [
        column
        for column in [
            "pixel_id",
            "city",
            "year",
            "px_key",
            "py_key",
            "weighted_mean_year",
            "age_class",
        ]
        if column in data.columns
    ]
    out = data[keep].copy()
    for band in BASE_BANDS:
        out[band] = get_best_obs_band(data, band)
    out = out.dropna(subset=BASE_BANDS)
    before = len(out)
    out = out[out["Blue"] <= BLUE_MAX].copy()
    return out, before - len(out)


def period_coverage(flat: pd.DataFrame) -> pd.DataFrame:
    pixels = flat[PID].drop_duplicates()
    early = (
        flat[flat["year"] <= EARLY_CUTOFF]
        .drop_duplicates(PID)
        .assign(has_early_data=1.0)
    )
    late = (
        flat[flat["year"] >= LATE_START]
        .drop_duplicates(PID)
        .assign(has_late_data=1.0)
    )
    return (
        pixels.merge(early[PID + ["has_early_data"]], on=PID, how="left")
        .merge(late[PID + ["has_late_data"]], on=PID, how="left")
        .fillna(0.0)
    )


def fill_time_series_gaps(flat: pd.DataFrame) -> pd.DataFrame:
    all_years = list(range(int(flat["year"].min()), int(flat["year"].max()) + 1))
    pixels = flat[PID + ["city", "weighted_mean_year", "age_class"]].drop_duplicates(PID)
    grid = pixels.assign(_k=1).merge(
        pd.DataFrame({"year": all_years, "_k": 1}), on="_k"
    ).drop("_k", axis=1)
    merged = grid.merge(flat[PID + ["year"] + BASE_BANDS], on=PID + ["year"], how="left")
    merged = merged.sort_values(PID + ["year"]).reset_index(drop=True)

    # Notebook 3 applies Series.interpolate().bfill().ffill() separately to each
    # pixel and band. The complete grid guarantees a rectangular pixel x year
    # matrix, so applying the same operations across matrix rows is equivalent
    # and avoids hundreds of thousands of Python group callbacks.
    year_count = len(all_years)
    pixel_count = len(pixels)
    for band in BASE_BANDS:
        matrix = merged[band].to_numpy().reshape(pixel_count, year_count)
        filled = pd.DataFrame(matrix).interpolate(axis=1).bfill(axis=1).ffill(axis=1)
        merged[band] = filled.to_numpy().reshape(-1)
    return merged


def compute_spectral_indices(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    blue, green, red = frame["Blue"], frame["Green"], frame["Red"]
    nir, swir1, swir2 = frame["NIR"], frame["SWIR1"], frame["SWIR2"]
    eps = 1e-6
    frame["NDVI"] = (nir - red) / (nir + red + eps)
    frame["NDBI"] = (swir1 - nir) / (swir1 + nir + eps)
    frame["UI"] = (swir2 - nir) / (swir2 + nir + eps)
    frame["MNDWI"] = (green - swir1) / (green + swir1 + eps)
    frame["BSI"] = ((swir1 + red) - (nir + blue)) / (
        (swir1 + red) + (nir + blue) + eps
    )
    return frame


def build_pixel_features(flat: pd.DataFrame) -> pd.DataFrame:
    series = flat[flat["age_class"].notna()].copy().sort_values(PID + ["year"])
    group = series.groupby(PID + ["age_class"])
    overall = pd.concat(
        [
            group[BASE_BANDS + INDEX_COLS].mean().add_suffix("_mean"),
            group[BASE_BANDS + INDEX_COLS].std().add_suffix("_std"),
        ],
        axis=1,
    ).reset_index()
    overall["age_class"] = overall["age_class"].astype(int)

    early_group = series[series["year"] <= EARLY_CUTOFF].groupby(PID)
    early = pd.concat(
        [
            early_group[BASE_BANDS].mean().add_suffix("_early_mean"),
            early_group[BASE_BANDS].std().add_suffix("_early_std"),
        ],
        axis=1,
    ).reset_index()
    late_group = series[series["year"] >= LATE_START].groupby(PID)
    late = pd.concat(
        [
            late_group[BASE_BANDS].mean().add_suffix("_late_mean"),
            late_group[BASE_BANDS].std().add_suffix("_late_std"),
        ],
        axis=1,
    ).reset_index()

    for band in BASE_BANDS:
        series[f"d_{band}"] = series.groupby(PID)[band].diff()
    diff_cols = [f"d_{band}" for band in BASE_BANDS]
    diff_group = series.groupby(PID)[diff_cols]
    diff = pd.concat(
        [
            diff_group.mean().add_suffix("_mean"),
            diff_group.std().add_suffix("_std"),
        ],
        axis=1,
    ).reset_index()
    years = series.groupby(PID)["weighted_mean_year"].first().reset_index()

    result = overall
    for extra in [early, late, diff, years]:
        result = result.merge(extra, on=PID, how="left")
    return result


def feature_columns() -> tuple[list[str], dict[str, list[str]]]:
    groups = {
        "overall": [f"{b}_mean" for b in BASE_BANDS]
        + [f"{b}_std" for b in BASE_BANDS],
        "indices": [f"{i}_mean" for i in INDEX_COLS]
        + [f"{i}_std" for i in INDEX_COLS],
        "early": [f"{b}_early_mean" for b in BASE_BANDS]
        + [f"{b}_early_std" for b in BASE_BANDS],
        "late": [f"{b}_late_mean" for b in BASE_BANDS]
        + [f"{b}_late_std" for b in BASE_BANDS],
        "yoy": [f"d_{b}_mean" for b in BASE_BANDS]
        + [f"d_{b}_std" for b in BASE_BANDS],
        "indicators": ["has_early_data", "has_late_data"],
    }
    return [feature for group in groups.values() for feature in group], groups


def preprocess_city(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = assign_age_class(pd.read_parquet(path))
    flat, haze_removed = make_flat_view(raw)
    indicators = period_coverage(flat)
    observed_rows = len(flat)
    complete = compute_spectral_indices(fill_time_series_gaps(flat))
    features = build_pixel_features(complete).merge(indicators, on=PID, how="left")
    details = {
        "raw_shape": list(raw.shape),
        "observed_pixel_year_rows": observed_rows,
        "haze_rows_removed": haze_removed,
        "gap_filled_rows": len(complete),
        "pixels": len(features),
        "class_distribution": {
            str(int(k)): int(v)
            for k, v in features["age_class"].value_counts().sort_index().items()
        },
    }
    return features, details


def prototype_predict(
    support_x: np.ndarray, support_y: np.ndarray, query_x: np.ndarray
) -> np.ndarray:
    classes = np.unique(support_y)
    prototypes = np.stack([support_x[support_y == cls].mean(axis=0) for cls in classes])
    distances = np.linalg.norm(query_x[:, None, :] - prototypes[None, :, :], axis=2)
    return classes[distances.argmin(axis=1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    started = time.time()

    print("Preprocessing Madrid...", flush=True)
    madrid, madrid_details = preprocess_city(ROOT / "Data" / "madrid_train.parquet")
    print("Preprocessing Amsterdam...", flush=True)
    amsterdam, amsterdam_details = preprocess_city(ROOT / "Data" / "amsterdam_data.parquet")
    feature_names, groups = feature_columns()

    madrid_x_raw = madrid[feature_names].to_numpy(dtype=np.float64)
    madrid_y = madrid["age_class"].to_numpy(dtype=int)
    amsterdam_x_raw = amsterdam[feature_names].to_numpy(dtype=np.float64)
    amsterdam_y = amsterdam["age_class"].to_numpy(dtype=int)

    # This deliberately mirrors Notebook 3: fit once on all Madrid pixels before CV.
    scaler = StandardScaler()
    madrid_x = scaler.fit_transform(madrid_x_raw)
    amsterdam_x = scaler.transform(amsterdam_x_raw)

    params = {
        "n_estimators": args.trees,
        "class_weight": "balanced",
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
    }
    splitter = RepeatedStratifiedKFold(
        n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed
    )
    fold_scores = []
    confusion_total = np.zeros((4, 4), dtype=np.int64)
    total_fits = args.folds * args.repeats
    for fit_number, (train_idx, validation_idx) in enumerate(
        splitter.split(madrid_x, madrid_y), 1
    ):
        model = RandomForestClassifier(**params)
        model.fit(madrid_x[train_idx], madrid_y[train_idx])
        prediction = model.predict(madrid_x[validation_idx])
        score = f1_score(madrid_y[validation_idx], prediction, average="macro")
        fold_scores.append(float(score))
        confusion_total += confusion_matrix(
            madrid_y[validation_idx], prediction, labels=[1, 2, 3, 4]
        )
        print(f"CV {fit_number}/{total_fits}: {score:.4f}", flush=True)

    final_model = RandomForestClassifier(**params)
    final_model.fit(madrid_x, madrid_y)
    madrid_in_sample_accuracy = accuracy_score(madrid_y, final_model.predict(madrid_x))
    amsterdam_zero_prediction = final_model.predict(amsterdam_x)
    amsterdam_zero_f1 = f1_score(
        amsterdam_y, amsterdam_zero_prediction, average="macro"
    )

    shot_sizes = [5, 10, 25, 50, 100, 200]
    rng = np.random.default_rng(args.seed)
    prototype_results: dict[str, list[float]] = {str(n): [] for n in shot_sizes}
    classes = np.unique(amsterdam_y)
    for shots in shot_sizes:
        for _ in range(args.trials):
            support = []
            for cls in classes:
                candidates = np.where(amsterdam_y == cls)[0]
                chosen = rng.choice(candidates, min(shots, len(candidates)), replace=False)
                support.extend(chosen.tolist())
            support_idx = np.asarray(support)
            query_mask = np.ones(len(amsterdam_y), dtype=bool)
            query_mask[support_idx] = False
            prediction = prototype_predict(
                amsterdam_x[support_idx], amsterdam_y[support_idx], amsterdam_x[query_mask]
            )
            prototype_results[str(shots)].append(
                float(
                    f1_score(
                        amsterdam_y[query_mask],
                        prediction,
                        average="macro",
                        zero_division=0,
                    )
                )
            )
        values = prototype_results[str(shots)]
        print(
            f"Amsterdam {shots} shots/class: {np.mean(values):.4f} +/- {np.std(values):.4f}",
            flush=True,
        )

    results = {
        "implementation": "faithful organiser notebook 3 and 4 reproduction",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "scikit_learn": sklearn.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "seed": args.seed,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "feature_groups": groups,
        "preprocessing": {
            "Madrid": madrid_details,
            "Amsterdam": amsterdam_details,
            "scaler_fit": "all Madrid pixels before Madrid CV; Amsterdam only transformed",
        },
        "model": {"type": "RandomForestClassifier", "parameters": params},
        "madrid_cv": {
            "folds": args.folds,
            "repeats": args.repeats,
            "scores": fold_scores,
            "macro_f1_mean": float(np.mean(fold_scores)),
            "macro_f1_std_population": float(np.std(fold_scores)),
            "confusion_matrix_counts": confusion_total.tolist(),
            "confusion_matrix_row_proportions": (
                confusion_total / confusion_total.sum(axis=1, keepdims=True)
            ).tolist(),
        },
        "madrid_in_sample_accuracy": float(madrid_in_sample_accuracy),
        "amsterdam_zero_shot_macro_f1": float(amsterdam_zero_f1),
        "amsterdam_prototype": {
            shots: {
                "scores": values,
                "macro_f1_mean": float(np.mean(values)),
                "macro_f1_std_population": float(np.std(values)),
            }
            for shots, values in prototype_results.items()
        },
        "elapsed_seconds": time.time() - started,
    }
    output = ROOT / "outputs" / "baseline_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
