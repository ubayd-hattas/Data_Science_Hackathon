"""Frozen Madrid -> Amsterdam two-tier Random Forest submission pipeline.

Public API: train_stage1, load_stage1, adapt, predict, and
extract_spectral_features. Query labels are neither accepted nor needed.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

BASE_BANDS = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
INDEX_COLS = ["NDVI", "NDBI", "UI", "MNDWI", "BSI"]
PIXEL_ID_COLUMNS = ["px_key", "py_key"]
CLASS_ORDER = np.array([1, 2, 3, 4], dtype=np.int64)
CLASS_EVENTS = {"Amsterdam": 1945, "Madrid": 1960}
BLUE_MAX = 15_000
EARLY_CUTOFF = 2003
LATE_START = 2004
TEMPORAL_YEAR_CONVENTION = "inclusive minimum-to-maximum year present in the supplied dataset"
RF_PARAMS = {"n_estimators": 500, "class_weight": "balanced", "random_state": 42, "n_jobs": -1}
REGIME_SWITCH_N = 50
PROTOTYPE_TEMPERATURE = 1.0
SCORE_EPSILON = 1e-6
MODEL_SEED = 42
EVALUATION_SEED = 42
VALID_BUDGETS = (5, 25, 50, 100, 200)


def feature_columns() -> list[str]:
    """Return the frozen 60-feature order."""
    return (
        [f"{b}_mean" for b in BASE_BANDS]
        + [f"{b}_std" for b in BASE_BANDS]
        + [f"{i}_mean" for i in INDEX_COLS]
        + [f"{i}_std" for i in INDEX_COLS]
        + [f"{b}_early_mean" for b in BASE_BANDS]
        + [f"{b}_early_std" for b in BASE_BANDS]
        + [f"{b}_late_mean" for b in BASE_BANDS]
        + [f"{b}_late_std" for b in BASE_BANDS]
        + [f"d_{b}_mean" for b in BASE_BANDS]
        + [f"d_{b}_std" for b in BASE_BANDS]
        + ["has_early_data", "has_late_data"]
    )


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def _best_valid_band(data: pd.DataFrame, band: str) -> np.ndarray:
    values = np.full(len(data), np.nan, dtype=np.float64)
    found = False
    for slot in (1, 2, 3):
        band_col, qa_col = f"{band}_{slot}", f"qa_valid_{slot}"
        if band_col not in data.columns or qa_col not in data.columns:
            continue
        found = True
        valid = data[qa_col].fillna(False).to_numpy(dtype=bool)
        candidate = pd.to_numeric(data[band_col], errors="coerce").to_numpy(dtype=np.float64)
        values = np.where(np.isnan(values) & valid, candidate, values)
    if not found:
        raise ValueError(f"raw records contain no observation columns for band {band!r}")
    return values


def extract_spectral_features(raw_records: pd.DataFrame) -> pd.DataFrame:
    """Create the frozen 60 features from labelled or genuinely unlabelled raw rows.

    Required fields are city, year, px_key, py_key, qa_valid_N and six band_N
    groups for at least one N in 1..3. Target columns, if present, are ignored.
    Each pixel must be supplied with its complete available annual history.
    """
    _require_columns(raw_records, ["city", "year", *PIXEL_ID_COLUMNS], "raw query data")
    if raw_records.empty:
        raise ValueError("raw query data is empty")
    data = raw_records.copy()
    if data[PIXEL_ID_COLUMNS].isna().any().any():
        raise ValueError("pixel identifiers may not be missing")

    flat = data[["city", "year", *PIXEL_ID_COLUMNS]].copy()
    for band in BASE_BANDS:
        flat[band] = _best_valid_band(data, band)
    flat = flat.dropna(subset=BASE_BANDS)
    flat = flat[flat["Blue"] <= BLUE_MAX].copy()
    if flat.empty:
        raise ValueError("no valid observations remain after QA and Blue <= 15000 cleaning")

    pixels = flat[[*PIXEL_ID_COLUMNS, "city"]].drop_duplicates(PIXEL_ID_COLUMNS)
    if pixels.duplicated(PIXEL_ID_COLUMNS).any():
        raise ValueError("a pixel maps to more than one city")
    has_early = flat[flat["year"] <= EARLY_CUTOFF][PIXEL_ID_COLUMNS].drop_duplicates().assign(has_early_data=1.0)
    has_late = flat[flat["year"] >= LATE_START][PIXEL_ID_COLUMNS].drop_duplicates().assign(has_late_data=1.0)
    indicators = pixels[PIXEL_ID_COLUMNS].merge(has_early, on=PIXEL_ID_COLUMNS, how="left").merge(has_late, on=PIXEL_ID_COLUMNS, how="left").fillna(0.0)

    years = np.arange(int(flat["year"].min()), int(flat["year"].max()) + 1)
    grid = pixels.assign(_join=1).merge(pd.DataFrame({"year": years, "_join": 1}), on="_join").drop(columns="_join")
    complete = grid.merge(flat[[*PIXEL_ID_COLUMNS, "year", *BASE_BANDS]], on=[*PIXEL_ID_COLUMNS, "year"], how="left")
    complete = complete.sort_values([*PIXEL_ID_COLUMNS, "year"]).reset_index(drop=True)
    for band in BASE_BANDS:
        complete[band] = complete.groupby(PIXEL_ID_COLUMNS, sort=False)[band].transform(lambda s: s.interpolate(method="linear").bfill().ffill())
    if complete[BASE_BANDS].isna().any().any():
        raise ValueError("at least one pixel has no usable spectral history")

    blue, green, red = complete["Blue"], complete["Green"], complete["Red"]
    nir, swir1, swir2 = complete["NIR"], complete["SWIR1"], complete["SWIR2"]
    eps = 1e-6
    complete["NDVI"] = (nir - red) / (nir + red + eps)
    complete["NDBI"] = (swir1 - nir) / (swir1 + nir + eps)
    complete["UI"] = (swir2 - nir) / (swir2 + nir + eps)
    complete["MNDWI"] = (green - swir1) / (green + swir1 + eps)
    complete["BSI"] = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + eps)

    group = complete.groupby(PIXEL_ID_COLUMNS, sort=True)
    overall = pd.concat([group[BASE_BANDS + INDEX_COLS].mean().add_suffix("_mean"), group[BASE_BANDS + INDEX_COLS].std().add_suffix("_std")], axis=1).reset_index()
    early_group = complete[complete["year"] <= EARLY_CUTOFF].groupby(PIXEL_ID_COLUMNS, sort=True)
    early = pd.concat([early_group[BASE_BANDS].mean().add_suffix("_early_mean"), early_group[BASE_BANDS].std().add_suffix("_early_std")], axis=1).reset_index()
    late_group = complete[complete["year"] >= LATE_START].groupby(PIXEL_ID_COLUMNS, sort=True)
    late = pd.concat([late_group[BASE_BANDS].mean().add_suffix("_late_mean"), late_group[BASE_BANDS].std().add_suffix("_late_std")], axis=1).reset_index()
    for band in BASE_BANDS:
        complete[f"d_{band}"] = complete.groupby(PIXEL_ID_COLUMNS, sort=False)[band].diff()
    difference_columns = [f"d_{b}" for b in BASE_BANDS]
    difference_group = complete.groupby(PIXEL_ID_COLUMNS, sort=True)[difference_columns]
    difference = pd.concat([difference_group.mean().add_suffix("_mean"), difference_group.std().add_suffix("_std")], axis=1).reset_index()

    result = pixels.merge(overall, on=PIXEL_ID_COLUMNS, how="inner")
    for extra in (early, late, difference, indicators):
        result = result.merge(extra, on=PIXEL_ID_COLUMNS, how="left")
    result = result.sort_values(PIXEL_ID_COLUMNS).reset_index(drop=True)
    result = result[[*PIXEL_ID_COLUMNS, "city", *feature_columns()]]
    if not np.isfinite(result[feature_columns()].to_numpy(dtype=np.float64)).all():
        raise ValueError("non-finite engineered feature value")
    return result


def labels_from_raw(raw_records: pd.DataFrame) -> pd.DataFrame:
    """Derive one label per pixel; intentionally separate from feature extraction."""
    _require_columns(raw_records, ["city", "weighted_mean_year", *PIXEL_ID_COLUMNS], "labelled data")
    rows = raw_records[[*PIXEL_ID_COLUMNS, "city", "weighted_mean_year"]].drop_duplicates()
    if rows.duplicated(PIXEL_ID_COLUMNS).any():
        raise ValueError("construction target is not constant within a pixel")
    if not rows["city"].isin(CLASS_EVENTS).all():
        unknown = sorted(rows.loc[~rows["city"].isin(CLASS_EVENTS), "city"].unique())
        raise ValueError(f"no class-boundary convention for cities: {unknown}")
    labels = np.empty(len(rows), dtype=np.int64)
    for city, event in CLASS_EVENTS.items():
        mask = rows["city"].eq(city).to_numpy()
        labels[mask] = pd.cut(rows.loc[mask, "weighted_mean_year"], [-np.inf, event, 1984, 2004, np.inf], labels=CLASS_ORDER).astype(int)
    rows["age_class"] = labels
    return rows[[*PIXEL_ID_COLUMNS, "age_class"]].sort_values(PIXEL_ID_COLUMNS).reset_index(drop=True)


def fitted_scaler_from_statistics(mean: np.ndarray, scale: np.ndarray, n_samples: int) -> StandardScaler:
    """Reconstruct the verified Madrid scaler stored in features_60.npz."""
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(mean, dtype=np.float64)
    scaler.scale_ = np.asarray(scale, dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    scaler.n_samples_seen_ = int(n_samples)
    return scaler


def _config_metadata(feature_names: list[str], training_rows: int, dtype: str) -> dict[str, Any]:
    return {
        "method": "frozen_two_tier_madrid_random_forest_transfer",
        "transfer_direction": "Madrid -> Amsterdam",
        "rf_parameters": RF_PARAMS,
        "class_order": CLASS_ORDER.tolist(),
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "training_rows": int(training_rows),
        "training_dtype": dtype,
        "model_seed": MODEL_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "regime_switch": {"product_if_n_lte": REGIME_SWITCH_N, "modal_leaf_if_n_gt": REGIME_SWITCH_N},
        "score_product": {"temperature": PROTOTYPE_TEMPERATURE, "epsilon": SCORE_EPSILON, "target_prior": "uniform", "source_prior_adjustment": "multiply RF scores by uniform/source empirical class frequency and renormalise"},
        "feature_generation": {"base_bands": BASE_BANDS, "indices": INDEX_COLS, "best_observation_slots": [1, 2, 3], "blue_max_inclusive": BLUE_MAX, "early_year_max": EARLY_CUTOFF, "late_year_min": LATE_START, "gap_filling": "linear interpolation, then backward fill and forward fill within pixel", "temporal_year_convention": TEMPORAL_YEAR_CONVENTION, "pandas_std_ddof": 1},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def train_stage1(features: pd.DataFrame | np.ndarray, labels: np.ndarray, output_path: str | Path, *, fitted_scaler: StandardScaler | None = None, features_are_scaled: bool = False, source_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    """Fit and serialize the one final Madrid Stage 1 forest."""
    names = feature_columns()
    matrix = features[names].to_numpy(dtype=np.float64) if isinstance(features, pd.DataFrame) else np.asarray(features)
    labels = np.asarray(labels, dtype=np.int64)
    if matrix.shape != (len(labels), len(names)):
        raise ValueError(f"expected {(len(labels), len(names))} feature matrix, got {matrix.shape}")
    if not np.array_equal(np.unique(labels), CLASS_ORDER):
        raise ValueError("Madrid labels must contain exactly classes 1, 2, 3 and 4")
    if features_are_scaled:
        if fitted_scaler is None:
            raise ValueError("a fitted scaler is required when features_are_scaled=True")
        scaled, scaler = np.asarray(matrix, dtype=np.float32), fitted_scaler
    else:
        scaler = StandardScaler().fit(matrix) if fitted_scaler is None else fitted_scaler
        scaled = scaler.transform(matrix).astype(np.float32, copy=False)
    forest = RandomForestClassifier(**RF_PARAMS)
    forest.fit(scaled, labels)
    prior = {str(c): float(np.mean(labels == c)) for c in CLASS_ORDER}
    metadata = _config_metadata(names, len(labels), str(scaled.dtype))
    metadata["source_hashes"] = source_hashes or {}
    metadata["madrid_class_prior"] = prior
    metadata["configuration_sha256"] = _canonical_hash(metadata)
    package = {"format_version": 1, "model": forest, "scaler": scaler, "metadata": metadata}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(package, stream, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="ascii")
    return package


def load_stage1(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as stream:
        package = pickle.load(stream)
    if package.get("format_version") != 1:
        raise ValueError("unsupported Stage 1 package format")
    metadata = package["metadata"]
    expected = metadata["configuration_sha256"]
    check = dict(metadata)
    del check["configuration_sha256"]
    if _canonical_hash(check) != expected:
        raise ValueError("Stage 1 configuration checksum mismatch")
    if metadata["feature_names"] != feature_columns() or not np.array_equal(package["model"].classes_, CLASS_ORDER):
        raise ValueError("Stage 1 feature or class order mismatch")
    return package


def _feature_matrix(package: dict[str, Any], frame: pd.DataFrame | np.ndarray) -> np.ndarray:
    names = package["metadata"]["feature_names"]
    if isinstance(frame, pd.DataFrame):
        _require_columns(frame, names, "engineered features")
        matrix = frame[names].to_numpy(dtype=np.float64)
    else:
        matrix = np.asarray(frame, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError(f"expected a two-dimensional matrix with {len(names)} columns")
    if not np.isfinite(matrix).all():
        raise ValueError("feature matrix contains non-finite values")
    return package["scaler"].transform(matrix).astype(np.float32, copy=False)


def _ids(frame: pd.DataFrame | np.ndarray, explicit_ids: np.ndarray | None) -> np.ndarray | None:
    if explicit_ids is not None:
        value = np.asarray(explicit_ids)
    elif isinstance(frame, pd.DataFrame) and all(c in frame.columns for c in PIXEL_ID_COLUMNS):
        value = frame[PIXEL_ID_COLUMNS].to_numpy()
    else:
        return None
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError("pixel IDs must have shape (rows, 2) for px_key and py_key")
    if len({tuple(v) for v in value.tolist()}) != len(value):
        raise ValueError("pixel IDs must be unique")
    return value


@dataclass(frozen=True)
class AdaptedState:
    n_per_class: int
    branch: str
    support_scaled: np.ndarray
    support_labels: np.ndarray
    support_ids: np.ndarray | None
    support_leaves: np.ndarray | None


def adapt(package: dict[str, Any], support_features: pd.DataFrame | np.ndarray, support_labels: np.ndarray | None, n: int, *, support_ids: np.ndarray | None = None) -> AdaptedState:
    """Validate and prepare exactly n labelled Amsterdam examples per class."""
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if support_labels is None and isinstance(support_features, pd.DataFrame) and "age_class" in support_features:
        support_labels = support_features["age_class"].to_numpy()
    if support_labels is None:
        raise ValueError("support labels are required")
    labels = np.asarray(support_labels, dtype=np.int64)
    scaled = _feature_matrix(package, support_features)
    if len(labels) != len(scaled) or len(labels) != 4 * n:
        raise ValueError(f"support must contain exactly {n} examples for each of four classes")
    counts = {int(c): int(np.sum(labels == c)) for c in CLASS_ORDER}
    if counts != {int(c): int(n) for c in CLASS_ORDER} or not np.isin(labels, CLASS_ORDER).all():
        raise ValueError(f"support class counts must each equal n={n}; received {counts}")
    ids = _ids(support_features, support_ids)
    branch = "modal_leaf_agreement" if n > REGIME_SWITCH_N else "score_product"
    leaves = package["model"].apply(scaled) if branch == "modal_leaf_agreement" else None
    return AdaptedState(n, branch, scaled, labels, ids, leaves)


def _score_product_predict(package: dict[str, Any], state: AdaptedState, query_scaled: np.ndarray) -> np.ndarray:
    rf_score = package["model"].predict_proba(query_scaled)
    prior = package["metadata"]["madrid_class_prior"]
    source_prior = np.array([prior[str(c)] for c in CLASS_ORDER])
    adjusted = rf_score * ((1.0 / len(CLASS_ORDER)) / source_prior)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    prototypes = np.stack([state.support_scaled[state.support_labels == c].mean(axis=0) for c in CLASS_ORDER])
    distances = np.linalg.norm(query_scaled[:, None, :] - prototypes[None, :, :], axis=2)
    logits = -distances / PROTOTYPE_TEMPERATURE
    logits -= logits.max(axis=1, keepdims=True)
    target_score = np.exp(logits)
    target_score /= target_score.sum(axis=1, keepdims=True)
    product = adjusted * target_score + SCORE_EPSILON
    product /= product.sum(axis=1, keepdims=True)
    return CLASS_ORDER[product.argmax(axis=1)]


def _modal_leaf_predict(package: dict[str, Any], state: AdaptedState, query_scaled: np.ndarray) -> np.ndarray:
    query_leaves = package["model"].apply(query_scaled)
    support_leaves = state.support_leaves
    assert support_leaves is not None
    modes = np.zeros((4, support_leaves.shape[1]), dtype=support_leaves.dtype)
    for class_index, cls in enumerate(CLASS_ORDER):
        rows = support_leaves[state.support_labels == cls]
        for tree_index in range(rows.shape[1]):
            values, counts = np.unique(rows[:, tree_index], return_counts=True)
            modes[class_index, tree_index] = values[np.argmax(counts)]
    agreement = np.stack([(query_leaves == modes[i]).mean(axis=1) for i in range(4)], axis=1)
    return CLASS_ORDER[agreement.argmax(axis=1)]


def predict(package: dict[str, Any], state: AdaptedState, query_features: pd.DataFrame | np.ndarray, *, query_ids: np.ndarray | None = None) -> pd.DataFrame:
    """Predict one class per unlabelled query pixel, preserving row order and IDs."""
    if isinstance(query_features, pd.DataFrame):
        forbidden = [c for c in query_features.columns if c == "age_class" or "construction" in c.lower() or c == "weighted_mean_year"]
        if forbidden:
            raise ValueError(f"query input must be unlabelled; remove target columns {forbidden}")
    scaled = _feature_matrix(package, query_features)
    ids = _ids(query_features, query_ids)
    if state.support_ids is not None and ids is not None:
        support_set = {tuple(v) for v in state.support_ids.tolist()}
        overlap = [tuple(v) for v in ids.tolist() if tuple(v) in support_set]
        if overlap:
            raise ValueError(f"support/query pixel IDs overlap; first overlap is {overlap[0]}")
    labels = _score_product_predict(package, state, scaled) if state.branch == "score_product" else _modal_leaf_predict(package, state, scaled)
    result = pd.DataFrame({"predicted_age_class": labels.astype(np.int64)})
    if ids is not None:
        result.insert(0, "py_key", ids[:, 1])
        result.insert(0, "px_key", ids[:, 0])
    return result
