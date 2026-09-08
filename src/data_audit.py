"""Read-only audit of the organiser parquet datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "Madrid": ROOT / "Data" / "madrid_train.parquet",
    "Amsterdam": ROOT / "Data" / "amsterdam_data.parquet",
}
CLASS_EVENTS = {"Madrid": 1960, "Amsterdam": 1945}


def assign_age_class(frame: pd.DataFrame, city: str) -> pd.Series:
    return pd.cut(
        frame["weighted_mean_year"],
        bins=[-np.inf, CLASS_EVENTS[city], 1984, 2004, np.inf],
        labels=[1, 2, 3, 4],
    ).astype("Int64")


def audit_dataset(city: str, path: Path) -> dict:
    parquet = pq.ParquetFile(path)
    frame = pd.read_parquet(path)
    labels = assign_age_class(frame, city)
    pixel_cols = ["px_key", "py_key"]
    unique_pixels = frame.drop_duplicates(pixel_cols).copy()
    unique_pixels["age_class"] = assign_age_class(unique_pixels, city)

    row_distribution = {
        str(int(k)): int(v) for k, v in labels.value_counts().sort_index().items()
    }
    pixel_distribution = {
        str(int(k)): int(v)
        for k, v in unique_pixels["age_class"].value_counts().sort_index().items()
    }
    dtype_counts = {
        str(k): int(v) for k, v in frame.dtypes.astype(str).value_counts().items()
    }
    per_column_dtypes = {column: str(dtype) for column, dtype in frame.dtypes.items()}

    pixel_year_duplicates = int(frame.duplicated(pixel_cols + ["year"]).sum())
    pixel_label_variation = int(
        (frame.groupby(pixel_cols)["weighted_mean_year"].nunique(dropna=False) > 1).sum()
    )
    exact_duplicate_rows = int(frame.duplicated().sum())

    return {
        "path": str(path.relative_to(ROOT)),
        "file_size_bytes": path.stat().st_size,
        "parquet_row_groups": parquet.num_row_groups,
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "columns": frame.columns.tolist(),
        "dtypes": per_column_dtypes,
        "dtype_counts": dtype_counts,
        "dataframe_memory_bytes_deep": int(frame.memory_usage(index=True, deep=True).sum()),
        "years": {
            "min": int(frame["year"].min()),
            "max": int(frame["year"].max()),
            "distinct": int(frame["year"].nunique()),
        },
        "cities": sorted(frame["city"].dropna().unique().tolist()),
        "unique_pixels": int(len(unique_pixels)),
        "target": {
            "source_column": "weighted_mean_year",
            "missing_rows": int(frame["weighted_mean_year"].isna().sum()),
            "min": float(frame["weighted_mean_year"].min()),
            "max": float(frame["weighted_mean_year"].max()),
            "mean": float(frame["weighted_mean_year"].mean()),
            "class_event_boundary": CLASS_EVENTS[city],
            "class_distribution_rows": row_distribution,
            "class_distribution_unique_pixels": pixel_distribution,
        },
        "quality_checks": {
            "exact_duplicate_rows": exact_duplicate_rows,
            "duplicate_pixel_year_rows": pixel_year_duplicates,
            "pixels_with_varying_target_over_years": pixel_label_variation,
            "coverage_below_0_15_rows": int((frame["coverage"] < 0.15).sum()),
            "usable_label_counts": {
                str(k): int(v)
                for k, v in frame["usable_label"].value_counts(dropna=False).items()
            },
            "qa_valid_true_rows_by_slot": {
                str(slot): int(frame[f"qa_valid_{slot}"].fillna(False).sum())
                for slot in (1, 2, 3)
            },
        },
    }


def main() -> None:
    result = {
        "datasets": {city: audit_dataset(city, path) for city, path in DATASETS.items()}
    }
    output = ROOT / "outputs" / "dataset_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
