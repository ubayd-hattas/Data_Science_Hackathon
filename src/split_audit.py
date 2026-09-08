"""Check exact and neighbouring-pixel separation under organiser-style random CV."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "Madrid": ROOT / "Data" / "madrid_train.parquet",
    "Amsterdam": ROOT / "Data" / "amsterdam_data.parquet",
}
EVENTS = {"Madrid": 1960, "Amsterdam": 1945}


def labels(years: pd.Series, event: int) -> np.ndarray:
    return pd.cut(
        years,
        [-np.inf, event, 1984, 2004, np.inf],
        labels=[1, 2, 3, 4],
    ).astype(int).to_numpy()


def audit(city: str, path: Path) -> dict:
    frame = pd.read_parquet(path, columns=["px_key", "py_key", "weighted_mean_year"])
    pixels = frame.drop_duplicates(["px_key", "py_key"]).reset_index(drop=True)
    y = labels(pixels["weighted_mean_year"], EVENTS[city])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = np.full(len(pixels), -1, dtype=np.int8)
    for fold_number, (_, validation) in enumerate(splitter.split(pixels, y)):
        fold[validation] = fold_number

    coords = list(zip(pixels["px_key"].astype(int), pixels["py_key"].astype(int)))
    index = {coord: position for position, coord in enumerate(coords)}
    neighbour_pairs = []
    for position, (x, y_coord) in enumerate(coords):
        for neighbour in ((x + 1, y_coord), (x, y_coord + 1)):
            other = index.get(neighbour)
            if other is not None:
                neighbour_pairs.append((position, other))
    pair_array = np.asarray(neighbour_pairs, dtype=np.int64)
    cross_fold = int((fold[pair_array[:, 0]] != fold[pair_array[:, 1]]).sum())
    return {
        "unique_pixels": len(pixels),
        "exact_coordinate_duplicates_after_deduplication": int(
            pixels.duplicated(["px_key", "py_key"]).sum()
        ),
        "orthogonally_adjacent_pixel_pairs": len(neighbour_pairs),
        "adjacent_pairs_assigned_to_different_folds": cross_fold,
        "adjacent_pairs_cross_fold_fraction": cross_fold / len(neighbour_pairs),
        "split": "StratifiedKFold(5, shuffle=True, random_state=42)",
    }


def main() -> None:
    result = {city: audit(city, path) for city, path in FILES.items()}
    output = ROOT / "outputs" / "split_audit.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
