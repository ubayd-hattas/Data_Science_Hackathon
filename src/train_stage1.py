"""Train only the frozen final Madrid Stage 1 model from features_60.npz."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from final_pipeline import feature_columns, fitted_scaler_from_statistics, train_stage1


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("outputs/features_60.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/stage1_madrid_rf.pkl"))
    args = parser.parse_args()
    with np.load(args.cache, allow_pickle=False) as cache:
        if cache["feature_names"].tolist() != feature_columns():
            raise ValueError("cached feature order does not match frozen code")
        x, y = cache["madrid_x"], cache["madrid_y"]
        scaler = fitted_scaler_from_statistics(cache["scaler_mean"], cache["scaler_scale"], len(y))
    train_stage1(x, y, args.output, fitted_scaler=scaler, features_are_scaled=True, source_hashes={str(args.cache): file_hash(args.cache)})
    print(args.output)


if __name__ == "__main__":
    main()
