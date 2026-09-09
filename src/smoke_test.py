"""Organizer-style held-out simulation using an external verified cache and split file."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from final_pipeline import adapt, feature_columns, load_stage1, predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--trial", type=int, default=0)
    args = parser.parse_args()
    package = load_stage1(args.model)
    with np.load(args.cache, allow_pickle=False) as cache, np.load(args.splits, allow_pickle=False) as splits:
        scaled = cache["amsterdam_x"]
        labels = cache["amsterdam_y"]
        ids = cache["amsterdam_pixel_keys"]
        raw_features = scaled.astype(np.float64) * cache["scaler_scale"] + cache["scaler_mean"]
        frame = pd.DataFrame(raw_features, columns=feature_columns())
        frame.insert(0, "py_key", ids[:, 1]); frame.insert(0, "px_key", ids[:, 0])
        support_idx = splits[f"support_{args.n}_{args.trial}"]
        query_idx = splits[f"query_{args.n}_{args.trial}"]
        support = frame.iloc[support_idx].reset_index(drop=True)
        query = frame.iloc[query_idx].reset_index(drop=True)
        state = adapt(package, support, labels[support_idx], args.n)
        first = predict(package, state, query)
        second = predict(package, state, query)
    print(json.dumps({
        "n": args.n,
        "branch": state.branch,
        "query_rows": len(query),
        "query_has_labels": any(c in query for c in ["age_class", "weighted_mean_year"]),
        "deterministic": first.equals(second),
        "prediction_sha256": hashlib.sha256(first.to_csv(index=False).encode()).hexdigest(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
