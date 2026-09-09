"""Command-line interface for frozen feature extraction and transfer prediction."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from final_pipeline import adapt, extract_spectral_features, labels_from_raw, load_stage1, predict


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen Madrid to Amsterdam RF transfer")
    commands = parser.add_subparsers(dest="command", required=True)
    extract_parser = commands.add_parser("extract", help="build the 60 features without labels")
    extract_parser.add_argument("--raw", required=True, type=Path)
    extract_parser.add_argument("--output", required=True, type=Path)
    predict_parser = commands.add_parser("predict", help="adapt on support and predict an unlabelled query")
    predict_parser.add_argument("--model", required=True, type=Path)
    for flag in ("support", "query"):
        predict_parser.add_argument(f"--{flag}", required=True, type=Path)
    predict_parser.add_argument("--n", required=True, type=int)
    predict_parser.add_argument("--output", required=True, type=Path)
    predict_parser.add_argument("--support-raw", action="store_true")
    predict_parser.add_argument("--query-raw", action="store_true")
    args = parser.parse_args()

    if args.command == "extract":
        output = extract_spectral_features(read_table(args.raw))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(args.output, index=False)
        return

    support_source, query_source = read_table(args.support), read_table(args.query)
    if args.support_raw:
        support = extract_spectral_features(support_source).merge(
            labels_from_raw(support_source), on=["px_key", "py_key"], how="inner"
        )
    else:
        support = support_source
    query = extract_spectral_features(query_source) if args.query_raw else query_source
    package = load_stage1(args.model)
    state = adapt(package, support, None, args.n)
    output = predict(package, state, query)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
