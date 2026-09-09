"""M8: does the Amsterdam few-shot support/query split have the same spatial
adjacency problem the Madrid CV split had (split_audit.py, 79.7% of adjacent
pairs cross folds)?

For a representative n, draws support samples the same way the sweep does,
then measures what fraction of the SUPPORT set's immediate spatial neighbours
end up in the QUERY set (as opposed to being excluded/unlabelled). If high,
a query pixel's correct answer may be partly "leaking" from a spatially
adjacent, near-identical support pixel rather than genuine transfer.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from organiser_baseline import feature_columns

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "features_cache.pkl"
SHOT_SIZES = [5, 25, 50, 100, 200]
N_TRIALS = 10
SEED = 42


def main():
    with open(CACHE, "rb") as f:
        _, amsterdam = pickle.load(f)
    y = amsterdam["age_class"].to_numpy(dtype=int)
    px = amsterdam["px_key"].to_numpy(dtype=np.int64)
    py = amsterdam["py_key"].to_numpy(dtype=np.int64)
    n_pixels = len(y)

    # adjacency index: for each pixel, its right and up neighbour (if present)
    coord_to_idx = {(int(x), int(yy)): i for i, (x, yy) in enumerate(zip(px, py))}
    neighbour_pairs = []
    for i, (x, yy) in enumerate(zip(px, py)):
        for nb in ((x + 1, yy), (x, yy + 1)):
            j = coord_to_idx.get((int(nb[0]), int(nb[1])))
            if j is not None:
                neighbour_pairs.append((i, j))
    neighbour_pairs = np.array(neighbour_pairs)
    print(f"Amsterdam: {n_pixels:,} pixels, {len(neighbour_pairs):,} orthogonally adjacent pairs", flush=True)

    rng = np.random.default_rng(SEED)
    classes = np.unique(y)
    results = {}
    for n in SHOT_SIZES:
        fractions = []
        for _ in range(N_TRIALS):
            support_idx = []
            for c in classes:
                idx = np.where(y == c)[0]
                support_idx.extend(rng.choice(idx, min(n, len(idx)), replace=False).tolist())
            support_set = set(support_idx)
            query_set_size = n_pixels - len(support_set)

            # count neighbour pairs where one side is support and the other is query
            a_in_support = np.array([a in support_set for a in neighbour_pairs[:, 0]])
            b_in_support = np.array([b in support_set for b in neighbour_pairs[:, 1]])
            support_query_pairs = (a_in_support & ~b_in_support) | (~a_in_support & b_in_support)
            n_support_query_pairs = int(support_query_pairs.sum())

            # fraction of the (small) support set that has >=1 query-side neighbour
            support_with_query_neighbour = set()
            for (a, b), is_sq in zip(neighbour_pairs, support_query_pairs):
                if is_sq:
                    support_with_query_neighbour.add(a if a in support_set else b)
            fraction_support_touching_query = len(support_with_query_neighbour) / len(support_set)
            fractions.append(fraction_support_touching_query)

        results[n] = {
            "mean_fraction_support_pixels_with_a_query_side_neighbour": float(np.mean(fractions)),
            "std": float(np.std(fractions)),
        }
        print(f"  n={n:4d}/class: {np.mean(fractions)*100:.1f}% of support pixels have >=1 immediate neighbour in the query set "
              f"(+/- {np.std(fractions)*100:.1f}pp across {N_TRIALS} trials)", flush=True)

    out_path = ROOT / "outputs" / "amsterdam_split_leakage.json"
    out_path.write_text(json.dumps({
        "total_pixels": n_pixels,
        "total_adjacent_pairs": len(neighbour_pairs),
        "by_shot_size": {str(n): v for n, v in results.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
