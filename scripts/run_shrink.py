"""Shrinkage whitening: can it fix the low-label hole and keep the high-label win?

Full whitening (shrink=0) wins at 50+ labels/class but loses badly at 5-25.
Plain standardisation (shrink=1) wins at 5-25. This sweeps the blend and also
tries the budget-driven heuristic ``shrink_for_shots``.

Usage:  python scripts/run_shrink.py [--quick]
"""

import functools
import sys
import time
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)  # noqa: A001

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapt import shrink_for_shots, zca_whiten
from src.data import build_city
from src.evaluate import few_shot_curve

SEED = 42
QUICK = "--quick" in sys.argv
SHOTS = (5, 25, 50, 100, 200)
GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def main() -> None:
    t0 = time.time()
    ams = build_city(str(ROOT / "data" / "amsterdam_data.parquet"))
    Xa, ya = ams.X, ams.y
    nf = Xa.shape[1]
    n_trials = 5 if QUICK else 15
    print(f"Amsterdam {Xa.shape}   {n_trials} draws/budget\n")

    # fixed-shrink sweep
    table = {}
    for d in GRID:
        fs = few_shot_curve(Xa, ya, shots=SHOTS, n_trials=n_trials,
                            transform=zca_whiten(Xa, shrink=d), seed=SEED)
        table[d] = fs.means
        print(f"  shrink={d:.1f}: " +
              "  ".join(f"{s}:{fs.per_shot[s].mean():.3f}" for s in SHOTS))

    # budget-driven heuristic: different shrink at each budget
    adaptive = []
    for s in SHOTS:
        d = shrink_for_shots(s, nf)
        fs = few_shot_curve(Xa, ya, shots=(s,), n_trials=n_trials,
                            transform=zca_whiten(Xa, shrink=d), seed=SEED)
        adaptive.append((s, d, fs.per_shot[s].mean()))

    print("\n" + "=" * 60)
    print(f"{'budget':>7} | " + " ".join(f"d={d:.1f}" for d in GRID) + " | adaptive")
    print("-" * 60)
    for i, s in enumerate(SHOTS):
        row = " ".join(f"{table[d][i]:.3f}" for d in GRID)
        sa, da, va = adaptive[i]
        best_d = GRID[int(np.argmax([table[d][i] for d in GRID]))]
        print(f"{s:>7} | {row} | {va:.3f} (d={da:.2f})   best fixed d={best_d:.1f}")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    np.savez(out / "shrink_scores.npz", shots=np.array(SHOTS), grid=np.array(GRID),
             sweep=np.stack([table[d] for d in GRID]),
             adaptive=np.array([v for _, _, v in adaptive]))
    print(f"\nwrote results/shrink_scores.npz  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
