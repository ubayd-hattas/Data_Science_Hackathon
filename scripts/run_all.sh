#!/usr/bin/env bash
# Reproduce the full Madrid -> Amsterdam transfer result end to end.
#
#   bash scripts/run_all.sh            # use cached Madrid CV if present (fast reruns)
#   bash scripts/run_all.sh --fresh    # recompute Madrid CV from scratch (~10 min)
#
# Notebook 5 is the whole pipeline: it builds the 108 features from src/,
# runs the Madrid cross-validation, the zero-shot class-conditional CORAL
# transfer, and the five few-shot budgets, and writes the curve + table.
set -euo pipefail

cd "$(dirname "$0")/.."          # repo root, wherever this script lives

if [ "${1:-}" = "--fresh" ]; then
  rm -f results/madrid_cv_cache.json
  echo "cleared Madrid CV cache -- it will recompute"
fi

echo "== data =="
ls -laL data/madrid_train.parquet data/amsterdam_data.parquet || {
  echo "!! parquet files missing from data/ -- fix the symlinks first"; exit 1;
}

echo
echo "== config =="
if [ -f results/overnight_best.json ]; then
  echo "results/overnight_best.json present -> tuned 108-feature config"
else
  echo "!! results/overnight_best.json missing -> notebook will fall back to 60 features"
fi

echo
echo "== executing notebooks/5-Transfer.ipynb (takes a while) =="
( cd notebooks && jupyter nbconvert --to notebook --execute --inplace 5-Transfer.ipynb )

echo
echo "== results =="
( cd notebooks && jupyter nbconvert --to markdown --stdout 5-Transfer.ipynb 2>/dev/null ) \
  | grep -E 'using tuned config|quick mode:|Madrid +\(7|Amsterdam +\(2|Madrid CV macro-F1|Madrid CV [0-9]|zero-shot  (raw|CORAL|class)|n= *[0-9]+  shrink' \
  || echo "(could not parse outputs -- open notebooks/5-Transfer.ipynb to check)"

echo
echo "done. Expected: Madrid CV ~0.664 | zero-shot cc-CORAL ~0.65 |"
echo "few-shot 5/25/50/100/200 ~ 0.66 / 0.68 / 0.70 / 0.72 / 0.74"
