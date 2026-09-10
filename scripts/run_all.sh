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
mkdir -p data
find_and_link () {
  local name="$1"
  # already a working link/file? nothing to do
  if [ -s "data/$name" ]; then return 0; fi
  # broken symlink from an earlier run -> clear it before searching
  [ -L "data/$name" ] && rm -f "data/$name"
  local found
  found="$(find "$HOME" -name "$name" -not -path "*/data/$name" -print -quit 2>/dev/null)"
  if [ -n "$found" ]; then
    ln -sf "$found" "data/$name"
    echo "  found $name -> $found"
  fi
}
find_and_link madrid_train.parquet
find_and_link amsterdam_data.parquet

ls -laL data/madrid_train.parquet data/amsterdam_data.parquet || {
  echo "!! parquet files not found anywhere under \$HOME -- point data/*.parquet"
  echo "   at the challenge data by hand, e.g.:"
  echo "   ln -s /path/to/madrid_train.parquet data/madrid_train.parquet"
  exit 1
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
