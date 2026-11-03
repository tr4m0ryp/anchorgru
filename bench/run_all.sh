#!/usr/bin/env bash
# Run every benchmark in sequence on 2 worker processes of 1 thread each. Resumable: finished splits are cached
# in bench/results/raw/ and skipped. Data downloads go to $ANCHORGRU_DATA (default ../_data).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1
PY=${PY:-python}
mkdir -p bench/logs
for b in physionet2012 elec2 synthetic physionet2012_strongprior physionet2012_gbmprior physionet2012_wd10; do
  echo "== $b $(date -u +%H:%M:%S)"
  "$PY" "bench/$b.py" > "bench/logs/$b.log" 2>&1
done
echo "== done $(date -u +%H:%M:%S)"
