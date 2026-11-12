#!/usr/bin/env bash
# Figure 7: GPU simulation time of SimPart partitions for replication thresholds
# gamma = 2^0 ... 2^11, plus the extremes gamma = 0 (pure DSP) and inf (pure RAP).
#
# usage: scripts/run_fig7.sh [benchmark ...]      (default: netcard)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_setup
if [[ $# -eq 0 ]]; then set -- netcard; fi
select_benchmarks "$@"

for name in "${SELECTED[@]}"; do
  for gamma in 0 1 2 4 8 16 32 64 128 256 512 1024 2048 inf; do
    run fig7 "$name" gamma$gamma -p simpart -k "$K" -g "$gamma" -s graph -r "$RUNS" --verify
  done
done

python3 "$SCRIPTS_DIR/summarize.py" fig7 "$RESULTS_DIR/fig7.csv"
