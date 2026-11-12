#!/usr/bin/env bash
# Figure 6: partitioning time, GPU simulation time and replication of SimPart
# and RepCut for k = 2, 4, ..., 128.
#
# usage: scripts/run_fig6.sh [benchmark ...]      (default: leon2 netcard)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_setup
if [[ $# -eq 0 ]]; then set -- leon2 netcard; fi
select_benchmarks "$@"

for name in "${SELECTED[@]}"; do
  for k in 2 4 8 16 32 64 128; do
    run fig6 "$name" simpart-k$k -p simpart -k "$k" -g "$GAMMA" -s graph -b "$BATCH" -r "$RUNS" --verify
    run fig6 "$name" repcut-k$k -p repcut -k "$k" -t "$THREADS" -s graph -b "$BATCH" \
      --part-runs "$REPCUT_RUNS" --part-warmup 0 --sim-runs "$RUNS" --verify
  done
done

summarize fig6
