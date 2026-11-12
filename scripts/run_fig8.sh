#!/usr/bin/env bash
# Figure 8: GPU simulation time with and without the conditional CUDA graph,
# for SimPart and RepCut partitions (k = $K, default 16; gamma = $GAMMA, default 1024).
#
# usage: scripts/run_fig8.sh [benchmark ...]      (default: all 12 benchmarks)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_setup
select_benchmarks "$@"

for name in "${SELECTED[@]}"; do
  for sim in stream graph; do
    run fig8 "$name" simpart-$sim -p simpart -k "$K" -g "$GAMMA" -s "$sim" -b "$BATCH" -r "$RUNS" --verify
    run fig8 "$name" repcut-$sim -p repcut -k "$K" -t "$THREADS" -s "$sim" -b "$BATCH" \
      --part-runs 1 --part-warmup 0 --sim-runs "$RUNS" --verify
  done
done

summarize fig8
