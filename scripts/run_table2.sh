#!/usr/bin/env bash
# Table 2: SimPart vs. RepCut on all benchmarks (k = $K, default 16).
#
#   SimPart   partitioning time, GPU simulation time, replication (gamma = $GAMMA, default 1024)
#   RepCut    hypergraph construction, Mt-KaHyPar with 1 and $THREADS threads,
#             CPU simulation with 1 and $THREADS threads, GPU simulation, replication
#
# usage: scripts/run_table2.sh [benchmark ...]      (default: all 12 benchmarks)
#   e.g. GAMMA=2048 scripts/run_table2.sh edit_dist netcard
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_setup
select_benchmarks "$@"

for name in "${SELECTED[@]}"; do
  run table2 "$name" simpart -p simpart -k "$K" -g "$GAMMA" -s graph -b "$BATCH" -r "$RUNS" --verify
  run table2 "$name" repcut-t${THREADS}-gpu -p repcut -k "$K" -t "$THREADS" -s graph -b "$BATCH" \
    --part-runs "$REPCUT_RUNS" --part-warmup 0 --sim-runs "$RUNS" --verify
  run table2 "$name" repcut-t${THREADS}-cpu -p repcut -k "$K" -t "$THREADS" -s cpu \
    --part-runs 1 --part-warmup 0 --sim-runs "$RUNS" --verify
  run table2 "$name" repcut-t1-cpu -p repcut -k "$K" -t 1 -s cpu \
    --part-runs "$REPCUT_RUNS" --part-warmup 0 --sim-runs "$REPCUT_RUNS" --verify
done

python3 "$SCRIPTS_DIR/summarize.py" table2 "$RESULTS_DIR/table2.csv"
