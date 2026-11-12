# Shared settings of the experiment scripts (sourced, not executed).
# Every setting can be overridden from the environment, e.g.
#   RUNS=3 BENCH_DIR=/data/simpart-benchmarks scripts/run_table2.sh

set -euo pipefail

SCRIPTS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPTS_DIR/.." && pwd)

SIMPART=${SIMPART:-$REPO_DIR/build/app/simpart}
BENCH_DIR=${BENCH_DIR:-$REPO_DIR/benchmark}
RESULTS_DIR=${RESULTS_DIR:-$REPO_DIR/results}
K=${K:-16}                     # number of partitions (paper default: 16)
GAMMA=${GAMMA:-1024}           # SimPart replication threshold (paper default: 1024 = GPU block size)
RUNS=${RUNS:-10}               # timed repetitions of SimPart and of every simulation
REPCUT_RUNS=${REPCUT_RUNS:-3}  # timed repetitions of RepCut partitioning (Mt-KaHyPar is slow)
THREADS=${THREADS:-16}         # CPU threads of Mt-KaHyPar and of the CPU simulator

# Benchmarks of the paper: <name> <circuit> <patterns>, relative to $BENCH_DIR.
declare -A CIRCUIT PATTERNS
add_benchmark() { CIRCUIT[$1]=$2; PATTERNS[$1]=$3; }
add_benchmark edit_dist        large_circuits/mgc_edit_dist_iccad_bfs.ckt               large_circuits/mgc_edit_dist_iccad.ptn
add_benchmark matrix_mult      large_circuits/mgc_matrix_mult_iccad_bfs.ckt             large_circuits/mgc_matrix_mult_iccad.ptn
add_benchmark b19              large_circuits/b19_iccad_bfs.ckt                         large_circuits/b19_iccad.ptn
add_benchmark leon2            large_circuits/leon2_iccad_eval_bfs.ckt                  large_circuits/leon2_iccad_eval.ptn
add_benchmark leon3mp          large_circuits/leon3mp_iccad_bfs.ckt                     large_circuits/leon3mp_iccad.ptn
add_benchmark netcard          large_circuits/netcard_iccad_bfs.ckt                     large_circuits/netcard_iccad.ptn
add_benchmark edit_dist_eval   extremely_large_circuits/8_mgc_edit_dist_iccad_bfs.ckt   extremely_large_circuits/8_mgc_edit_dist_iccad.ptn
add_benchmark matrix_mult_eval extremely_large_circuits/8_mgc_matrix_mult_iccad_bfs.ckt extremely_large_circuits/8_mgc_matrix_mult_iccad.ptn
add_benchmark b19_eval         extremely_large_circuits/8_b19_iccad_bfs.ckt             extremely_large_circuits/8_b19_iccad.ptn
add_benchmark leon2_eval       extremely_large_circuits/8_leon2_iccad_eval_bfs.ckt      extremely_large_circuits/8_leon2_iccad_eval.ptn
add_benchmark leon3_eval       extremely_large_circuits/16_leon3mp_iccad_bfs.ckt        extremely_large_circuits/16_leon3mp_iccad.ptn
add_benchmark netcard_eval     extremely_large_circuits/16_netcard_iccad_bfs.ckt        extremely_large_circuits/16_netcard_iccad.ptn

BASE_BENCHMARKS=(edit_dist matrix_mult b19 leon2 leon3mp netcard)
EVAL_BENCHMARKS=(edit_dist_eval matrix_mult_eval b19_eval leon2_eval leon3_eval netcard_eval)
ALL_BENCHMARKS=("${BASE_BENCHMARKS[@]}" "${EVAL_BENCHMARKS[@]}")

# Benchmarks given on the command line, or the default list.
select_benchmarks() {
  if [[ $# -gt 0 ]]; then SELECTED=("$@"); else SELECTED=("${ALL_BENCHMARKS[@]}"); fi
  for name in "${SELECTED[@]}"; do
    [[ -n "${CIRCUIT[$name]:-}" ]] || { echo "unknown benchmark '$name'" >&2; exit 2; }
  done
}

check_setup() {
  [[ -x "$SIMPART" ]] || { echo "simpart not found at $SIMPART (run scripts/build.sh first)" >&2; exit 1; }
}

# run <experiment> <benchmark> <tag> [simpart options...]
# Runs simpart on a benchmark, appends a CSV row to results/<experiment>.csv and
# keeps the full output in results/logs/<experiment>/<benchmark>.<tag>.log.
run() {
  local experiment=$1 name=$2 tag=$3
  shift 3
  local circuit=$BENCH_DIR/${CIRCUIT[$name]} patterns=$BENCH_DIR/${PATTERNS[$name]}
  if [[ ! -f "$circuit" || ! -f "$patterns" ]]; then
    echo "skip $name: missing $circuit or $patterns (see benchmark/README.md)" >&2
    return 0
  fi
  local log_dir=$RESULTS_DIR/logs/$experiment
  mkdir -p "$log_dir"
  echo "[$experiment] $name ($tag): simpart $*"
  if ! "$SIMPART" "$circuit" "$patterns" --label "$name" --csv "$RESULTS_DIR/$experiment.csv" "$@" \
      > "$log_dir/$name.$tag.log" 2>&1; then
    echo "  FAILED, see $log_dir/$name.$tag.log" >&2
  fi
}
