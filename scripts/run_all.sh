#!/usr/bin/env bash
# Runs every experiment of the paper (Table 2, Figures 6-8). Results are written
# to results/local/*.csv, full logs to results/local/logs/.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

"$SCRIPTS_DIR/run_table2.sh"
"$SCRIPTS_DIR/run_fig6.sh"
"$SCRIPTS_DIR/run_fig7.sh"
"$SCRIPTS_DIR/run_fig8.sh"
