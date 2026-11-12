#!/usr/bin/env python3
"""Pretty-print result CSV files written by `simpart --csv`.

usage: scripts/show_csv.py results.csv [more.csv ...] [--columns c1,c2,...]
"""
import argparse
import csv
import sys

DEFAULT_COLUMNS = [
    "circuit", "partitioner", "k", "gamma", "threads", "simulator", "levels", "dsp_levels",
    "part_ms", "grouping_ms", "hybrid_ms", "build_ms", "node_ratio", "edge_ratio",
    "sim_setup_ms", "sim_ms", "sim_std_ms", "verified",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+")
    parser.add_argument("--columns", default=",".join(DEFAULT_COLUMNS))
    args = parser.parse_args()

    rows = []
    for path in args.files:
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        sys.exit("no rows")
    columns = [c for c in args.columns.split(",") if c in rows[0]]
    # Drop columns that are empty in every row.
    columns = [c for c in columns if any(r[c] for r in rows)]
    table = [columns] + [[r[c].replace("_bfs.ckt", "") for c in columns] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(columns))]
    for row in table:
        print("  ".join(cell.rjust(w) if i else cell.ljust(w)
                        for i, (cell, w) in enumerate(zip(row, widths))))


if __name__ == "__main__":
    main()
