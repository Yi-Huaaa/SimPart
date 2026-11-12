#!/usr/bin/env python3
"""Turn the CSV files written by the run_*.sh scripts into Markdown tables
laid out like Table 2 and Figures 6-8 of the paper.

usage: scripts/summarize.py table2|fig6|fig7|fig8 results/<experiment>.csv
"""
import argparse
import csv
import re
import sys

BENCHMARK_ORDER = [
    "edit_dist", "matrix_mult", "b19", "leon2", "leon3mp", "netcard",
    "edit_dist_eval", "matrix_mult_eval", "b19_eval", "leon2_eval", "leon3_eval", "netcard_eval",
]


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(row, key):
    if row is None or row.get(key, "") == "":
        return None
    return float(row[key])


def ordered(names):
    known = [b for b in BENCHMARK_ORDER if b in names]
    return known + sorted(n for n in names if n not in BENCHMARK_ORDER)


def size(n):
    if n is None:
        return "-"
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


def value(x):
    """Times and ratios: integers from 1000 on, three decimals below."""
    if x is None:
        return "-"
    return f"{x:.0f}" if abs(round(x, 3)) >= 1000 else f"{x:.3f}"


def ratio(x):
    """Replication ratios R|V| and R|E|: five decimals."""
    return "-" if x is None else f"{x:.5f}"


def md(cell):
    """Escapes the pipes of a Markdown table cell."""
    return re.sub(r"(?<!\\)\|", r"\\|", cell)


def print_table(header, rows):
    print("| " + " | ".join(md(h) for h in header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        print("| " + " | ".join(md(c) for c in r) + " |")


def last_row(rows, **match):
    """Last row whose columns equal the given values (re-runs replace older rows)."""
    found = None
    for r in rows:
        if all(r.get(k) == str(v) for k, v in match.items()):
            found = r
    return found


def check_comparable(rows, *columns):
    """Exits if a row in a table failed verification, or if the rows compared in
    a table were run with different settings (e.g. after runs with another
    --batch were appended to the same CSV file)."""
    rows = [r for r in rows if r is not None]
    failed = sorted({r["benchmark"] for r in rows if r.get("verified") == "failed"})
    if failed:
        sys.exit(f"summarize.py: results of {', '.join(failed)} failed verification")
    for col in columns:
        values = sorted({r[col] for r in rows})
        if len(values) > 1:
            sys.exit(f"summarize.py: the results to compare mix {col} = {', '.join(values)}; "
                     "rerun them with the same settings or use a separate CSV file")


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def table2(rows):
    names = ordered({r["benchmark"] for r in rows})
    header = ["Benchmark", "|V|", "|E|",
              "RepCut T_Build", "T_Part^C1", "T_Part^Cn", "T_Sim^C1", "T_Sim^Cn", "T_Sim^G",
              "R|V|", "R|E|",
              "SimPart T_Part", "T_Sim^G", "R|V|", "R|E|"]
    out, ratios = [], {k: [] for k in ("part1", "partn", "sim1", "simn", "simg", "rv", "re", "srv", "sre")}
    threads = None
    used = {"all": [], "gpu": [], "simpart": [], "repcut_n": []}
    for b in names:
        sp = last_row(rows, benchmark=b, partitioner="simpart")
        rc_gpu = next((r for r in reversed(rows) if r["benchmark"] == b and r["partitioner"] == "repcut"
                       and r["simulator"] in ("graph", "stream")), None)
        n = rc_gpu["threads"] if rc_gpu else None
        threads = threads or n
        rc_cpun = next((r for r in reversed(rows) if r["benchmark"] == b and r["partitioner"] == "repcut"
                        and r["simulator"] == "cpu" and (r["threads"] == n if n else r["threads"] != "1")),
                       None)
        rc_cpu1 = last_row(rows, benchmark=b, partitioner="repcut", simulator="cpu", threads=1)
        used["all"] += [sp, rc_gpu, rc_cpun, rc_cpu1]
        used["gpu"] += [sp, rc_gpu]
        used["simpart"].append(sp)
        used["repcut_n"] += [rc_gpu, rc_cpun]
        nodes = num(sp or rc_gpu, "nodes")
        edges = num(sp or rc_gpu, "edges")
        t_part_sp, t_sim_sp = num(sp, "part_ms"), num(sp, "sim_ms")
        vals = dict(build=num(rc_gpu, "build_ms"), part1=num(rc_cpu1, "part_ms"), partn=num(rc_gpu, "part_ms"),
                    sim1=num(rc_cpu1, "sim_ms"), simn=num(rc_cpun, "sim_ms"), simg=num(rc_gpu, "sim_ms"),
                    rv=num(rc_gpu, "node_ratio"), re=num(rc_gpu, "edge_ratio"))
        out.append([b, size(nodes), size(edges), value(vals["build"]), value(vals["part1"]),
                    value(vals["partn"]), value(vals["sim1"]), value(vals["simn"]), value(vals["simg"]),
                    ratio(vals["rv"]), ratio(vals["re"]), value(t_part_sp), value(t_sim_sp),
                    ratio(num(sp, "node_ratio")), ratio(num(sp, "edge_ratio"))])

        def rel(x, base):
            return x / base if x is not None and base else None
        ratios["part1"].append(rel(vals["part1"], t_part_sp))
        ratios["partn"].append(rel(vals["partn"], t_part_sp))
        ratios["sim1"].append(rel(vals["sim1"], t_sim_sp))
        ratios["simn"].append(rel(vals["simn"], t_sim_sp))
        ratios["simg"].append(rel(vals["simg"], t_sim_sp))
        ratios["rv"].append(vals["rv"])
        ratios["re"].append(vals["re"])
        ratios["srv"].append(num(sp, "node_ratio"))
        ratios["sre"].append(num(sp, "edge_ratio"))

    check_comparable(used["all"], "k")
    check_comparable(used["gpu"], "batch")
    check_comparable(used["simpart"], "gamma")
    check_comparable(used["repcut_n"], "threads")

    def avg(key, fmt=value):
        return fmt(mean(ratios[key]))
    out.append(["**Avg. ratio**", "1.000", "1.000", "-", avg("part1"), avg("partn"), avg("sim1"),
                avg("simn"), avg("simg"), avg("rv", ratio), avg("re", ratio), "1.000", "1.000",
                avg("srv", ratio), avg("sre", ratio)])
    n = threads or "n"
    header = [h.replace("^Cn", f"^C{n}") for h in header]
    print(f"Times in ms. Ratios of partitioning/simulation times are relative to SimPart's T_Part/T_Sim^G.\n")
    print_table(header, out)


def fig6(rows):
    for b in ordered({r["benchmark"] for r in rows}):
        ks = sorted({int(r["k"]) for r in rows if r["benchmark"] == b})
        table, rcs, sps = [], [], []
        for k in ks:
            rc = last_row(rows, benchmark=b, partitioner="repcut", k=k)
            sp = last_row(rows, benchmark=b, partitioner="simpart", k=k)
            rcs.append(rc)
            sps.append(sp)
            rep = lambda r: "-" if r is None else value((float(r["node_ratio"]) - 1) * 100)
            table.append([str(k), value(num(rc, "part_ms")), value(num(sp, "part_ms")),
                          value(num(rc, "sim_ms")), value(num(sp, "sim_ms")), rep(rc), rep(sp)])
        check_comparable(rcs + sps, "batch")
        check_comparable(sps, "gamma")
        check_comparable(rcs, "threads")
        print(f"\n**{b}**\n")
        print_table(["k", "RepCut T_Part (ms)", "SimPart T_Part (ms)", "RepCut T_Sim^G (ms)",
                     "SimPart T_Sim^G (ms)", "RepCut replication (%)", "SimPart replication (%)"], table)


def fig7(rows):
    def gamma_key(g):
        return float("inf") if g == "inf" else float(g)
    for b in ordered({r["benchmark"] for r in rows}):
        gammas = sorted({r["gamma"] for r in rows if r["benchmark"] == b}, key=gamma_key)
        table, used = [], []
        for g in gammas:
            r = last_row(rows, benchmark=b, partitioner="simpart", gamma=g)
            used.append(r)
            table.append([g, r["dsp_levels"] + " / " + r["levels"], value(num(r, "sim_ms")),
                          ratio(num(r, "node_ratio"))])
        check_comparable(used, "k", "batch")
        print(f"\n**{b}**\n")
        print_table(["gamma", "DSP levels", "T_Sim^G (ms)", "R|V|"], table)


def fig8(rows):
    table, used = [], {"simpart": [], "repcut": []}
    for b in ordered({r["benchmark"] for r in rows}):
        cell = {}
        for p in ("simpart", "repcut"):
            for s in ("stream", "graph"):
                r = last_row(rows, benchmark=b, partitioner=p, simulator=s)
                used[p].append(r)
                cell[(p, s)] = num(r, "sim_ms")
        speedup = (cell[("simpart", "stream")] / cell[("simpart", "graph")]
                   if cell[("simpart", "stream")] and cell[("simpart", "graph")] else None)
        table.append([b, value(cell[("repcut", "stream")]), value(cell[("repcut", "graph")]),
                      value(cell[("simpart", "stream")]), value(cell[("simpart", "graph")]),
                      "-" if speedup is None else value(speedup) + "x"])
    check_comparable(used["simpart"] + used["repcut"], "k", "batch")
    check_comparable(used["simpart"], "gamma")
    check_comparable(used["repcut"], "threads")
    print_table(["Benchmark", "RepCut stream", "RepCut graph", "SimPart w/o graph (stream)",
                 "SimPart w/ graph", "SimPart graph speedup"], table)
    print("\nGPU simulation times in ms.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment", choices=["table2", "fig6", "fig7", "fig8"])
    parser.add_argument("csv")
    args = parser.parse_args()
    rows = load(args.csv)
    if not rows:
        sys.exit("no results in " + args.csv)
    {"table2": table2, "fig6": fig6, "fig7": fig7, "fig8": fig8}[args.experiment](rows)


if __name__ == "__main__":
    main()
