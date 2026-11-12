# Benchmarks

| Directory | Contents | In git |
|---|---|---|
| `simple_circuits/` | tiny hand-written circuits (unit tests) | yes |
| `c_circuits/` | ISCAS'85 circuits c17 ... c7552 (unit tests) | yes |
| `large_circuits/` | random circuits `random1-3` (unit tests, quick start); the six circuits of the paper | `random1-3` only |
| `extremely_large_circuits/` | the `_eval` circuits of the paper | no |
| `opentimer_dump/` | OpenTimer dumps the small circuits were converted from | yes |

## Benchmarks of the paper

The 12 benchmarks of the paper (table below) are not included in this repository:

- **Size.** The six circuits and their patterns take about 130 MB, and their
  `_eval` variants about 1.7 GB. Single files are up to 630 MB, far above
  GitHub's 100 MB file limit.
- **Origin.** They are converted from third-party timing-contest benchmarks
  (TAU/ICCAD, as distributed with OpenTimer).

Everything else works without them: the unit tests and the quick start use the
circuits in this directory, and the experiment scripts skip benchmarks they
cannot find. Our results on the 12 benchmarks are in [results/](../results/README.md).

If you have the circuit files, put the six `*_bfs.ckt` files and their `.ptn`
files in `large_circuits/` and generate the `_eval` variants (8 or 16 copies of
a circuit side by side):

```bash
cd benchmark
for c in mgc_edit_dist_iccad mgc_matrix_mult_iccad b19_iccad leon2_iccad_eval; do
  ../tools/scale_benchmark.py 8 large_circuits/${c}_bfs.ckt large_circuits/$c.ptn \
    extremely_large_circuits/8_${c}_bfs.ckt extremely_large_circuits/8_$c.ptn
done
for c in leon3mp_iccad netcard_iccad; do
  ../tools/scale_benchmark.py 16 large_circuits/${c}_bfs.ckt large_circuits/$c.ptn \
    extremely_large_circuits/16_${c}_bfs.ckt extremely_large_circuits/16_$c.ptn
done
```

The experiment scripts look for the benchmarks in this directory; set
`BENCH_DIR` to use another location.

| Paper name | Circuit file | \|V\| |
|---|---|---|
| edit_dist | `large_circuits/mgc_edit_dist_iccad_bfs.ckt` | 164K |
| matrix_mult | `large_circuits/mgc_matrix_mult_iccad_bfs.ckt` | 176K |
| b19 | `large_circuits/b19_iccad_bfs.ckt` | 255K |
| leon2 | `large_circuits/leon2_iccad_eval_bfs.ckt` | 1.6M |
| leon3mp | `large_circuits/leon3mp_iccad_bfs.ckt` | 1.2M |
| netcard | `large_circuits/netcard_iccad_bfs.ckt` | 1.5M |
| edit_dist_eval | `extremely_large_circuits/8_mgc_edit_dist_iccad_bfs.ckt` | 1.3M |
| matrix_mult_eval | `extremely_large_circuits/8_mgc_matrix_mult_iccad_bfs.ckt` | 1.4M |
| b19_eval | `extremely_large_circuits/8_b19_iccad_bfs.ckt` | 2.0M |
| leon2_eval | `extremely_large_circuits/8_leon2_iccad_eval_bfs.ckt` | 12.9M |
| leon3_eval | `extremely_large_circuits/16_leon3mp_iccad_bfs.ckt` | 20.0M |
| netcard_eval | `extremely_large_circuits/16_netcard_iccad_bfs.ckt` | 24.0M |

## Circuit format (`.ckt`)

A combinational circuit as a list of integers separated by whitespace:

```
<#PIs> <#POs> <#gates> <#wires>
<driver> <driver pin> <#sinks> <sink 1> <pin 1> ... <sink n> <pin n>    (one line per wire)
<type of gate 0> <type of gate 1> ...                                     (#gates types)
```

Nodes are numbered PIs first, then POs, then gates: PI `i` is node `i`, PO `j`
is node `#PIs + j`, and gate `g` is node `#PIs + #POs + g`. Pins are ignored;
the fanins of a node are ordered as they appear in the file. Gate types:

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| INV | AND | OR | XOR | NAND | NOR | XNOR | MUX (inputs a, b, s) | BUF |

POs are single-input buffers. Example (`simple_circuits/simple.ckt`):
PIs 0-2, PO 3, gates 4 (INV), 5 (NOR), 6 (OR) and 7 (AND):

```
3
1
4
7
7 0 1 5 1
4 0 1 5 2
5 0 1 3 1
6 0 2 7 2 4 1
1 0 1 6 1
0 0 1 6 2
2 0 1 7 1
0 5 2 1
```

The circuits of the paper were converted from OpenTimer dumps with
`tools/opentimer_to_ckt.py <dump> <out.ckt> bfs`: library cells are mapped to the
gate types above (e.g. AOI/OAI cells and flip-flops become NAND/NOR gates), and
loops are broken by dropping the edges that point back to nodes already visited
in a BFS from the PIs (hence `_bfs`).

## Pattern format (`.ptn`)

```
<#patterns>
<word of PI 0> <word of PI 1> ... <word of PI #PIs-1>      (round 0)
...                                                          (ceil(#patterns / 64) rounds)
```

Patterns are bit-packed: bit `b` of the word of PI `i` in round `r` is the value
of PI `i` in pattern `64 r + b`. One simulation round evaluates 64 patterns.
