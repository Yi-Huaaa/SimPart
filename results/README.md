# Results

Results of the experiment scripts (`scripts/run_*.sh`) on the machine below, with
the default settings of the paper (k = 16, γ = 1024) unless stated otherwise.

- `rtx-a4000/<experiment>.csv`: one row per run, as written by `simpart --csv`
- `rtx-a4000/logs/<experiment>/`: the full output of every run
- The tables below were produced with `scripts/summarize.py <experiment> rtx-a4000/<experiment>.csv`.

## Machine

| | |
|---|---|
| GPU | NVIDIA RTX A4000 (Ampere, sm_86, 48 SMs, 16 GB), driver 550.163.01 |
| CPU | Intel Core i5-13500 (14 cores, 20 threads, up to 4.8 GHz, 24 MB L3) |
| Memory | 125 GiB |
| OS | Ubuntu 24.04.4 LTS, Linux 6.8.0-138 |
| Compilers | CUDA 12.3.2 (nvcc 12.3.107), GCC 12.4.0, CMake 3.28.3 |
| Mt-KaHyPar | v1.4-92-g39e06c75 |
| Build | `scripts/build.sh` (Release, `-O3`, sm_86) |

## Settings

- Every benchmark is simulated with 1024 input patterns (16 rounds of 64).
- SimPart partitioning and every simulation: mean of 10 timed runs after one
  warm-up run. RepCut partitioning (Mt-KaHyPar): mean of 3 runs without warm-up,
  1 run for leon3_eval and netcard_eval.
- Every simulation was checked against a sequential CPU simulation (all PO
  values of all rounds and all node values of the last round). All passed.
- All times are in milliseconds. With the default settings, `scripts/run_table2.sh`
  takes about 70 minutes on this machine, most of it in single-threaded
  Mt-KaHyPar on the largest benchmarks (about 12 minutes per run on netcard_eval).

## Table 2: SimPart vs. RepCut

Times in ms. Ratios of partitioning/simulation times are relative to SimPart's T_Part/T_Sim^G.

| Benchmark | \|V\| | \|E\| | RepCut T_Build | T_Part^C1 | T_Part^C16 | T_Sim^C1 | T_Sim^C16 | T_Sim^G | R\|V\| | R\|E\| | SimPart T_Part | T_Sim^G | R\|V\| | R\|E\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edit_dist | 164K | 180K | 31.520 | 1110 | 184.063 | 17.915 | 4.518 | 1.749 | 1.01726 | 1.01294 | 3.557 | 4.251 | 1.00000 | 1.00000 |
| matrix_mult | 176K | 236K | 43.236 | 767.733 | 118.146 | 26.107 | 6.372 | 1.914 | 1.00207 | 1.00135 | 4.349 | 3.612 | 1.00017 | 1.00022 |
| b19 | 255K | 323K | 83.555 | 3979 | 524.284 | 38.568 | 7.690 | 3.973 | 1.00953 | 1.00811 | 6.153 | 5.498 | 1.00000 | 1.00000 |
| leon2 | 1.6M | 1.7M | 522.652 | 9323 | 1466 | 368.402 | 70.739 | 23.617 | 1.00159 | 1.00145 | 21.750 | 20.045 | 1.00000 | 1.00000 |
| leon3mp | 1.2M | 1.4M | 349.406 | 10233 | 1462 | 222.915 | 54.137 | 17.980 | 1.00492 | 1.00450 | 18.133 | 16.338 | 1.00000 | 1.00000 |
| netcard | 1.5M | 1.7M | 502.917 | 20721 | 3205 | 338.509 | 70.087 | 21.542 | 1.02052 | 1.01936 | 21.663 | 18.492 | 1.00000 | 1.00000 |
| edit_dist_eval | 1.3M | 1.4M | 419.683 | 5012 | 662.922 | 186.626 | 59.574 | 14.069 | 1.00212 | 1.00150 | 15.952 | 14.543 | 1.00000 | 1.00000 |
| matrix_mult_eval | 1.4M | 1.9M | 551.588 | 3106 | 464.598 | 280.144 | 66.899 | 17.621 | 1.00003 | 1.00002 | 18.638 | 15.256 | 1.00000 | 1.00000 |
| b19_eval | 2.0M | 2.6M | 1057 | 29569 | 3326 | 330.972 | 107.910 | 22.796 | 1.00155 | 1.00130 | 27.895 | 21.032 | 1.00000 | 1.00000 |
| leon2_eval | 12.9M | 13.9M | 6616 | 60942 | 11393 | 4249 | 1786 | 180.235 | 1.00021 | 1.00018 | 184.168 | 142.877 | 1.00000 | 1.00000 |
| leon3_eval | 20.0M | 22.3M | 10196 | 88266 | 16308 | 4078 | 2709 | 228.778 | 1.00004 | 1.00003 | 280.885 | 212.359 | 1.00000 | 1.00000 |
| netcard_eval | 24.0M | 26.7M | 13090 | 712564 | 150818 | 5208 | 3123 | 268.227 | 1.00000 | 1.00000 | 345.888 | 242.165 | 1.00000 | 1.00000 |
| **Avg. ratio** | 1.000 | 1.000 | - | 610.894 | 100.148 | 15.514 | 5.552 | 0.980 | 1.00499 | 1.00423 | 1.000 | 1.000 | 1.00001 | 1.00002 |

| Column | Meaning |
|---|---|
| T_Build | RepCut proxy hypergraph construction (levelization, fanin cones, clusters), 1 thread |
| T_Part^C1, T_Part^C16 | Mt-KaHyPar partitioning time with 1 and 16 threads |
| T_Sim^C1, T_Sim^C16 | CPU simulation of the RepCut partition with 1 and 16 OpenMP threads (one partition per thread) |
| T_Sim^G | GPU simulation, conditional CUDA graph (RepCut partitions use the same kernels: a single RAP kernel per round) |
| T_Part (SimPart) | GPU majority-based grouping and levelization plus DSP-RAP hybrid partitioning, including graph upload/download |
| R\|V\|, R\|E\| | simulated nodes and fanin edges (replicas included) relative to the circuit size |
| Avg. ratio | mean over the benchmarks of the time relative to SimPart's T_Part (partitioning columns) or T_Sim^G (simulation columns) |

## Figure 6: number of partitions k

**leon2**

| k | RepCut T_Part (ms) | SimPart T_Part (ms) | RepCut T_Sim^G (ms) | SimPart T_Sim^G (ms) | RepCut replication (%) | SimPart replication (%) |
|---|---|---|---|---|---|---|
| 2 | 951.624 | 21.259 | 58.373 | 20.040 | 0.012 | 0.000 |
| 4 | 1031 | 21.191 | 36.747 | 20.060 | 0.042 | 0.000 |
| 8 | 1353 | 21.561 | 26.519 | 20.075 | 0.093 | 0.000 |
| 16 | 1439 | 21.638 | 23.448 | 20.076 | 0.166 | 0.000 |
| 32 | 2004 | 22.042 | 24.118 | 20.077 | 0.281 | 0.000 |
| 64 | 3977 | 22.824 | 25.032 | 20.075 | 0.536 | 0.000 |
| 128 | 6637 | 23.863 | 26.302 | 20.204 | 0.995 | 0.000 |

**netcard**

| k | RepCut T_Part (ms) | SimPart T_Part (ms) | RepCut T_Sim^G (ms) | SimPart T_Sim^G (ms) | RepCut replication (%) | SimPart replication (%) |
|---|---|---|---|---|---|---|
| 2 | 3396 | 21.281 | 50.865 | 18.515 | 0.035 | 0.000 |
| 4 | 2517 | 21.254 | 31.713 | 18.504 | 0.646 | 0.000 |
| 8 | 3485 | 21.433 | 24.161 | 18.525 | 1.093 | 0.000 |
| 16 | 3602 | 21.693 | 21.550 | 18.514 | 2.070 | 0.000 |
| 32 | 4800 | 21.981 | 22.709 | 18.521 | 3.306 | 0.000 |
| 64 | 5731 | 22.651 | 23.625 | 18.518 | 4.975 | 0.000 |
| 128 | 7893 | 23.909 | 24.265 | 18.668 | 7.248 | 0.000 |

## Figure 7: replication threshold γ

**netcard**

| gamma | DSP levels | T_Sim^G (ms) | R\|V\| |
|---|---|---|---|
| 0 | 124 / 124 | 18.945 | 1.00000 |
| 1 | 123 / 124 | 18.967 | 1.00000 |
| 2 | 121 / 124 | 18.908 | 1.00000 |
| 4 | 118 / 124 | 18.858 | 1.00000 |
| 8 | 115 / 124 | 18.808 | 1.00000 |
| 16 | 111 / 124 | 18.749 | 1.00000 |
| 32 | 109 / 124 | 18.721 | 1.00000 |
| 64 | 106 / 124 | 18.672 | 1.00000 |
| 128 | 102 / 124 | 18.655 | 1.00000 |
| 256 | 98 / 124 | 18.598 | 1.00000 |
| 512 | 94 / 124 | 18.553 | 1.00000 |
| 1024 | 88 / 124 | 18.490 | 1.00000 |
| 2048 | 81 / 124 | 18.753 | 1.00000 |
| inf | 0 / 124 | 45.436 | 1.18041 |

## Figure 8: conditional CUDA graph

| Benchmark | RepCut stream | RepCut graph | SimPart w/o graph (stream) | SimPart w/ graph | SimPart graph speedup |
|---|---|---|---|---|---|
| edit_dist | 1.797 | 1.914 | 4.661 | 4.249 | 1.097x |
| matrix_mult | 1.810 | 2.010 | 3.770 | 3.619 | 1.042x |
| b19 | 3.602 | 3.935 | 5.939 | 5.514 | 1.077x |
| leon2 | 23.183 | 23.377 | 21.533 | 20.072 | 1.073x |
| leon3mp | 17.996 | 18.061 | 17.652 | 16.358 | 1.079x |
| netcard | 21.511 | 21.464 | 19.796 | 18.508 | 1.070x |
| edit_dist_eval | 14.037 | 14.449 | 15.549 | 14.542 | 1.069x |
| matrix_mult_eval | 17.352 | 17.517 | 16.026 | 15.261 | 1.050x |
| b19_eval | 22.968 | 23.059 | 21.884 | 21.022 | 1.041x |
| leon2_eval | 177.502 | 181.888 | 144.474 | 142.909 | 1.011x |
| leon3_eval | 228.620 | 228.705 | 214.021 | 212.441 | 1.007x |
| netcard_eval | 268.061 | 268.080 | 243.766 | 242.191 | 1.007x |

GPU simulation times in ms.
