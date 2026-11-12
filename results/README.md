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
| edit_dist | 164K | 180K | 31.5 | 1109.7 | 184.1 | 17.9 | 4.52 | 1.75 | 1.01726 | 1.01294 | 3.56 | 4.25 | 1.00000 | 1.00000 |
| matrix_mult | 176K | 236K | 43.2 | 767.7 | 118.1 | 26.1 | 6.37 | 1.91 | 1.00207 | 1.00135 | 4.35 | 3.61 | 1.00017 | 1.00022 |
| b19 | 255K | 323K | 83.6 | 3978.7 | 524.3 | 38.6 | 7.69 | 3.97 | 1.00953 | 1.00811 | 6.15 | 5.50 | 1.00000 | 1.00000 |
| leon2 | 1.6M | 1.7M | 522.7 | 9323.1 | 1465.8 | 368.4 | 70.7 | 23.6 | 1.00159 | 1.00145 | 21.8 | 20.0 | 1.00000 | 1.00000 |
| leon3mp | 1.2M | 1.4M | 349.4 | 10233.3 | 1461.9 | 222.9 | 54.1 | 18.0 | 1.00492 | 1.00450 | 18.1 | 16.3 | 1.00000 | 1.00000 |
| netcard | 1.5M | 1.7M | 502.9 | 20720.7 | 3205.4 | 338.5 | 70.1 | 21.5 | 1.02052 | 1.01936 | 21.7 | 18.5 | 1.00000 | 1.00000 |
| edit_dist_eval | 1.3M | 1.4M | 419.7 | 5011.9 | 662.9 | 186.6 | 59.6 | 14.1 | 1.00212 | 1.00150 | 16.0 | 14.5 | 1.00000 | 1.00000 |
| matrix_mult_eval | 1.4M | 1.9M | 551.6 | 3106.2 | 464.6 | 280.1 | 66.9 | 17.6 | 1.00003 | 1.00002 | 18.6 | 15.3 | 1.00000 | 1.00000 |
| b19_eval | 2.0M | 2.6M | 1057.0 | 29569.3 | 3326.0 | 331.0 | 107.9 | 22.8 | 1.00155 | 1.00130 | 27.9 | 21.0 | 1.00000 | 1.00000 |
| leon2_eval | 12.9M | 13.9M | 6615.5 | 60941.9 | 11393.5 | 4249.1 | 1786.2 | 180.2 | 1.00021 | 1.00018 | 184.2 | 142.9 | 1.00000 | 1.00000 |
| leon3_eval | 20.0M | 22.3M | 10195.7 | 88265.8 | 16308.1 | 4077.7 | 2708.6 | 228.8 | 1.00004 | 1.00003 | 280.9 | 212.4 | 1.00000 | 1.00000 |
| netcard_eval | 24.0M | 26.7M | 13089.9 | 712563.6 | 150818.3 | 5207.8 | 3122.7 | 268.2 | 1.00000 | 1.00000 | 345.9 | 242.2 | 1.00000 | 1.00000 |
| **Avg. ratio** | 1.000 | 1.000 | - | 610.89 | 100.15 | 15.51 | 5.55 | 0.98 | 1.005 | 1.004 | 1.000 | 1.000 | 1.000 | 1.000 |

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
| 2 | 951.6 | 21.3 | 58.4 | 20.0 | 0.012 | 0.000 |
| 4 | 1031.2 | 21.2 | 36.7 | 20.1 | 0.042 | 0.000 |
| 8 | 1352.9 | 21.6 | 26.5 | 20.1 | 0.093 | 0.000 |
| 16 | 1438.9 | 21.6 | 23.4 | 20.1 | 0.166 | 0.000 |
| 32 | 2003.9 | 22.0 | 24.1 | 20.1 | 0.281 | 0.000 |
| 64 | 3977.1 | 22.8 | 25.0 | 20.1 | 0.536 | 0.000 |
| 128 | 6636.9 | 23.9 | 26.3 | 20.2 | 0.995 | 0.000 |

**netcard**

| k | RepCut T_Part (ms) | SimPart T_Part (ms) | RepCut T_Sim^G (ms) | SimPart T_Sim^G (ms) | RepCut replication (%) | SimPart replication (%) |
|---|---|---|---|---|---|---|
| 2 | 3396.4 | 21.3 | 50.9 | 18.5 | 0.035 | 0.000 |
| 4 | 2517.5 | 21.3 | 31.7 | 18.5 | 0.646 | 0.000 |
| 8 | 3484.9 | 21.4 | 24.2 | 18.5 | 1.093 | 0.000 |
| 16 | 3602.3 | 21.7 | 21.6 | 18.5 | 2.070 | 0.000 |
| 32 | 4799.6 | 22.0 | 22.7 | 18.5 | 3.306 | 0.000 |
| 64 | 5731.5 | 22.7 | 23.6 | 18.5 | 4.975 | 0.000 |
| 128 | 7892.8 | 23.9 | 24.3 | 18.7 | 7.248 | 0.000 |

## Figure 7: replication threshold γ

**netcard**

| gamma | DSP levels | T_Sim^G (ms) | R|V| |
|---|---|---|---|
| 0 | 124 / 124 | 18.9 | 1.00000 |
| 1 | 123 / 124 | 19.0 | 1.00000 |
| 2 | 121 / 124 | 18.9 | 1.00000 |
| 4 | 118 / 124 | 18.9 | 1.00000 |
| 8 | 115 / 124 | 18.8 | 1.00000 |
| 16 | 111 / 124 | 18.7 | 1.00000 |
| 32 | 109 / 124 | 18.7 | 1.00000 |
| 64 | 106 / 124 | 18.7 | 1.00000 |
| 128 | 102 / 124 | 18.7 | 1.00000 |
| 256 | 98 / 124 | 18.6 | 1.00000 |
| 512 | 94 / 124 | 18.6 | 1.00000 |
| 1024 | 88 / 124 | 18.5 | 1.00000 |
| 2048 | 81 / 124 | 18.8 | 1.00000 |
| inf | 0 / 124 | 45.4 | 1.18041 |

## Figure 8: conditional CUDA graph

| Benchmark | RepCut stream | RepCut graph | SimPart w/o graph (stream) | SimPart w/ graph | graph speedup |
|---|---|---|---|---|---|
| edit_dist | 1.80 | 1.91 | 4.66 | 4.25 | 1.10x |
| matrix_mult | 1.81 | 2.01 | 3.77 | 3.62 | 1.04x |
| b19 | 3.60 | 3.94 | 5.94 | 5.51 | 1.08x |
| leon2 | 23.2 | 23.4 | 21.5 | 20.1 | 1.07x |
| leon3mp | 18.0 | 18.1 | 17.7 | 16.4 | 1.08x |
| netcard | 21.5 | 21.5 | 19.8 | 18.5 | 1.07x |
| edit_dist_eval | 14.0 | 14.4 | 15.5 | 14.5 | 1.07x |
| matrix_mult_eval | 17.4 | 17.5 | 16.0 | 15.3 | 1.05x |
| b19_eval | 23.0 | 23.1 | 21.9 | 21.0 | 1.04x |
| leon2_eval | 177.5 | 181.9 | 144.5 | 142.9 | 1.01x |
| leon3_eval | 228.6 | 228.7 | 214.0 | 212.4 | 1.01x |
| netcard_eval | 268.1 | 268.1 | 243.8 | 242.2 | 1.01x |

GPU simulation times in ms.
