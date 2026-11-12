# Results

Results of the experiment scripts (`scripts/run_*.sh`) on the machine below, with
the default settings of the paper (k = 16, γ = 1024) unless stated otherwise.

- `rtx-a4000/<experiment>.csv`: one row per run, as written by `simpart --csv`
- `rtx-a4000/logs/<experiment>/`: the full output of every run
- `rtx-a4000/run.out`: the console output of the scripts (commands run, summary tables)
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

- Every benchmark is simulated with 1024 input patterns (16 rounds of 64). The
  GPU simulations run 4 rounds per pass (`--batch 4`, the default).
- SimPart partitioning, the GPU simulations and the 16-thread CPU simulations:
  mean of 10 timed runs after one warm-up run. RepCut partitioning (Mt-KaHyPar)
  and the 1-thread CPU simulations: mean of 3 timed runs (partitioning without
  warm-up).
- Every simulation was checked against a sequential CPU simulation (all PO
  values of all rounds and all node values of the last round). All passed.
- All times are in milliseconds. With the default settings, `scripts/run_table2.sh`
  took 69 minutes on this machine, most of it in single-threaded
  Mt-KaHyPar on the largest benchmarks.

## Table 2: SimPart vs. RepCut

Times in ms. Ratios of partitioning/simulation times are relative to SimPart's T_Part/T_Sim^G.

| Benchmark | \|V\| | \|E\| | RepCut T_Build | T_Part^C1 | T_Part^C16 | T_Sim^C1 | T_Sim^C16 | T_Sim^G | R\|V\| | R\|E\| | SimPart T_Part | T_Sim^G | R\|V\| | R\|E\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| edit_dist | 164K | 180K | 33.434 | 1116 | 170.741 | 17.871 | 4.696 | 0.669 | 1.01690 | 1.01255 | 3.536 | 1.025 | 1.00000 | 1.00000 |
| matrix_mult | 176K | 236K | 42.803 | 765.707 | 117.712 | 25.650 | 5.966 | 0.832 | 1.00183 | 1.00126 | 4.316 | 1.074 | 1.00017 | 1.00022 |
| b19 | 255K | 323K | 83.797 | 3996 | 530.792 | 39.053 | 7.573 | 1.009 | 1.01186 | 1.01007 | 6.097 | 1.315 | 1.00000 | 1.00000 |
| leon2 | 1.6M | 1.7M | 521.788 | 9364 | 1446 | 378.316 | 73.497 | 4.041 | 1.00168 | 1.00154 | 21.060 | 3.308 | 1.00000 | 1.00000 |
| leon3mp | 1.2M | 1.4M | 352.969 | 10268 | 1656 | 239.831 | 49.396 | 3.036 | 1.00502 | 1.00453 | 17.635 | 2.814 | 1.00000 | 1.00000 |
| netcard | 1.5M | 1.7M | 503.743 | 20651 | 3258 | 338.787 | 69.662 | 3.530 | 1.02115 | 1.01994 | 21.205 | 3.252 | 1.00000 | 1.00000 |
| edit_dist_eval | 1.3M | 1.4M | 396.505 | 4979 | 641.768 | 182.459 | 46.077 | 2.994 | 1.00200 | 1.00149 | 15.630 | 2.673 | 1.00000 | 1.00000 |
| matrix_mult_eval | 1.4M | 1.9M | 552.493 | 3080 | 458.153 | 271.798 | 70.176 | 3.789 | 1.00004 | 1.00002 | 18.281 | 3.569 | 1.00000 | 1.00000 |
| b19_eval | 2.0M | 2.6M | 1067 | 29631 | 3738 | 333.774 | 111.196 | 4.863 | 1.00167 | 1.00143 | 27.181 | 4.562 | 1.00000 | 1.00000 |
| leon2_eval | 12.9M | 13.9M | 6555 | 60959 | 11425 | 4204 | 1785 | 30.047 | 1.00020 | 1.00018 | 180.105 | 28.068 | 1.00000 | 1.00000 |
| leon3_eval | 20.0M | 22.3M | 10101 | 87750 | 16182 | 3834 | 2715 | 46.085 | 1.00005 | 1.00004 | 273.391 | 43.444 | 1.00000 | 1.00000 |
| netcard_eval | 24.0M | 26.7M | 13041 | 708765 | 148824 | 5009 | 3136 | 54.503 | 1.00000 | 1.00000 | 335.180 | 49.760 | 1.00000 | 1.00000 |
| **Avg. ratio** | 1.000 | 1.000 | - | 625.019 | 104.093 | 77.589 | 27.290 | 1.005 | 1.00520 | 1.00442 | 1.000 | 1.000 | 1.00001 | 1.00002 |

| Column | Meaning |
|---|---|
| T_Build | RepCut proxy hypergraph construction (levelization, fanin cones, clusters), 1 thread |
| T_Part^C1, T_Part^C16 | Mt-KaHyPar partitioning time with 1 and 16 threads |
| T_Sim^C1, T_Sim^C16 | CPU simulation of the RepCut partition with 1 and 16 OpenMP threads (one partition per thread) |
| T_Sim^G | GPU simulation, conditional CUDA graph, 4 rounds per pass (RepCut partitions use the same kernels: a single RAP kernel per pass) |
| T_Part (SimPart) | GPU majority-based grouping and levelization plus DSP-RAP hybrid partitioning, including graph upload/download |
| R\|V\|, R\|E\| | simulated nodes and fanin edges (replicas included) relative to the circuit size |
| Avg. ratio | mean over the benchmarks of the time relative to SimPart's T_Part (partitioning columns) or T_Sim^G (simulation columns) |

## Figure 6: number of partitions k

**leon2**

| k | RepCut T_Part (ms) | SimPart T_Part (ms) | RepCut T_Sim^G (ms) | SimPart T_Sim^G (ms) | RepCut replication (%) | SimPart replication (%) |
|---|---|---|---|---|---|---|
| 2 | 993.296 | 20.670 | 15.920 | 3.304 | 0.013 | 0.000 |
| 4 | 1020 | 20.808 | 8.578 | 3.304 | 0.043 | 0.000 |
| 8 | 1255 | 20.837 | 4.915 | 3.309 | 0.091 | 0.000 |
| 16 | 1445 | 21.137 | 4.049 | 3.310 | 0.171 | 0.000 |
| 32 | 2397 | 21.435 | 3.567 | 3.306 | 0.293 | 0.000 |
| 64 | 2828 | 22.037 | 3.246 | 3.305 | 0.566 | 0.000 |
| 128 | 6377 | 23.419 | 2.877 | 3.338 | 1.010 | 0.000 |

**netcard**

| k | RepCut T_Part (ms) | SimPart T_Part (ms) | RepCut T_Sim^G (ms) | SimPart T_Sim^G (ms) | RepCut replication (%) | SimPart replication (%) |
|---|---|---|---|---|---|---|
| 2 | 3508 | 20.651 | 14.999 | 3.252 | 0.028 | 0.000 |
| 4 | 2803 | 20.919 | 8.156 | 3.249 | 0.598 | 0.000 |
| 8 | 3198 | 21.041 | 4.369 | 3.249 | 1.049 | 0.000 |
| 16 | 3932 | 21.177 | 3.510 | 3.250 | 2.037 | 0.000 |
| 32 | 4483 | 21.611 | 3.399 | 3.251 | 3.236 | 0.000 |
| 64 | 6002 | 22.200 | 3.199 | 3.250 | 4.913 | 0.000 |
| 128 | 8050 | 23.293 | 2.945 | 3.278 | 7.115 | 0.000 |

## Figure 7: replication threshold γ

**netcard**

| gamma | DSP levels | T_Sim^G (ms) | R\|V\| |
|---|---|---|---|
| 0 | 124 / 124 | 3.305 | 1.00000 |
| 1 | 123 / 124 | 3.307 | 1.00000 |
| 2 | 121 / 124 | 3.304 | 1.00000 |
| 4 | 118 / 124 | 3.291 | 1.00000 |
| 8 | 115 / 124 | 3.281 | 1.00000 |
| 16 | 111 / 124 | 3.275 | 1.00000 |
| 32 | 109 / 124 | 3.272 | 1.00000 |
| 64 | 106 / 124 | 3.258 | 1.00000 |
| 128 | 102 / 124 | 3.242 | 1.00000 |
| 256 | 98 / 124 | 3.235 | 1.00000 |
| 512 | 94 / 124 | 3.223 | 1.00000 |
| 1024 | 88 / 124 | 3.247 | 1.00000 |
| 2048 | 81 / 124 | 3.369 | 1.00000 |
| inf | 0 / 124 | 10.555 | 1.18041 |

## Figure 8: conditional CUDA graph

| Benchmark | RepCut stream | RepCut graph | SimPart w/o graph (stream) | SimPart w/ graph | SimPart graph speedup |
|---|---|---|---|---|---|
| edit_dist | 0.623 | 0.668 | 1.129 | 1.025 | 1.101x |
| matrix_mult | 0.826 | 0.823 | 1.121 | 1.075 | 1.043x |
| b19 | 0.975 | 0.960 | 1.419 | 1.316 | 1.078x |
| leon2 | 3.865 | 3.777 | 3.635 | 3.311 | 1.098x |
| leon3mp | 2.821 | 2.950 | 3.101 | 2.813 | 1.102x |
| netcard | 3.408 | 3.520 | 3.529 | 3.250 | 1.086x |
| edit_dist_eval | 2.947 | 2.997 | 2.898 | 2.673 | 1.084x |
| matrix_mult_eval | 3.785 | 3.822 | 3.749 | 3.574 | 1.049x |
| b19_eval | 4.858 | 4.810 | 4.754 | 4.557 | 1.043x |
| leon2_eval | 30.455 | 30.453 | 28.512 | 28.067 | 1.016x |
| leon3_eval | 45.993 | 46.019 | 43.888 | 43.473 | 1.010x |
| netcard_eval | 54.403 | 54.451 | 50.160 | 49.770 | 1.008x |

GPU simulation times in ms.
